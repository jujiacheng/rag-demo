"""pymilvus 新旧 API 不兼容的临时桥接（import 时自动生效）。

背景：
    langchain-milvus（所有已发布版本）内部同时使用 pymilvus 的两套 API——
    新版 `MilvusClient`（用于建集合/插入/检索）和旧版 ORM `Collection`/`utility`
    （用于读 schema、索引）。`MilvusClient` 把连接挂在它自己生成的别名（形如
    `cm-<数字>`）上，但**不会**把该别名注册进 ORM 的 `connections` 单例；
    而旧版 `Collection(using=该别名)` / `utility.has_collection(using=该别名)`
    恰恰从 `connections` 单例取连接，于是抛
    `ConnectionNotExistException: should create connection first.`。

    这个问题对所有后端（milvus-lite / 完整服务器）都成立，与 MILVUS_URI 无关，
    也不是 langchain-milvus 某个版本的回归——降级 langchain-milvus（0.1.9 /
    0.2.2 / 0.3.x 均验证过）无效，因为它们都用同样的混合写法。

修复思路：
    `Milvus.__init__` 的执行顺序是：
        self._milvus_client = MilvusClient(...)   # 生成 cm-xxxx 别名
        self.alias = self.client._using
        ... drop_old 逻辑 ...
        self._init(...)                            # 这里调用 self.col -> Collection(...) 会崩
    我们在 `Milvus._init` 外面套一层：在原 `_init` 运行**之前**，用同一个 uri
    调一次 `connections.connect(alias=self.alias, uri=...)`，把别名显式注册进 ORM
    单例。这样无论 drop_old 是否重建、集合是否已存在，后续 `Collection(using=alias)`
    都能找到连接。

    只要应用里 `from app.milvus_compat import patch_langchain_milvus`（或直接
    `import app.milvus_compat`），patch 就全局生效；所有 `Milvus(...)` 构造点
    （ingest、vectorstore 等）无需各自处理。

注意：
    一旦 langchain-milvus / pymilvus 上游彻底统一了两套 API（让 `MilvusClient`
    自动向 `connections` 单例注册别名，或 langchain-milvus 不再混用 ORM
    `Collection`），这个文件就可以整个删除。
"""
from __future__ import annotations

import langchain_milvus.vectorstores.milvus as _lm
from pymilvus.orm.connections import connections

_patched = False


def patch_langchain_milvus() -> None:
    """给 `langchain_milvus.Milvus._init` 套一层别名桥接。幂等，可重复调用。"""
    global _patched
    if _patched:
        return

    _orig_init = _lm.Milvus._init

    def _patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # 在原 _init 触发 self.col -> Collection(using=self.alias) 之前，
        # 把别名注册进 ORM connections 单例。
        alias = getattr(self, "alias", None)
        uri = self._connection_args.get("uri") if self._connection_args else None
        if alias and uri and alias not in connections._alias_handlers:
            connect_uri = uri
            # 本地 milvus-lite：MilvusClient.__init__ 已经启动了 server 并持锁，
            # 这里绝不能再拿 .db uri 去 connections.connect——那会触发第二次
            # milvus-lite 启动，flock 冲突，server 启动失败但锁残留，后续每次
            # 请求都死循环。改成向 server_manager 复用已启动的 http port，
            # 用 http uri 走 ORM 的正常 gRPC 连接路径。
            if isinstance(uri, str) and uri.endswith(".db"):
                try:
                    from milvus_lite.server_manager import (  # noqa: PLC0415
                        server_manager_instance,
                    )
                    local_uri = server_manager_instance.start_and_get_uri(uri)
                except ImportError:
                    local_uri = None
                if local_uri:
                    connect_uri = local_uri
                else:
                    # server 还没启动或启动失败——此时让 _init 自己抛
                    # ConnectionNotExistException，比触发锁冲突死循环好。
                    connect_uri = None
            if connect_uri is not None:
                connections.connect(alias=alias, uri=connect_uri)
        return _orig_init(self, *args, **kwargs)

    _lm.Milvus._init = _patched_init
    _patched = True


# import 本模块即自动应用 patch，调用方无需显式调用。
patch_langchain_milvus()
