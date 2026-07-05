"""原生 MilvusClient 用法演示（不依赖 LangChain）。

这份文件对应分享文档第 4 节「动手：Milvus 基础操作」，完整演示：
  连接 -> 建 schema（含 2.6 nullable）-> 建 HNSW 索引 -> 插入 -> 加载 -> 检索

可以直接运行：
    uv run python -m app.milvus_raw

注意：milvus-lite 底层只支持 Flat 索引，运行时会自动降级，
      但 API 写法与完整服务器版完全一致，迁移只需改 MILVUS_URI。
"""
from __future__ import annotations

from pymilvus import DataType, MilvusClient

from app.config import settings


def build_client() -> MilvusClient:
    """连接 Milvus。本地 milvus-lite（文件路径）或完整服务器（http uri）都支持。"""
    return MilvusClient(uri=settings.milvus_uri_resolved)


def create_demo_collection(client: MilvusClient, name: str = "milvus_raw_demo") -> None:
    """建一个演示集合：自增 id + 稠密向量 + 文本（可空，演示 2.6 nullable）。"""
    if client.has_collection(name):
        client.drop_collection(name)
        print(f"[drop] 旧集合 {name} 已删除")

    # 1) 定义 schema —— 用 MilvusClient 的静态工厂
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=8)  # demo 用 8 维
    schema.add_field(
        "text", DataType.VARCHAR, max_length=512, nullable=True  # 2.6 新特性：可空字段
    )

    # 2) 准备索引参数 —— 推荐用 HNSW 做低延迟近邻检索
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",            # milvus-lite 下会自动降级为 Flat
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256},
    )

    # 3) 一次性建集合 + 建索引
    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )
    print(f"[create] 集合 {name} 创建成功（HNSW / COSINE / dim=8）")


def insert_demo_data(client: MilvusClient, name: str = "milvus_raw_demo") -> None:
    """插入若干条演示数据，text 字段故意留一条空值。"""
    data = [
        {"vector": [0.12, 0.45, 0.78, 0.23, 0.91, 0.34, 0.56, 0.11], "text": "Milvus 是向量数据库"},
        {"vector": [0.88, 0.21, 0.67, 0.45, 0.12, 0.89, 0.33, 0.77], "text": "HNSW 是常用索引"},
        {"vector": [0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55], "text": "RAG = 检索增强生成"},
        {"vector": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80], "text": None},  # nullable
    ]
    client.insert(collection_name=name, data=data)
    print(f"[insert] 写入 {len(data)} 条数据")


def search_demo(client: MilvusClient, name: str = "milvus_raw_demo") -> None:
    """检索：用一条 query 向量找 top-3 最相似的记录。"""
    client.load_collection(name)

    query = [0.50, 0.50, 0.60, 0.40, 0.50, 0.50, 0.50, 0.50]
    results = client.search(
        collection_name=name,
        data=[query],
        anns_field="vector",
        limit=3,
        output_fields=["text"],
        search_params={"params": {"ef": 64}},  # HNSW 查询参数；milvus-lite 下被忽略
    )

    print("[search] top-3 结果：")
    for hit in results[0]:
        distance = hit.get("distance")
        text = hit.get("entity", {}).get("text")
        print(f"  - id={hit.get('id')}  distance={distance:.4f}  text={text!r}")


def list_collections(client: MilvusClient) -> None:
    print(f"[info] 当前所有集合：{client.list_collections()}")


def main() -> None:
    """端到端跑一遍：连接 -> 建表 -> 插入 -> 检索。"""
    print(f"连接 Milvus: {settings.milvus_uri_resolved}")
    client = build_client()

    create_demo_collection(client)
    insert_demo_data(client)
    search_demo(client)
    list_collections(client)

    client.close()
    print("完成。")


if __name__ == "__main__":
    main()
