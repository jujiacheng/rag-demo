"""Milvus CRUD 演示 · 改（Update）

演示：upsert（按 id 整条覆盖）。

注意：Milvus 没有「update 单字段」的 API，只能 upsert 整条——
      给定 id + 完整字段，id 已存在则覆盖，不存在则插入。

前置：先跑 crud_create 建集合并写入数据。

运行：
    cd backend
    uv run python -m app.crud_update
"""
from __future__ import annotations

from pathlib import Path

from pymilvus import MilvusClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
URI = str(BACKEND_DIR / "crud_demo.db")
COLLECTION = "crud_demo"


def main() -> None:
    print(f"连接 Milvus（milvus-lite）：{URI}")
    client = MilvusClient(uri=URI)
    client.load_collection(COLLECTION)

    print()
    print("=== 改之前：get id=1 ===")
    before = client.get(collection_name=COLLECTION, ids=[1], output_fields=["text", "vector"])
    print(f"  {before}")

    print()
    print("=== upsert id=1：text「苹果」→「红富士苹果」，vector 也微调 ===")
    client.upsert(
        collection_name=COLLECTION,
        data=[{"id": 1, "vector": [0.15, 0.25, 0.35, 0.45], "text": "红富士苹果"}],
    )
    print("  upsert 完成")

    print()
    print("=== 改之后：再 get id=1 ===")
    after = client.get(collection_name=COLLECTION, ids=[1], output_fields=["text", "vector"])
    print(f"  {after}")

    client.close()
    print()
    print("完成。")
    print()
    print("要点：")
    print("  - upsert = insert + update 合一：id 存在则覆盖整条，不存在则插入")
    print("  - 不能只改一个字段，必须给完整字段（vector + text 都要带上）")


if __name__ == "__main__":
    main()
