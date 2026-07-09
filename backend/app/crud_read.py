"""Milvus CRUD 演示 · 查（Read）

演示三种查询方式（容易混，重点区分）：
  - get    ：按主键 id 直接取（不涉及向量、不涉及相似度）
  - query  ：标量过滤（给过滤表达式，如 id >= 2，不涉及向量）
  - search ：向量相似度检索（给一个 query 向量，找最相似的 top-k）

前置：先跑 crud_create 建集合并写入数据。

运行：
    cd backend
    uv run python -m app.crud_read
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
    print("=== 1) get：按主键 id 直接取（不走向量、不走索引） ===")
    rows = client.get(collection_name=COLLECTION, ids=[1, 2], output_fields=["text"])
    for r in rows:
        print(f"  {r}")

    print()
    print("=== 2) query：标量过滤（id >= 2，不涉及向量相似度） ===")
    rows = client.query(
        collection_name=COLLECTION,
        filter="id >= 2",
        output_fields=["id", "text"],
        limit=10,
    )
    for r in rows:
        print(f"  {r}")

    print()
    print("=== 3) search：向量相似度检索（给 query 向量找 top-3） ===")
    query_vector = [0.1, 0.2, 0.3, 0.4]  # 和「苹果」一样，应排第一
    results = client.search(
        collection_name=COLLECTION,
        data=[query_vector],
        anns_field="vector",
        limit=3,
        output_fields=["text"],
        search_params={"params": {"ef": 64}},
    )
    for hit in results[0]:
        distance = hit.get("distance")
        text = hit.get("entity", {}).get("text")
        print(f"  id={hit.get('id')}  distance={distance:.4f}  text={text!r}")

    client.close()
    print()
    print("完成。")
    print()
    print("关键区分：")
    print("  get    = 按主键精确取（O(1)，不走向量）")
    print("  query  = 标量过滤（WHERE id >= 2，不走向量）")
    print("  search = 向量相似度（给向量找最近的，走 ANN 索引）")


if __name__ == "__main__":
    main()
