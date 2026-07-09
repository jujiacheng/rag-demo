"""Milvus CRUD 演示 · 删（Delete）

演示：
  - delete          ：按 id 删行（集合还在，只删指定行）
  - drop_collection ：删整个集合（集合 + 数据 + 索引全没）

前置：先跑 crud_create 建集合并写入数据。

运行：
    cd backend
    uv run python -m app.crud_delete
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
    print("=== 删之前：所有数据 ===")
    rows = client.query(
        collection_name=COLLECTION,
        filter="",
        output_fields=["id", "text"],
        limit=10,
    )
    for r in rows:
        print(f"  {r}")
    print(f"  共 {len(rows)} 条")

    print()
    print("=== delete id=3 和 id=4（按主键删行） ===")
    client.delete(collection_name=COLLECTION, ids=[3, 4])
    print("  删除完成")

    print()
    print("=== 删之后：剩余数据 ===")
    rows = client.query(
        collection_name=COLLECTION,
        filter="",
        output_fields=["id", "text"],
        limit=10,
    )
    for r in rows:
        print(f"  {r}")
    print(f"  共 {len(rows)} 条")

    print()
    print("=== drop_collection：删整个集合 ===")
    client.drop_collection(COLLECTION)
    print(f"  集合 {COLLECTION} 已删除")
    print(f"  剩余集合：{client.list_collections()}")

    # 清理 db 文件（可选，演示完留个干净环境）
    client.close()
    print()
    print("完成。")
    print()
    print("区分：")
    print("  delete           = 删行，集合 schema/索引保留")
    print("  drop_collection  = 删集合，schema + 索引 + 数据全没")


if __name__ == "__main__":
    main()
