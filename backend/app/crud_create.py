"""Milvus CRUD 演示 · 增（Create）

演示：连接 → 建 schema → 建 HNSW 索引 → insert 插入 4 条。

用独立的 crud_demo.db，不影响主知识库 milvus_demo.db，后端在跑也不冲突。

运行：
    cd backend
    uv run python -m app.crud_create
"""
from __future__ import annotations

from pathlib import Path

from pymilvus import DataType, MilvusClient

# 锚定到 backend/，从任意目录跑都能找到同一个 db 文件
BACKEND_DIR = Path(__file__).resolve().parent.parent
URI = str(BACKEND_DIR / "crud_demo.db")
COLLECTION = "crud_demo"


def main() -> None:
    print(f"连接 Milvus（milvus-lite）：{URI}")
    client = MilvusClient(uri=URI)

    # 0) 若集合已存在，先删掉，保证可重复跑
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)
        print(f"[drop] 旧集合 {COLLECTION} 已删除")

    # 1) 定义 schema：自增主键 + 4 维向量 + 可空文本
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("text", DataType.VARCHAR, max_length=100, nullable=True)

    # 2) 准备索引：HNSW + COSINE
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 128},
    )

    # 3) 一次性建集合 + 建索引
    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    print(f"[create] 集合 {COLLECTION} 创建成功（HNSW / COSINE / dim=4）")

    # 4) insert 4 条（第 4 条 text 故意留空，演示 2.6 nullable）
    data = [
        {"vector": [0.1, 0.2, 0.3, 0.4], "text": "苹果"},
        {"vector": [0.8, 0.7, 0.6, 0.5], "text": "香蕉"},
        {"vector": [0.9, 0.1, 0.1, 0.9], "text": "猫"},
        {"vector": [0.5, 0.5, 0.5, 0.5], "text": None},
    ]
    result = client.insert(collection_name=COLLECTION, data=data)
    print(f"[insert] 写入 {len(data)} 条，自增主键 id = {result['ids']}")

    # 5) 确认行数
    client.load_collection(COLLECTION)
    stats = client.get_collection_stats(COLLECTION)
    print(f"[info] 当前行数：{stats['row_count']}")

    client.close()
    print("完成。")


if __name__ == "__main__":
    main()
