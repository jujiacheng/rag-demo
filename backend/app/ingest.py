"""知识库入库脚本（离线阶段）。

流程：读取 data/*.md -> 切分 -> embedding -> 写入 Milvus。

运行：
    uv run python -m app.ingest
"""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.llm import get_embeddings


def load_documents() -> list[str]:
    """读取 data 目录下所有 .md 文件的纯文本。"""
    docs_dir = settings.data_dir
    if not docs_dir.exists():
        print(f"[error] 数据目录不存在：{docs_dir}", file=sys.stderr)
        sys.exit(1)

    chunks_text: list[str] = []
    for md_file in sorted(docs_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        chunks_text.append(f"# 来源文件：{md_file.name}\n\n{text}")
        print(f"[load] {md_file.name}  ({len(text)} 字符)")

    if not chunks_text:
        print(f"[error] 数据目录里没有 .md 文件：{docs_dir}", file=sys.stderr)
        sys.exit(1)

    return chunks_text


def build_vectorstore(drop_old: bool = True) -> Milvus:
    """建一个新的向量库连接；drop_old=True 时会清掉旧集合重建。"""
    return Milvus(
        embedding_function=get_embeddings(),
        connection_args={"uri": settings.milvus_uri_resolved},
        collection_name=settings.collection_name,
        auto_id=True,
        drop_old=drop_old,
    )


def main() -> None:
    print(f"连接 Milvus: {settings.milvus_uri_resolved}")
    print(f"集合名:      {settings.collection_name}")
    print(f"切分参数:    chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}")

    # 1) 读取语料
    raw_texts = load_documents()

    # 2) 切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " "],
    )
    chunks = splitter.create_documents(raw_texts)
    print(f"[split] 切分为 {len(chunks)} 个 chunk")

    # 3) 入库（drop_old=True 重建集合，保证幂等）
    vs = build_vectorstore(drop_old=True)
    vs.add_documents(documents=chunks)
    print(f"[done] 已写入 {len(chunks)} 个 chunk 到集合 {settings.collection_name}")


if __name__ == "__main__":
    main()
