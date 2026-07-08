"""基于 langchain-milvus 的向量库封装。

这一层把 LangChain 的 `Milvus` 向量库包装成单例，供 ingest（写）和
agent（读）共用，避免重复建连接。
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_milvus import Milvus

from app.config import settings
from app.llm import get_embeddings
import app.milvus_compat  # noqa: F401  应用 pymilvus 新旧 API 桥接（见该模块）


@lru_cache(maxsize=1)
def get_vectorstore() -> Milvus:
    """返回单例向量库。"""
    return Milvus(
        embedding_function=get_embeddings(),
        connection_args={"uri": settings.milvus_uri_resolved},
        collection_name=settings.collection_name,
        auto_id=True,
        drop_old=False,  # 生产/演示都设 False，避免重启清库；ingest 里显式 drop
    )


def as_retriever(k: int | None = None):
    """返回 LangChain retriever，供 Agent 工具使用。"""
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k or settings.k},
    )


def similarity_search(query: str, k: int | None = None) -> list[Document]:
    """便捷方法：直接做一次相似度检索。"""
    return get_vectorstore().similarity_search(query, k=k or settings.k)
