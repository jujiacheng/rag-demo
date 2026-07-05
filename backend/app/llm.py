"""LLM 与 Embedding 工厂。

智谱 GLM 提供 OpenAI 兼容接口，因此可以直接用 langchain-openai 的
ChatOpenAI / OpenAIEmbeddings，只需把 base_url 指向智谱即可。
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings


def get_llm(streaming: bool = True) -> ChatOpenAI:
    """返回对接智谱 GLM 的聊天模型。"""
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.zhipuai_api_key,
        base_url=settings.llm_base_url,
        streaming=streaming,
        temperature=0.3,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """返回对接智谱的 embedding 模型。"""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.zhipuai_api_key,
        base_url=settings.llm_base_url,
    )


# 智谱 embedding-3 默认输出 2048 维；建集合时需要保持一致。
# 如果改用其它 embedding 模型，请同步修改这里的维度。
EMBEDDING_DIM = 2048
