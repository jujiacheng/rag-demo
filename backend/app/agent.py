"""Agentic RAG：把向量检索做成智能体的工具。

对应分享文档第 5 节。与「朴素 RAG」的区别在于：
  - 朴素 RAG：固定管线，每次都先检索再生成；
  - Agentic RAG：由 Agent 自己决定要不要检索、检索什么。

LangChain 1.x 推荐写法：
  - from langchain.agents import create_agent
  - from langchain.tools import tool

`@tool(response_format="content_and_artifact")` 让工具既能给模型返回“序列化文本”，
又能让应用层拿到“原始文档”，从而在前端展示引用来源。
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain.tools import tool

from app.config import settings
from app.llm import get_llm
from app.vectorstore import get_vectorstore

SYSTEM_PROMPT = """你是一名熟悉 Milvus 向量数据库的技术支持助手。

回答规则：
1. 对于涉及 Milvus / RAG / 向量检索的问题，必须先调用知识库检索工具获取资料，再基于资料回答。
2. 只使用检索到的资料作答；资料中没有的内容，要明确说明“知识库中未提及”，不要编造。
3. 回答用简洁清晰的中文，可在关键处引用资料原文。
4. 对于闲聊或与知识库无关的问题，可以直接作答，不必检索。
"""


@tool(response_format="content_and_artifact")
def search_milvus_kb(query: str):
    """在 Milvus 知识库（2.6 中文 FAQ / 产品手册）中检索与问题最相关的片段。

    Args:
        query: 用户的检索问题或关键词。
    """
    docs = get_vectorstore().similarity_search(query, k=settings.k)
    serialized = "\n\n".join(
        f"[来源 #{i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )
    # 返回 (给模型看的文本, 给应用层用的原始文档)
    return serialized, docs


def build_agent():
    """构建并返回一个 Agentic RAG 智能体。"""
    llm = get_llm()
    return create_agent(
        model=llm,
        tools=[search_milvus_kb],
        system_prompt=SYSTEM_PROMPT,
    )


# 全局单例，FastAPI 启动时即构建，避免每次请求重建
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent
