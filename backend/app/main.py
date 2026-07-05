"""FastAPI 入口：提供聊天接口（SSE 流式）与健康检查。

启动：
    uv run uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent import get_agent
from app.config import settings

app = FastAPI(title="RAG + Milvus Demo API", version="0.1.0")

# 允许前端跨域（演示用，生产应收紧来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "milvus_uri": settings.milvus_uri,
        "collection": settings.collection_name,
    }


def _sse(event: str, data: dict) -> str:
    """组装一条 SSE 消息。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_agent(question: str) -> AsyncIterator[str]:
    """把 Agent 的流式输出转成 SSE。

    SSE 事件类型：
      - token：LLM 逐 token 输出
      - sources：检索到的引用来源（工具调用产生）
      - done：回答结束
    """
    agent = get_agent()
    sent_sources = False

    # stream_mode="messages" + version="v2" 把 LLM/工具的消息块逐个吐出，
    # chunk 结构统一为 {"type": "messages", "data": (message_chunk, metadata)}
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="messages",
        version="v2",
    ):
        if chunk.get("type") != "messages":
            continue

        message_chunk, metadata = chunk["data"]
        node = metadata.get("langgraph_node", "")

        # 1) 工具节点（content_and_artifact 工具会把原始文档挂在 .artifact 上）
        #    把检索到的文档作为「引用来源」推送一次给前端
        if node == "tools" and not sent_sources:
            artifact = getattr(message_chunk, "artifact", None)
            if artifact:
                sources = [
                    {
                        "content": getattr(doc, "page_content", str(doc)),
                        "source": getattr(doc, "metadata", {}).get("source", "")
                        if hasattr(doc, "metadata")
                        else "",
                    }
                    for doc in artifact
                ]
                yield _sse("sources", {"sources": sources})
                sent_sources = True

        # 2) Agent 节点的正文 -> 逐 token 推送
        if node == "agent" and message_chunk.content:
            yield _sse("token", {"text": message_chunk.content})

    yield _sse("done", {})


@app.post("/api/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        _stream_agent(req.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
