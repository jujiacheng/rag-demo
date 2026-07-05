# RAG + Milvus 2.6 技术分享 Demo

一个完整的 **Agentic RAG** 示例：用 LangChain 1.x 智能体 + Milvus 2.6 向量库搭建一个「Milvus 知识库问答助手」，前端 Next.js 聊天界面，后端 FastAPI 流式输出。

> 配套分享文档见 [`docs/RAG-with-Milvus-技术分享.md`](./docs/RAG-with-Milvus-技术分享.md)（约 40 分钟）。

---

## 架构一览

```
┌─────────────────┐   SSE    ┌──────────────────┐  stream_mode  ┌─────────────┐
│  Next.js 前端   │ ───────► │  FastAPI 后端    │ ────────────► │  Agent      │
│  (聊天 UI)      │ ◄─────── │  /api/chat       │ ◄──────────── │  (LC 1.x)   │
└─────────────────┘  token   └──────────────────┘   sources     └──────┬──────┘
        │                                                       │ 工具调用
        │ 同源代理                                               ▼
        ▼                                              ┌─────────────────┐
   app/api/chat/route.ts                               │  Milvus 2.6     │
                                                       │  (langchain-    │
                                                       │   milvus 封装)  │
                                                       └─────────────────┘
```

**核心思路**：把 Milvus 检索封装成 Agent 的一个工具（`@tool`），由 LLM 自己决定要不要检索。这就是 Agentic RAG。

---

## 技术栈

| 层 | 选型 |
|---|---|
| 向量库 | **Milvus 2.6**（本地用 milvus-lite 单文件，零运维） |
| SDK | pymilvus `2.6.12` + langchain-milvus `0.3.x` |
| Agent | **LangChain 1.x** `create_agent` + `@tool`（retriever 即工具） |
| LLM / Embedding | 智谱 GLM（OpenAI 兼容接口）：`glm-4.6` + `embedding-3` |
| 后端 | FastAPI + uvicorn，SSE 流式 |
| 前端 | Next.js 15 (App Router) + React 19 + TypeScript |
| 包管理 | Python 用 **uv**，前端用 **yarn** |

> **关于 pymilvus 版本**：刻意 pin 在 `2.6.12` 而非最新 `2.6.16`。
> 2.6.16 引入的 ConnectionManager 与 langchain-milvus 混用时存在已知 bug（`ConnectionNotExistException`，见 [milvus#48641](https://github.com/milvus-io/milvus/issues/48641)）。这是分享文档里的一个踩坑案例。

---

## 5 分钟快速开始

### 前置准备
- Python ≥ 3.11
- Node.js ≥ 18
- [uv](https://docs.astral.sh/uv/)（Python 包管理）：`curl -LsSf https://astral.sh/uv/install.sh | sh`
- [yarn](https://yarnpkg.com/)：`npm i -g yarn`
- **智谱 API Key**：在 https://open.bigmodel.com/ 注册获取

### 1. 配置后端

```bash
cd backend
cp .env.example .env
# 编辑 .env，填入你的智谱 API Key：
#   ZHIPUAI_API_KEY=xxxxx
```

### 2. 安装依赖并写入知识库

```bash
uv sync                              # 安装 Python 依赖（自动建 .venv）
uv run python -m app.ingest          # 把 data/milvus_faq.md 切分+embedding 写入 Milvus
# 也可以先跑原生 demo 看看 Milvus 基础操作：
uv run python -m app.milvus_raw
```

### 3. 启动后端

```bash
uv run uvicorn app.main:app --reload --port 8000
# 访问 http://localhost:8000/api/health 应返回 ok
```

### 4. 启动前端

```bash
cd ../frontend
cp .env.example .env                # 默认 BACKEND_URL=http://localhost:8000
yarn install
yarn dev                            # 访问 http://localhost:3000
```

打开 http://localhost:3000，提问「Milvus 2.6 有什么新特性？」即可看到流式回答 + 引用来源。

---

## 目录结构

```
.
├── docs/RAG-with-Milvus-技术分享.md   # 分享主文档（讲稿 + 代码走读）
├── docker-compose.yml                 # 可选：完整 Milvus standalone
├── backend/
│   ├── pyproject.toml                 # uv 管理，版本钉死
│   ├── .env.example
│   ├── app/
│   │   ├── config.py                  # 配置（pydantic-settings）
│   │   ├── llm.py                     # 智谱 LLM + Embedding 工厂
│   │   ├── milvus_raw.py              # 原生 MilvusClient 用法演示（第 4 节走读）
│   │   ├── vectorstore.py             # langchain-milvus 封装
│   │   ├── ingest.py                  # 文档切分 + 入库（离线）
│   │   ├── agent.py                   # Agentic RAG（第 5 节走读）
│   │   └── main.py                    # FastAPI + SSE
│   └── data/milvus_faq.md             # 知识库语料（20 条 Q&A）
└── frontend/
    ├── package.json
    ├── app/{layout,page}.tsx, api/chat/route.ts
    └── components/ChatBox.tsx         # 聊天 + 流式渲染 + 引用来源
```

---

## 切换到完整 Milvus 服务器

当需要 HNSW 完整索引或上规模时，用 Docker 起 standalone：

```bash
docker compose up -d                          # 起 etcd + minio + milvus
# 修改 backend/.env：MILVUS_URI=http://localhost:19530
# 修改 backend/pyproject.toml：把 pymilvus[milvus-lite] 改成裸 pymilvus
uv run python -m app.ingest                   # 重新入库
```

---

## 常见问题

**Q：启动报 `ConnectionNotExistException`？**
A：这是 pymilvus 2.6.16+ 的已知问题。本 demo 已 pin 在 `2.6.12` 规避，请确认 `uv.lock` 没被升级。

**Q：智谱接口报 401？**
A：检查 `backend/.env` 里的 `ZHIPUAI_API_KEY` 是否正确、`LLM_BASE_URL` 是否为 `https://open.bigmodel.com/api/paas/v4/`。

**Q：embedding 维度不匹配？**
A：智谱 `embedding-3` 默认 2048 维（见 `app/llm.py` 的 `EMBEDDING_DIM`）。换 embedding 模型时务必同步修改集合维度，否则入库/检索报错。

**Q：想换成 OpenAI / DeepSeek / 通义千问？**
A：改 `backend/.env` 里的 `LLM_BASE_URL` / `LLM_MODEL` / `EMBEDDING_MODEL` 即可，代码用 OpenAI 兼容接口，无需改动。
