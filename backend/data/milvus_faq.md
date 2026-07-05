# Milvus 2.6 中文 FAQ / 产品手册

> 本文件是 RAG demo 的知识库语料。内容围绕 Milvus 2.6 的核心概念与新特性组织，
> 每条 Q&A 都是一个独立的可检索单元。你可以自由扩充。

---

## Q1: Milvus 是什么？

Milvus 是一款开源的、专为向量相似度检索而生的分布式数据库。它能够对海量高维向量进行高效的存储、索引与近邻检索（ANN），广泛应用于图像/语音识别、推荐系统、语义搜索、RAG 等场景。Milvus 由 Zilliz 团队创建并捐赠给 LF AI & Data 基金会，是目前最流行的开源向量数据库之一。

## Q2: Milvus 2.6 有哪些重要新特性？

Milvus 2.6 于 2025 年 8 月发布 GA，主打“为规模而生、为降本而设计”。主要新特性包括：
1. **内置 BM25 全文检索**：通过原生的 Function（FunctionType.BM25）把 VARCHAR 文本字段自动转成稀疏向量，无需再外接稀疏嵌入模型，吞吐比 Elasticsearch 高约 4 倍。
2. **可空字段与默认值**（nullable & default_value）：字段可以为空或带默认值，schema 更灵活。
3. **动态加字段**（add_field / alter_index）：不必重建集合就能给已有集合增加字段。
4. **JSON Path Index 与 JSON Flat Index**：对 JSON 字段的元数据过滤速度提升约 100 倍。
5. **空值感知过滤、更高效的 COUNT(\*)**，以及进一步的内存与成本优化。

## Q3: pymilvus 推荐用哪种连接方式？

在 2.6 中，官方推荐使用新的 **MilvusClient** API：`from pymilvus import MilvusClient`，然后用 `MilvusClient(uri=...)` 建立连接。它是无全局状态的、扁平的接口，自带异步支持（AsyncMilvusClient），是当前文档与示例的主推方式。旧的 ORM 风格（`connections.connect` + `Collection`）仍然兼容，但不建议用于新代码。

## Q4: 本地开发怎么最快跑起 Milvus？

最快的办法是用 **milvus-lite**：它是 pymilvus 的一个扩展，安装后直接把 MilvusClient 的 uri 指向一个本地文件路径（例如 `MilvusClient("./milvus_demo.db")`），就能以进程内嵌的方式运行，无需任何容器或服务。注意 milvus-lite 只支持 Flat 索引，且仅限 macOS / Linux、Python 3.10+。如果需要完整功能（HNSW/IVF、集群等），请改用 Docker standalone 版。

## Q5: 如何在 Milvus 中创建一个集合并插入向量？

使用 MilvusClient 的标准流程：先 `MilvusClient.create_schema()` 定义字段，再 `client.prepare_index_params()` 准备索引参数，然后 `client.create_collection(collection_name, schema, index_params)` 一次建表+建索引；接着 `client.insert(collection_name, data)` 插入数据；最后 `client.load_collection(name)` 加载到内存即可检索。

## Q6: HNSW 索引怎么配置？关键参数有哪些？

HNSW（Hierarchical Navigable Small World）是 Milvus 中低延迟近邻检索的推荐索引。建索引时通过 `index_params.add_index(field_name, index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 256})` 配置。M 控制图的连接度（默认 16），efConstruction 控制建图质量（默认 256）。查询时在 search_params 里传 `{"params": {"ef": 64}}` 控制召回精度，ef 越大越准但越慢。

## Q7: 向量相似度检索怎么调用？

通过 `client.search(collection_name, data=[query_vector], anns_field="vector", limit=5, output_fields=["text"], search_params={"params": {"ef": 64}})` 执行检索。data 是查询向量列表，limit 是返回条数，output_fields 指定要回带的标量字段，metric_type 已在建索引时确定。结果会按相似度从高到低排序返回。

## Q8: Milvus 2.6 的混合检索（dense + sparse）怎么用？

混合检索同时利用稠密向量（语义）和稀疏向量（关键词/BM25）。在 schema 中同时定义一个 `FLOAT_VECTOR` 字段和一个 `SPARSE_FLOAT_VECTOR` 字段，分别建索引（稀疏用 `SPARSE_INVERTED_INDEX` + `IP`）。查询时构造多个 `AnnSearchRequest`（每个向量字段一个），再用 `client.hybrid_search(..., ranker=RRFRanker(k=60))` 或 `WeightedRanker(0.7, 0.3)` 做重排融合。

## Q9: 什么是 RAG？它解决什么问题？

RAG（Retrieval-Augmented Generation，检索增强生成）通过“先检索、再生成”的方式增强大语言模型。它解决 LLM 的三大硬伤：**知识时效性**（训练数据有截止日期）、**幻觉**（编造事实）、**私域知识**（无法访问企业内部文档）。RAG 把外部知识作为上下文喂给 LLM，让其基于真实资料回答，既准确又可追溯来源。

## Q10: 一个典型的 RAG 流程包含哪些步骤？

典型 RAG 分两阶段：
1. **建库阶段（离线）**：加载文档 → 切分（chunking）→ embedding → 存入向量数据库并建索引。
2. **检索生成阶段（在线）**：用户提问 → query embedding → 向量数据库相似度检索 top-k → 把检索片段拼进 prompt → LLM 生成回答。

## Q11: 朴素 RAG 和 Agentic RAG 有什么区别？

朴素 RAG 是“固定管线”：每次都先检索再生成，不管问题是否需要检索。Agentic RAG 把检索变成**智能体的一个工具**：由 LLM（Agent）自己判断当前问题要不要检索、检索几次、检索什么关键词。这样对简单问候（“你好”）不必检索，对复杂问题可以多次检索或改写 query，灵活性和准确率都更高。

## Q12: 在 LangChain 1.x 里怎么把向量检索做成 Agent 的工具？

LangChain 1.x 推荐用 `from langchain.agents import create_agent`，并用 `from langchain.tools import tool` 装饰器定义工具。典型做法：定义一个 `@tool(response_format="content_and_artifact")` 的函数，函数内调用 `vector_store.similarity_search(query, k=4)`，返回（序列化文本, 原始文档列表）。这样 Agent 调用工具时只看到文本，而应用层还能拿到原始文档用于展示引用来源。

## Q13: LangChain 1.x 的 create_agent 和旧的 create_react_agent 有什么不同？

`create_agent`（来自 `langchain.agents`）是 LangChain 1.x 推荐的统一 Agent 构建入口，封装了 ReAct 循环、工具调用、系统提示等，开箱即用。旧的 `create_react_agent`（来自 `langgraph.prebuilt`）仍然可用，但属于更底层的 LangGraph API。新项目建议直接用 `create_agent`，更简洁、更贴近 1.x 的标准用法。

## Q14: Milvus 支持哪些度量方式（metric type）？

Milvus 支持三种主流度量：**L2（欧氏距离）**、**IP（内积）**、**COSINE（余弦相似度）**。文本语义检索用 embedding 时，通常用 COSINE 或 IP；图像视觉向量常用 L2。metric_type 在建索引时指定，整个集合字段共享同一度量。

## Q15: 向量维度（dim）怎么选？

向量维度由你使用的 embedding 模型决定，必须严格一致。例如智谱 embedding-3 默认输出 2048 维，OpenAI text-embedding-3-small 是 1536 维。建集合时在 `add_field(datatype=DataType.FLOAT_VECTOR, dim=2048)` 里写死维度，之后插入的所有向量都必须是这个维度，否则会报错。

## Q16: Milvus 2.6 的 nullable 字段有什么用？

nullable 字段允许某些记录的某字段为空，配合 default_value 还能给定默认值。这在数据不完整、字段可选、增量补全数据等场景非常有用——以前要么必须填一个占位值，要么重建集合。2.6 之后可以直接在 schema 里 `add_field(..., nullable=True, default_value=...)`。

## Q17: 切分文档（chunking）对 RAG 效果有多大影响？

非常大。chunk 太长，一个片段里混入多个主题，检索精度下降；chunk 太短，上下文不完整，回答易断章取义。一般经验：中文用 300–500 字、英文用 500–1000 字符，并保留一定 overlap（如 50–100）防止语义在切分处断裂。LangChain 的 `RecursiveCharacterTextSplitter` 按“段落 → 句子 → 词”的层级切分，是比较稳妥的默认选择。

## Q18: 什么情况下该从 milvus-lite 切到完整服务器版？

当出现以下任一情况就该切：数据量上百万甚至亿级、需要 HNSW/IVF 等高级索引、需要多副本/高可用、需要集群水平扩展、或要在 Windows 上运行。切换很简单：用 docker-compose 起一个 standalone，然后把 `MILVUS_URI` 从本地文件路径改成 `http://localhost:19530`，代码几乎不用改。

## Q19: 检索的 K 值（top-k）设多少合适？

没有标准答案，通常从 k=3~5 起步。k 太小容易漏掉相关信息，k 太大会引入噪声、拖慢生成、还可能超出 LLM 上下文窗口。配合 rerank（先用向量检索召回较多候选，再用模型精排）往往能拿到更好效果。RAG demo 里一般设 k=4 作为平衡点。

## Q20: 如何评估一个 RAG 系统的好坏？

主要看两个层面：**检索质量**（召回率、准确率、MRR、NDCG，看相关文档有没有进 top-k）和**生成质量**（答案准确度、是否忠实于检索内容、是否会产生幻觉、完整性）。常用做法是构造一批“问题-标准答案-相关文档”的评测集，分别评估检索和生成。LangSmith、Ragas 等工具可以辅助评估与追踪。
