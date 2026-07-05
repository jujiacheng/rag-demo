"use client";

import { useRef, useState } from "react";

type Source = { content: string; source: string };
type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

const SUGGESTIONS = [
  "Milvus 2.6 有什么新特性？",
  "本地开发怎么最快跑起 Milvus？",
  "朴素 RAG 和 Agentic RAG 有什么区别？",
  "HNSW 索引的关键参数有哪些？",
];

export default function ChatBox() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  async function send(question: string) {
    const q = question.trim();
    if (!q || loading) return;

    setInput("");
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: q }]);

    // 占位一条空的 assistant 消息，随后逐 token 填充
    const assistantIdx = messages.length + 1;
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", sources: [] },
    ]);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
        signal: controller.signal,
      });
      if (!resp.ok || !resp.body) throw new Error(`请求失败：${resp.status}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      // 解析 SSE 流：按 \n\n 分帧，取 event: 和 data:
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";

        for (const frame of frames) {
          let event = "";
          let data = "";
          for (const line of frame.split("\n")) {
            if (line.startsWith("event: ")) event = line.slice(7).trim();
            else if (line.startsWith("data: ")) data += line.slice(6);
          }
          if (!data) continue;

          try {
            const parsed = JSON.parse(data);
            if (event === "token") {
              setMessages((prev) => {
                const next = [...prev];
                next[assistantIdx] = {
                  ...next[assistantIdx],
                  content: next[assistantIdx].content + parsed.text,
                };
                return next;
              });
            } else if (event === "sources") {
              setMessages((prev) => {
                const next = [...prev];
                next[assistantIdx] = {
                  ...next[assistantIdx],
                  sources: parsed.sources as Source[],
                };
                return next;
              });
            }
          } catch {
            /* 忽略无法解析的帧 */
          }
        }
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setMessages((prev) => {
        const next = [...prev];
        if (next[assistantIdx]) {
          next[assistantIdx] = {
            ...next[assistantIdx],
            content: `⚠️ 出错了：${msg}`,
          };
        }
        return next;
      });
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  return (
    <div className="page">
      <div className="header">
        <h1>RAG + Milvus 2.6 演示</h1>
        <p>LangChain 1.x 智能体 × Milvus 向量库 · Agentic RAG</p>
      </div>

      <div className="suggestions">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            className="suggestion"
            onClick={() => send(s)}
            disabled={loading}
          >
            {s}
          </button>
        ))}
      </div>

      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            {m.content || (loading && m.role === "assistant" ? "思考中…" : "")}
            {m.sources && m.sources.length > 0 && (
              <div className="sources">
                <h4>📎 引用来源（来自 Milvus 检索）</h4>
                {m.sources.map((s, j) => (
                  <div key={j} className="source-item">
                    {s.content}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="input-bar">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") send(input);
          }}
          placeholder="问点关于 Milvus / RAG 的问题…"
          disabled={loading}
        />
        <button onClick={() => send(input)} disabled={loading || !input.trim()}>
          {loading ? "回答中" : "发送"}
        </button>
      </div>
    </div>
  );
}
