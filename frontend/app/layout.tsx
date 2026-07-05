import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG + Milvus Demo",
  description: "LangChain 1.x × Milvus 2.6 的 Agentic RAG 演示",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
