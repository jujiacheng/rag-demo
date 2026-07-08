"""集中读取 .env 配置。

所有模块都通过 `settings` 这个单例获取配置，避免散落的环境变量读取。
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# backend/ 根目录，用于把相对路径（如 ./milvus_demo.db、./data/）锚定到这里
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """从 backend/.env 读取的应用配置。"""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 智谱 GLM（OpenAI 兼容接口）----
    zhipuai_api_key: str = "your_zhipuai_api_key_here"
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    llm_model: str = "glm-5.2"
    embedding_model: str = "embedding-3"

    # ---- Milvus ----
    # 默认 milvus-lite 单文件；切完整服务器版改成 http://localhost:19530
    milvus_uri: str = "./milvus_demo.db"
    collection_name: str = "rag_demo"

    # ---- 切分与检索 ----
    chunk_size: int = 500
    chunk_overlap: int = 50
    k: int = 4

    @property
    def milvus_uri_resolved(self) -> str:
        """若是本地文件路径（milvus-lite），锚定到 backend/ 目录。"""
        u = self.milvus_uri
        if "://" in u or u.startswith("/"):
            return u  # 已经是绝对路径或 http uri
        return str((BACKEND_DIR / u).resolve())

    @property
    def data_dir(self) -> Path:
        return BACKEND_DIR / "data"


settings = Settings()
