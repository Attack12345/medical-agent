"""配置加载（pydantic-settings 读 .env）。

M1 只用到 MySQL 连接与路径常量；LLM/Neo4j/Qdrant 字段按文档预留，
后续里程碑（M2-M8）直接复用本模块。
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # medical-agent/
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # MySQL（本机服务，root/1234 仅本机开发）
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_db: str = "medical_agent"

    # LLM（阿里云百炼，OpenAI 兼容端点）
    dashscope_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen-plus"

    # 嵌入（百炼 text-embedding-v3，1024 维，批量上限 10）
    embedding_model: str = "text-embedding-v3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 10

    # Neo4j（M2 用，Docker compose infra/，宿主端口 7688/7475）
    neo4j_uri: str = "bolt://127.0.0.1:7688"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j1234"

    # Qdrant（M3 用，Docker compose infra/，宿主端口 6335/6336）
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6335

    # JWT（M6 用，密钥只放 .env）
    jwt_secret: str = "dev-secret-change-me"
    jwt_expire_hours: int = 24


settings = Settings()
