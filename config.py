"""
Unified configuration management for the Aparavi data processing pipeline.
Handles connections to Neo4j, Qdrant, OpenAI, and LlamaParse.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Also try to load from graph_stream/.env if it exists
load_dotenv("graph_stream/.env")


@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""
    uri: str
    username: str
    password: str
    database: str = "neo4j"


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration."""
    url: str
    api_key: str
    collection_name: str = "documents"


@dataclass
class LlamaParseConfig:
    """LlamaParse configuration."""
    api_key: str


@dataclass
class AppConfig:
    """Main application configuration."""
    neo4j: Neo4jConfig
    qdrant: QdrantConfig
    llama_parse: LlamaParseConfig
    openai_api_key: str


def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    
    # Neo4j Configuration
    neo4j = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "neo4j+s://ce843f2b.databases.neo4j.io"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "").strip(),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    
    # Qdrant Configuration
    qdrant = QdrantConfig(
        url=os.getenv("QDRANT_URL", "https://your-cluster-url.qdrant.io"),
        api_key=os.getenv("QDRANT_API_KEY", "").strip(),
        collection_name=os.getenv("QDRANT_COLLECTION", "documents"),
    )
    
    # LlamaParse Configuration
    llama_parse = LlamaParseConfig(
        api_key=os.getenv("LLAMA_PARSE_API_KEY", "").strip(),
    )
    
    return AppConfig(
        neo4j=neo4j,
        qdrant=qdrant,
        llama_parse=llama_parse,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
    )


def validate_config(config: AppConfig) -> None:
    """Validate that required configuration is present."""
    errors = []
    
    if not config.openai_api_key:
        errors.append("OPENAI_API_KEY is required")
    
    if not config.llama_parse.api_key:
        errors.append("LLAMA_PARSE_API_KEY is required")
    
    # Optional validations (only warn, don't error)
    warnings = []
    
    if not config.neo4j.password:
        warnings.append("NEO4J_PASSWORD is not set (optional)")
    
    if not config.qdrant.api_key:
        warnings.append("QDRANT_API_KEY is not set (optional)")
    
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"- {error}" for error in errors))
