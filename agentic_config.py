#!/usr/bin/env python3
"""
Configuration for the Agentic RAG Pipeline
"""

import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class LightRAGConfig:
    """LightRAG API configuration"""
    api_url: str = "http://localhost:8080"
    timeout: int = 60
    max_retries: int = 3

@dataclass
class PhoenixConfig:
    """Phoenix tracing configuration"""
    project_name: str = "agentic-rag-pipeline"
    host: str = "localhost"
    port: int = 6006
    endpoint: str = "http://localhost:6006/v1/traces"

@dataclass
class LLMConfig:
    """Language model configuration"""
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4000
    api_key: str = None

@dataclass
class RetrievalConfig:
    """Hybrid retrieval configuration"""
    default_mode: str = "hybrid"  # naive, local, global, hybrid
    max_chunks: int = 10
    confidence_threshold: float = 0.7
    
    # Mode selection rules
    complex_query_mode: str = "hybrid"
    multi_doc_mode: str = "global"
    simple_query_mode: str = "local"

@dataclass
class MultilingualConfig:
    """Multilingual support configuration"""
    default_language: str = "en"
    supported_languages: list = None
    auto_translate: bool = True
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"]

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    enable_evaluation: bool = True
    relevance_weight: float = 0.3
    qa_weight: float = 0.3
    hallucination_weight: float = 0.3
    source_attribution_weight: float = 0.1
    
    # Evaluation thresholds
    min_relevance_score: float = 0.6
    max_hallucination_score: float = 0.3
    min_overall_score: float = 0.7

class AgenticRAGConfig:
    """Main configuration class for the Agentic RAG Pipeline"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        # Load from environment or defaults
        self.lightrag = LightRAGConfig(
            api_url=os.getenv("LIGHTRAG_API_URL", "https://lightrag-h7h2g9atabdjg3af.canadacentral-01.azurewebsites.net")
        )
        
        self.phoenix = PhoenixConfig(
            host=os.getenv("PHOENIX_HOST", "localhost"),
            port=int(os.getenv("PHOENIX_PORT", "6006"))
        )
        
        self.llm = LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.retrieval = RetrievalConfig()
        self.multilingual = MultilingualConfig()
        self.evaluation = EvaluationConfig()
        
        # Override with provided config
        if config_dict:
            self._update_from_dict(config_dict)
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self, section) and isinstance(values, dict):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        if not self.llm.api_key:
            errors.append("OpenAI API key is required")
        
        if not self.lightrag.api_url:
            errors.append("LightRAG API URL is required")
        
        if self.retrieval.confidence_threshold < 0 or self.retrieval.confidence_threshold > 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "lightrag": {
                "api_url": self.lightrag.api_url,
                "timeout": self.lightrag.timeout,
                "max_retries": self.lightrag.max_retries
            },
            "phoenix": {
                "project_name": self.phoenix.project_name,
                "host": self.phoenix.host,
                "port": self.phoenix.port,
                "endpoint": self.phoenix.endpoint
            },
            "llm": {
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens
            },
            "retrieval": {
                "default_mode": self.retrieval.default_mode,
                "max_chunks": self.retrieval.max_chunks,
                "confidence_threshold": self.retrieval.confidence_threshold
            },
            "multilingual": {
                "default_language": self.multilingual.default_language,
                "supported_languages": self.multilingual.supported_languages,
                "auto_translate": self.multilingual.auto_translate
            },
            "evaluation": {
                "enable_evaluation": self.evaluation.enable_evaluation,
                "relevance_weight": self.evaluation.relevance_weight,
                "qa_weight": self.evaluation.qa_weight,
                "hallucination_weight": self.evaluation.hallucination_weight,
                "source_attribution_weight": self.evaluation.source_attribution_weight
            }
        }

# Predefined configurations for different use cases
DEVELOPMENT_CONFIG = {
    "evaluation": {
        "enable_evaluation": True,
        "min_overall_score": 0.6  # Lower threshold for development
    },
    "retrieval": {
        "max_chunks": 15,
        "confidence_threshold": 0.6
    }
}

PRODUCTION_CONFIG = {
    "evaluation": {
        "enable_evaluation": True,
        "min_overall_score": 0.8  # Higher threshold for production
    },
    "retrieval": {
        "max_chunks": 10,
        "confidence_threshold": 0.7
    },
    "llm": {
        "temperature": 0.0  # More deterministic for production
    }
}

MULTILINGUAL_CONFIG = {
    "multilingual": {
        "auto_translate": True,
        "supported_languages": ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ar", "ru"]
    },
    "retrieval": {
        "max_chunks": 12  # More chunks for multilingual queries
    }
}

def get_config(config_type: str = "development") -> AgenticRAGConfig:
    """Get predefined configuration"""
    config_map = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
        "multilingual": MULTILINGUAL_CONFIG
    }
    
    base_config = config_map.get(config_type, {})
    config = AgenticRAGConfig(base_config)
    config.validate()
    
    return config

if __name__ == "__main__":
    # Test configuration
    config = get_config("development")
    print("📋 Agentic RAG Configuration:")
    print("-" * 40)
    
    import json
    print(json.dumps(config.to_dict(), indent=2))
    
    print(f"\n✅ Configuration validated successfully")
    print(f"🔗 LightRAG API: {config.lightrag.api_url}")
    print(f"🔍 Phoenix UI: http://{config.phoenix.host}:{config.phoenix.port}")
    print(f"🌍 Multilingual: {len(config.multilingual.supported_languages)} languages supported")
