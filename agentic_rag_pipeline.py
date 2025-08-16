#!/usr/bin/env python3
"""
Agentic RAG Pipeline with Hybrid Retrieval, Multilingual Support, and Evaluation
Integrates with LightRAG API and uses Phoenix for tracing and evaluation
"""

import os
import json
import requests
import pandas as pd
import re
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
import logging
from datetime import datetime

# Simple workflow management without LangGraph complexity
from typing_extensions import Annotated

# LangChain components
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Phoenix for tracing and evaluation
import phoenix as px
from phoenix.otel import register
from phoenix.trace import using_project
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals
)
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language detection
try:
    from langdetect import detect, LangDetectError
except ImportError:
    logger.warning("langdetect not installed. Install with: pip install langdetect")
    detect = None

@dataclass
class RetrievalContext:
    """Context from LightRAG retrieval"""
    content: str
    sources: List[str]
    query_mode: str
    language: str
    confidence: float

@dataclass
class EvaluationResult:
    """Evaluation metrics"""
    relevance_score: float
    answer_quality_score: float
    hallucination_score: float
    source_attribution_score: float
    overall_score: float

class QueryClassification(BaseModel):
    """Classification of the user query"""
    intent: Literal["financial_analysis", "comparison", "time_series", "factual", "calculation", "other"] = Field(
        description="Primary intent of the query"
    )
    complexity: Literal["simple", "medium", "complex"] = Field(
        description="Complexity level of the query"
    )
    requires_multi_doc: bool = Field(description="Whether query needs multiple documents")
    time_sensitive: bool = Field(description="Whether query involves time-based analysis")
    language: str = Field(description="Detected language of query")
    entities: List[str] = Field(description="Key entities mentioned in query")

class AgenticAnswer(BaseModel):
    """Final structured answer"""
    answer: str = Field(description="Comprehensive answer to the query")
    sources: List[str] = Field(description="Source documents and page citations")
    confidence: float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="Step-by-step reasoning process")
    query_classification: QueryClassification = Field(description="Query analysis")
    retrieval_mode: str = Field(description="LightRAG mode used for retrieval")
    language: str = Field(description="Response language")

class AgenticRAGState(BaseModel):
    """State for the agentic RAG workflow"""
    original_query: str
    translated_query: Optional[str] = None
    query_classification: Optional[QueryClassification] = None
    retrieval_context: Optional[RetrievalContext] = None
    answer: Optional[AgenticAnswer] = None
    evaluation: Optional[EvaluationResult] = None
    messages: List[BaseMessage] = []
    trace_id: Optional[str] = None

class LightRAGClient:
    """Enhanced client for LightRAG API interactions with full parameter support"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip('/')
        
    def query(self, query: str, mode: str = "mix", **kwargs) -> Dict[str, Any]:
        """Query LightRAG with full parameter support as per API specification"""
        
        # Build request parameters matching your curl example
        query_params = {
            "query": query,
            "mode": mode,
            "only_need_context": kwargs.get("only_need_context", False),
            "only_need_prompt": kwargs.get("only_need_prompt", False),
            "response_type": kwargs.get("response_type", "string"),
            "top_k": kwargs.get("top_k", 10),
            "chunk_top_k": kwargs.get("chunk_top_k", 5),
            "max_entity_tokens": kwargs.get("max_entity_tokens", 4000),
            "max_relation_tokens": kwargs.get("max_relation_tokens", 2000),
            "max_total_tokens": kwargs.get("max_total_tokens", 8000),
            "conversation_history": kwargs.get("conversation_history", []),
            "history_turns": kwargs.get("history_turns", 0),
            "ids": kwargs.get("ids", []),
            "user_prompt": kwargs.get("user_prompt", ""),
            "enable_rerank": kwargs.get("enable_rerank", True),
            # Cache busting - force fresh retrieval every time
            "timestamp": datetime.now().isoformat(),
            "cache": False,  # Disable caching if supported
            "force_refresh": True  # Force fresh analysis
        }
        
        try:
            # Include API key if available
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            # Add API key authentication if provided
            api_key = kwargs.get('api_key') or os.getenv('LIGHTRAG_API_KEY')
            if api_key:
                headers['X-API-Key'] = api_key
            
            response = requests.post(
                f"{self.api_url}/query",
                json=query_params,
                headers=headers,
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"LightRAG API error: {response.status_code} - {response.text}")
                return {"error": f"API error: {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to LightRAG API: {e}")
            return {"error": f"Connection error: {e}"}
    
    def get_optimal_params(self, selected_mode: str, query_classification) -> Dict[str, Any]:
        """Get optimal LightRAG parameters based on query classification"""
        params = {}
        
        # Enhanced parameters for hybrid mode - better context quality
        if query_classification.complexity == "complex":
            params.update({
                "top_k": 20,  # More entities for complex queries
                "chunk_top_k": 12,  # More context chunks
                "max_entity_tokens": 6000,  # More detailed entities
                "max_relation_tokens": 3000,  # More relationships
                "max_total_tokens": 15000,
                "enable_rerank": True
            })
        elif query_classification.complexity == "medium":
            params.update({
                "top_k": 15,
                "chunk_top_k": 8,
                "max_entity_tokens": 5000,
                "max_relation_tokens": 2500,
                "max_total_tokens": 10000,
                "enable_rerank": True
            })
        else:  # simple
            params.update({
                "top_k": 15,  # Much more entities even for simple queries
                "chunk_top_k": 10,  # Many more document chunks
                "max_entity_tokens": 6000,  # More detailed entity content
                "max_relation_tokens": 3000,  # More relationship content  
                "max_total_tokens": 12000,  # Much more total content
                "enable_rerank": True  # Always rerank in hybrid mode
            })
        
        # Mode will be passed in from intelligent selection
        params["mode"] = selected_mode
        
        # Enhanced parameters based on selected mode
        if selected_mode == "global":
            # Global mode: boost document chunks for comprehensive coverage
            params.update({
                "chunk_top_k": params.get("chunk_top_k", 10) + 4,  # More document chunks
                "max_total_tokens": params.get("max_total_tokens", 12000) + 4000,  # More comprehensive content
                "enable_rerank": True
            })
        elif selected_mode == "hybrid":
            # Hybrid mode: boost both entities AND document chunks for comprehensive quarterly coverage
            params.update({
                "top_k": params["top_k"] + 5,  # More entities for relationship analysis
                "chunk_top_k": params.get("chunk_top_k", 8) + 6,  # MORE document chunks for quarterly coverage
                "max_entity_tokens": params.get("max_entity_tokens", 6000) + 2000,  # More entity detail
                "max_relation_tokens": params.get("max_relation_tokens", 3000) + 1500,  # More relationships
                "max_total_tokens": params.get("max_total_tokens", 10000) + 6000,  # Much more total content
                "enable_rerank": True
            })
        else:  # mix mode (dual retrieval - maximum parameters)
            # Mix mode: MAXIMUM parameters for dual retrieval (global + hybrid combined)
            params.update({
                "top_k": params["top_k"] + 8,  # MAXIMUM entities for dual retrieval
                "chunk_top_k": params.get("chunk_top_k", 8) + 10,  # MAXIMUM document chunks
                "max_entity_tokens": params.get("max_entity_tokens", 6000) + 3000,  # MAXIMUM entity detail
                "max_relation_tokens": params.get("max_relation_tokens", 3000) + 2000,  # MAXIMUM relationships
                "max_total_tokens": params.get("max_total_tokens", 10000) + 8000,  # MAXIMUM total content (18K tokens)
                "enable_rerank": True
            })
        
        return params

class AgenticRAGPipeline:
    """Agentic RAG Pipeline with hybrid retrieval and multilingual support"""
    
    def _load_env(self):
        """Load environment variables from .env file"""
        try:
            if os.path.exists('.env'):
                with open('.env', 'r') as f:
                    for line in f:
                        if '=' in line and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            value = value.strip('"').strip("'")
                            os.environ[key] = value
        except:
            pass
    
    def __init__(self, lightrag_api_url: str, openai_api_key: str = None, lightrag_api_key: str = None):
        # Load .env file first
        self._load_env()
        
        self.lightrag_client = LightRAGClient(lightrag_api_url)
        self.lightrag_api_key = lightrag_api_key or os.getenv("LIGHTRAG_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # System instructions for LightRAG API based on CSV patterns
        self.lightrag_system_instructions = self._create_lightrag_instructions()
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        if not self.lightrag_api_key:
            logger.warning("LightRAG API key not found - retrieval will fail")
        else:
            logger.info(f"LightRAG API key loaded: {self.lightrag_api_key[:5]}...")
        
        # Initialize Phoenix tracing
        self._setup_phoenix()
        
        # Initialize direct OpenAI client for GPT-5 nano (uses /v1/responses endpoint)
        from openai import OpenAI
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.model_name = "gpt-5-nano"  # GPT-5 nano: fastest, most cost-effective
        
        # Simple workflow - no LangGraph needed
        self.initialized = True
        
        # Initialize evaluators
        self._setup_evaluators()
        
    def _setup_phoenix(self):
        """Initialize Phoenix tracing"""
        try:
            # Launch Phoenix app
            session = px.launch_app()
            logger.info(f"Phoenix UI available at: {session.url}")
            
            # Register tracing
            tracer_provider = register(
                project_name="agentic-rag-pipeline",
                endpoint="http://localhost:6006/v1/traces"
            )
            
            self.tracer = trace.get_tracer(__name__)
            logger.info("Phoenix tracing initialized successfully")
            
        except Exception as e:
            logger.warning(f"Phoenix setup failed: {e}. Continuing without tracing.")
            self.tracer = None
    
    def _setup_evaluators(self):
        """Initialize Phoenix evaluators"""
        try:
            self.relevance_evaluator = RelevanceEvaluator()
            self.qa_evaluator = QAEvaluator()
            self.hallucination_evaluator = HallucinationEvaluator()
            logger.info("Evaluators initialized successfully")
        except Exception as e:
            logger.warning(f"Evaluator setup failed: {e}")
            self.relevance_evaluator = None
            self.qa_evaluator = None
            self.hallucination_evaluator = None
    
    def _detect_language(self, text: str) -> str:
        """Detect language of input text"""
        if detect is None:
            return "en"  # Default to English
        
        try:
            return detect(text)
        except (LangDetectError, Exception):
            return "en"
    
    def _translate_query(self, query: str, target_lang: str = "en") -> str:
        """Translate query to target language if needed"""
        detected_lang = self._detect_language(query)
        
        if detected_lang == target_lang:
            return query
        
        # Use GPT-5 nano for translation
        translation_prompt = f"""
        Translate the following query to {target_lang}. 
        Keep technical and financial terms accurate. 
        Only return the translation, nothing else.

        Query: {query}
        """
        
        try:
            # Use GPT-5 nano with /v1/responses endpoint
            response = self.openai_client.responses.create(
                model=self.model_name,
                reasoning={"effort": "low"},  # Fast translation
                input=translation_prompt
            )
            return response.output_text.strip()
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return query
    
    def _run_pipeline(self, question: str) -> AgenticAnswer:
        """Run the agentic pipeline steps sequentially - no LangGraph needed"""
        
        # Create simple state object
        state = AgenticRAGState(
            original_query=question,
            trace_id=f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Step 1: Classify query
        state = self._classify_query_node(state)
        
        # Step 2: Hybrid retrieval
        state = self._hybrid_retrieval_node(state)
        
        # Step 3: Generate answer
        state = self._generate_answer_node(state)
        
        # Step 4: Evaluate response
        state = self._evaluate_response_node(state)
        
        return state.answer
    
    def _classify_query_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Classify and analyze the input query"""
        with self.tracer.start_as_current_span("classify_query") if self.tracer else nullcontext():
            # Detect language
            detected_lang = self._detect_language(state.original_query)
            
            # Translate if needed
            if detected_lang != "en":
                state.translated_query = self._translate_query(state.original_query, "en")
            else:
                state.translated_query = state.original_query
            
            # Classify query using GPT-5 nano
            classification_prompt = f"""
            Analyze the following query and classify it according to the schema.
            
            Query: {state.translated_query}
            
            Classify the intent, complexity, and extract key information:
            - Intent: analysis, comparison, time_series, factual, calculation, research, other
            - Complexity: simple, medium, complex
            - Whether it requires multiple documents
            - Whether it's time-sensitive
            - Key entities mentioned (people, companies, places, concepts, etc.)
            
            Respond with a JSON object in this exact format:
            {{
                "intent": "analysis",
                "complexity": "medium",
                "requires_multi_doc": true,
                "time_sensitive": false,
                "entities": ["Apple", "Microsoft"]
            }}
            """
            
            try:
                # Use GPT-5 nano with /v1/responses endpoint
                response = self.openai_client.responses.create(
                    model=self.model_name,
                    reasoning={"effort": "low"},  # Fast classification
                    input=classification_prompt
                )
                
                # Parse JSON response
                import json
                classification_data = json.loads(response.output_text.strip())
                
                state.query_classification = QueryClassification(
                    intent=classification_data.get("intent", "other"),
                    complexity=classification_data.get("complexity", "medium"),
                    requires_multi_doc=classification_data.get("requires_multi_doc", False),
                    time_sensitive=classification_data.get("time_sensitive", False),
                    language=detected_lang,
                    entities=classification_data.get("entities", [])
                )
                
                logger.info(f"Query classified: {state.query_classification.intent}, complexity: {state.query_classification.complexity}")
                
            except Exception as e:
                logger.error(f"Query classification failed: {e}")
                # Fallback classification
                state.query_classification = QueryClassification(
                    intent="other",
                    complexity="medium",
                    requires_multi_doc=False,
                    time_sensitive=False,
                    language=detected_lang,
                    entities=[]
                )
        
        return state
    
    def _intelligent_mode_selection(self, query: str, query_classification) -> str:
        """Use GPT-5 nano to intelligently choose between 'global', 'hybrid', and 'mix' retrieval modes"""
        try:
            mode_selection_prompt = f"""
            You are a retrieval mode selector for a document analysis RAG system. Choose the optimal LightRAG retrieval mode for this query.

            RETRIEVAL MODES:
            - "global": Best for broad, comprehensive questions needing extensive document coverage and cross-document analysis
            - "hybrid": Best for specific, targeted questions where entity relationships and focused retrieval are key
            - "mix": Best for comprehensive analysis requiring BOTH broad coverage AND targeted relationships (dual retrieval)

            QUERY ANALYSIS:
            Query: "{query}"
            Intent: {query_classification.intent}
            Complexity: {query_classification.complexity}
            Multi-doc: {query_classification.requires_multi_doc}

            DECISION CRITERIA:
            - Use "global" for: broad thematic analysis, cross-document comparisons, high-level overviews, industry trends
            - Use "hybrid" for: specific numerical data queries, precise metrics extraction, quarterly/annual financial data, tax rates, earnings data, single-entity focused questions
            - Use "mix" for: comprehensive analysis, time-series comparisons, multi-period data requiring both precise numbers AND broader context (PREFERRED for comprehensive questions)

            OUTPUT: Return ONLY the mode name: "global", "hybrid", or "mix"
            """
            
            # Use GPT-5 nano with /v1/responses endpoint
            response = self.openai_client.responses.create(
                model=self.model_name,
                reasoning={"effort": "low"},  # Fast mode selection
                input=mode_selection_prompt
            )
            
            selected_mode = response.output_text.strip().lower()
            
            # Validate and default to mix if unclear
            if selected_mode in ["global", "hybrid", "mix"]:
                logger.info(f"🤖 GPT-5 nano selected retrieval mode: {selected_mode}")
                return selected_mode
            else:
                logger.warning(f"Invalid mode selection '{selected_mode}', defaulting to mix")
                return "mix"
                
        except Exception as e:
            logger.error(f"Mode selection failed: {e}, defaulting to global")
            return "global"
    
    def _hybrid_retrieval_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Perform hybrid retrieval using LightRAG with optimal parameters"""
        with self.tracer.start_as_current_span("hybrid_retrieval") if self.tracer else nullcontext():
            # 🤖 Intelligent mode selection based on query content
            query_to_use = state.translated_query or state.original_query
            selected_mode = self._intelligent_mode_selection(query_to_use, state.query_classification)
            
            # Get optimal parameters with intelligently selected mode
            optimal_params = self.lightrag_client.get_optimal_params(selected_mode, state.query_classification)
            
            # Debug logging
            logger.info(f"🔍 Query classification: intent={state.query_classification.intent}, complexity={state.query_classification.complexity}")
            logger.info(f"🤖 Selected retrieval mode: {selected_mode}")
            logger.info(f"🔧 LightRAG parameters: mode={optimal_params.get('mode')}, top_k={optimal_params.get('top_k')}, chunk_top_k={optimal_params.get('chunk_top_k')}")
            
            # Query LightRAG with optimal parameters, API key, and system instructions
            query_to_use = state.translated_query or state.original_query
            logger.info(f"📝 Query to LightRAG: '{query_to_use}'")
            
            response = self.lightrag_client.query(
                query=query_to_use,
                api_key=self.lightrag_api_key,
                user_prompt=self.lightrag_system_instructions,
                **optimal_params
            )
            
            if "error" in response:
                logger.error(f"LightRAG retrieval failed: {response['error']}")
                raise RuntimeError(f"LightRAG retrieval failed: {response['error']}. Fix your LightRAG API key!")
            else:
                # Extract sources from response
                sources = self._extract_sources(response.get("response", ""))
                
                # Get additional context if available
                context_info = response.get("context", "")
                full_content = response.get("response", "")
                
                # If context is available, combine it with response
                if context_info and context_info != full_content:
                    full_content = f"{full_content}\n\nAdditional Context:\n{context_info}"
                
                # DEBUG: Log raw LightRAG response to understand data discrepancies
                logger.info(f"🔍 DEBUG - Raw LightRAG content (first 500 chars): {full_content[:500]}...")
                logger.info(f"🔍 DEBUG - LightRAG sources: {sources}")
                logger.info(f"🔍 DEBUG - LightRAG mode used: {optimal_params.get('mode', 'mix')}")
                
                # Calculate actual confidence based on retrieval quality
                actual_confidence = self._calculate_retrieval_confidence(
                    response, sources, full_content, optimal_params
                )
                
                state.retrieval_context = RetrievalContext(
                    content=full_content,
                    sources=sources,
                    query_mode=optimal_params["mode"],  # Should always have mode
                    language=state.query_classification.language,
                    confidence=actual_confidence
                )
                
                logger.info(f"Retrieved content using {optimal_params['mode']} mode, found {len(sources)} sources")
        
        return state
    
    def _calculate_retrieval_confidence(self, response: Dict, sources: List[str], content: str, params: Dict) -> float:
        """Calculate actual confidence based on retrieval quality metrics"""
        confidence = 0.0
        
        # Base confidence from LightRAG if available
        if response.get("confidence"):
            confidence = float(response["confidence"])
        else:
            # Calculate confidence based on retrieval quality
            confidence = 0.5  # Start with medium confidence
            
            # Boost confidence based on factors:
            if sources and len(sources) > 0:
                confidence += 0.2  # Found sources
            if content and len(content.strip()) > 100:
                confidence += 0.1  # Substantial content
            if params.get("mode") in ["hybrid", "mix"]:
                confidence += 0.1  # Using advanced retrieval modes
            if "error" not in response:
                confidence += 0.1  # No API errors
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _generate_answer_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Generate comprehensive answer with reasoning"""
        with self.tracer.start_as_current_span("generate_answer") if self.tracer else nullcontext():
            # Determine response language
            response_lang = state.query_classification.language
            
            generation_prompt = f"""
            Organize and summarize the following document analysis. 
            
            RULES:
            - Keep the output brief and concise and to the point. Max 250 words STRICTLY
            1. Keep ALL information, numbers, and insights exactly as provided
            2. Remove any internal references like (DC: filename; KG: Entity X)
            3. Organize information clearly with proper structure
            4. Use numbered lists for data points and factors
            5. Add proper SOURCE(S): line at the end with actual PDF names
            6. Do NOT add any external knowledge or analysis
            7. Convert any .json references to .pdf in citations
            
            DOCUMENT ANALYSIS TO CLEAN UP:
            {state.retrieval_context.content}
            
            Task: Clean up the structure and remove metadata, but keep all the actual content and data.
            """
            
            try:
                # Use GPT-5 nano with /v1/responses endpoint
                response = self.openai_client.responses.create(
                    model=self.model_name,
                    reasoning={"effort": "medium"},  # Medium effort for formatting
                    input=generation_prompt
                )
                
                # Ensure proper SOURCE(S) format at the end
                answer_text = response.output_text.strip()
                sources = self._extract_sources(answer_text)
                
                # If no proper SOURCE(S) line at the end, add it
                if not answer_text.endswith(')') and 'SOURCE(S):' not in answer_text[-100:]:
                    if sources:
                        # Convert .json to .pdf and format properly
                        pdf_sources = []
                        for source in sources:
                            if source.endswith('.json'):
                                pdf_sources.append(source.replace('.json', '.pdf'))
                            elif source.endswith('.pdf'):
                                pdf_sources.append(source)
                            else:
                                pdf_sources.append(f"{source}.pdf")
                        
                        source_line = f"\n\nSOURCE(S): {', '.join(pdf_sources)}"
                        answer_text += source_line
                
                # Apply consistent formatting to the answer
                formatted_answer = self._format_response(answer_text, state.query_classification)
                
                # Create structured answer
                state.answer = AgenticAnswer(
                    answer=formatted_answer,
                    sources=sources,
                    confidence=state.retrieval_context.confidence,
                    reasoning="Generated using agentic RAG with hybrid retrieval and multilingual support",
                    query_classification=state.query_classification,
                    retrieval_mode=state.retrieval_context.query_mode,
                    language=response_lang
                )
                
                logger.info(f"Answer generated in {response_lang} with {len(state.answer.sources)} sources")
                
            except Exception as e:
                logger.error(f"Answer generation failed: {e}")
                state.answer = AgenticAnswer(
                    answer=f"Error generating answer: {e}",
                    sources=[],
                    confidence=0.0,
                    reasoning="Error in generation",
                    query_classification=state.query_classification,
                    retrieval_mode=state.retrieval_context.query_mode,
                    language=response_lang
                )
        
        return state
    
    def _create_lightrag_instructions(self) -> str:
        """LightRAG instructions: Analyze and reason with retrieved documents"""
        return """
        You are an intelligent document analyst. Use ONLY the information from the retrieved documents to provide comprehensive analysis.

        ANALYZE THE RETRIEVED DOCUMENTS:
        - Extract ALL relevant information that answers the user's specific question
        - For numerical data queries: Find exact figures, percentages, and metrics from tables and financial statements
        - Compare data across time periods when multiple periods are present  
        - Look for specific quarterly/annual data in financial tables and consolidated statements
        - Calculate trends, changes, and patterns from the retrieved data
        - Identify key factors, drivers, and relationships mentioned in the documents
        - Include specific numbers, percentages, dates, and precise details when available
        - Present data chronologically when dates are available
        - When looking for tax rates, find effective tax rate data from income statements or tax footnotes
        - Provide reasoning and insights based on what you find in the documents
        - Be thorough in extracting all relevant information that addresses the user's question

        CRITICAL RULES:
        - ONLY use information explicitly found in the retrieved documents
        - Do NOT add external knowledge beyond what's in the documents
        - Always include source file names for each piece of data
        - Provide exact details when available (numbers, dates, names, etc.)
        - Compare information across documents when multiple sources are available
        - If documents mention causes, effects, or relationships, include those in your analysis

        OUTPUT: Comprehensive analysis based ONLY on retrieved document content with source citations.
        """
    
    def _format_response(self, raw_response: str, query_classification: QueryClassification) -> str:
        """Stage 2: Clean up LightRAG metadata and improve readability"""
        try:
            logger.info("🎯 Stage 2: Cleaning up LightRAG metadata")
            
            formatted = raw_response.strip()
            
            # Remove internal LightRAG metadata 
            formatted = self._clean_lightrag_metadata(formatted)
            
            # Clean up source citations for consistency
            formatted = self._clean_lightrag_sources_only(formatted)
            
            logger.info(f"✅ Stage 2: Cleaned response ({len(formatted)} chars)")
            return formatted
            
        except Exception as e:
            logger.error(f"Stage 2 formatting failed: {e}")
            return raw_response
    
    def _restructure_lightrag_content_only(self, text: str) -> str:
        """ZERO MODEL KNOWLEDGE: Only restructure LightRAG retrieval content to bullets"""
        lines = []
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        for sentence in sentences:
            # ONLY restructure - NO content interpretation or addition
            
            # Convert financial data points to bullets (pure structural change)
            if any(keyword in sentence.lower() for keyword in 
                  ['quarter ended', 'million', '$', 'percent', '%', 'billion', 'revenue', 'sales']):
                lines.append(f"- {sentence}.")
            else:
                # Keep as regular text - NO model interpretation
                lines.append(f"{sentence}.")
        
        return '\n'.join(lines)
    
    def _clean_lightrag_metadata(self, text: str) -> str:
        """Remove internal LightRAG metadata like DC:, KG: references"""
        # Remove DC (Document Chunk) references like "(DC: 2023 Q3 AAPL.page_014; KG: Entity 36)"
        text = re.sub(r'\(DC:[^)]+\)', '', text)
        
        # Remove KG (Knowledge Graph) references like "(KG: Entity 22, 38)"  
        text = re.sub(r'\(KG:[^)]+\)', '', text)
        
        # Remove mixed references like "(DC: filename; KG: Entity X)"
        text = re.sub(r'\([^)]*(?:DC|KG):[^)]*\)', '', text)
        
        # Clean up extra spaces and semicolons left behind
        text = re.sub(r'\s*;\s*', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _clean_lightrag_sources_only(self, text: str) -> str:
        """ZERO MODEL KNOWLEDGE: Only clean source formatting from LightRAG output"""
        # ONLY clean formatting of sources that LightRAG already provided
        text = re.sub(r'\(source[s]?:\s*([^)]+)\)', r'(SOURCE: \1)', text, flags=re.IGNORECASE)
        
        # Collect ONLY existing sources from LightRAG (add nothing new)
        sources = re.findall(r'\(SOURCE:\s*([^)]+)\)', text, re.IGNORECASE)
        
        if sources:
            unique_sources = list(dict.fromkeys(sources))
            
            # Only add SOURCE(S) line if sources exist but no final line present
            if not re.search(r'SOURCE\(S\):', text, re.IGNORECASE):
                # Clean source format: remove dashes and extra formatting
                clean_sources = [source.strip('- ').strip() for source in unique_sources]
                text += f"\n\nSOURCE(S): {', '.join(clean_sources)}"
        
        return text
    
    def _spacing_cleanup_only(self, text: str) -> str:
        """ZERO MODEL KNOWLEDGE: Only fix spacing - NO content interpretation"""
        # PURE mechanical cleanup - no content decisions
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r'([.!?])\s*\n', r'\1\n', text)
        
        # Ensure proper bullet spacing (mechanical only)
        text = re.sub(r'\n-\s*', '\n\n- ', text)
        
        return text.strip()
    
    def _evaluate_response_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Evaluate the generated response"""
        with self.tracer.start_as_current_span("evaluate_response") if self.tracer else nullcontext():
            if not all([self.relevance_evaluator, self.qa_evaluator, self.hallucination_evaluator]):
                logger.warning("Evaluators not available, skipping evaluation")
                return state
            
            try:
                # Prepare evaluation data
                context = state.retrieval_context.content
                query = state.original_query
                response = state.answer.answer
                
                # Run evaluations
                # Phoenix Arize uses semantic similarity and LLM-based evaluation
                # to handle cases where content is similar but wording differs
                relevance_result = self.relevance_evaluator.evaluate(
                    input=query,
                    retrieval_context=context
                )
                
                qa_result = self.qa_evaluator.evaluate(
                    input=query,
                    context=context,
                    response=response
                )
                
                # Hallucination evaluation checks factual consistency
                # even with different wording styles
                hallucination_result = self.hallucination_evaluator.evaluate(
                    input=query,
                    context=context,
                    response=response
                )
                
                # Calculate source attribution score (simplified)
                source_attribution_score = len(state.answer.sources) / max(1, len(state.retrieval_context.sources))
                source_attribution_score = min(1.0, source_attribution_score)
                
                # Calculate overall score
                overall_score = (
                    relevance_result.score * 0.3 +
                    qa_result.score * 0.3 +
                    (1 - hallucination_result.score) * 0.3 +  # Invert hallucination score
                    source_attribution_score * 0.1
                )
                
                state.evaluation = EvaluationResult(
                    relevance_score=relevance_result.score,
                    answer_quality_score=qa_result.score,
                    hallucination_score=hallucination_result.score,
                    source_attribution_score=source_attribution_score,
                    overall_score=overall_score
                )
                
                logger.info(f"Evaluation complete - Overall score: {overall_score:.3f}")
                
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                state.evaluation = EvaluationResult(
                    relevance_score=0.0,
                    answer_quality_score=0.0,
                    hallucination_score=1.0,
                    source_attribution_score=0.0,
                    overall_score=0.0
                )
        
        return state
    
    def _extract_sources(self, response: str) -> List[str]:
        """Extract source citations from LightRAG response with support for KG and DC references"""
        import re
        sources = []
        
        # LightRAG-specific reference patterns (like in your example)
        lightrag_patterns = [
            # Knowledge Graph references: [KG] Entities (id: 16, 2023 Q3 AAPL.json; 2023 Q1 AAPL.json)
            r'\[KG\][^(]*\([^:]*:\s*[^,]*,\s*([^)]+)\)',
            # Document Chunk references: [DC] Page 20 from "2023 Q2 AAPL.pdf"
            r'\[DC\][^"]*"([^"]+\.pdf)"',
            # Extract documents from KG entity lists
            r'(\d{4}\s+Q\d\s+[A-Z]+\.(?:json|pdf))',
            r'([A-Z]+\s+Q\d\s+\d{4}\.(?:json|pdf))',
        ]
        
        # Standard citation patterns for fallback
        standard_patterns = [
            # Standard SOURCE: patterns
            r'SOURCE:\s*([^,\n\)]+\.pdf)',
            r'SOURCE\(S\):\s*([^,\n\)]+\.pdf[^,\n\)]*)',
            # Parenthetical citations
            r'\(SOURCE:\s*([^,\n\)]+\.pdf[^)]*)\)',
            r'\(([^)]*(?:Q\d|AAPL|MSFT|AMZN|INTC|NVDA)[^)]*\.pdf[^)]*)\)',
            # Financial document patterns
            r'(\d{4}\s+Q\d\s+[A-Z]+\.pdf)',
            r'([A-Z]+\s+Q\d\s+\d{4}\.pdf)',
            # Invoice and other document patterns
            r'(\d+_[^,\s]+\.pdf)',
            r'([A-Za-z]+[-_][A-Za-z0-9]+\.pdf)',
        ]
        
        # First try LightRAG-specific patterns
        for pattern in lightrag_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                
                # Handle semicolon-separated lists (from KG references)
                if ';' in match:
                    for sub_match in match.split(';'):
                        clean_match = sub_match.strip().rstrip(',').rstrip('.')
                        if clean_match and ('.pdf' in clean_match or '.json' in clean_match):
                            # Convert .json to .pdf for consistency
                            if clean_match.endswith('.json'):
                                clean_match = clean_match.replace('.json', '.pdf')
                            sources.append(clean_match)
                else:
                    clean_match = match.strip().rstrip(',').rstrip('.')
                    if clean_match and ('.pdf' in clean_match or '.json' in clean_match):
                        if clean_match.endswith('.json'):
                            clean_match = clean_match.replace('.json', '.pdf')
                        sources.append(clean_match)
        
        # Then try standard patterns as fallback
        for pattern in standard_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                
                clean_match = match.strip().rstrip(',').rstrip(';').rstrip('.')
                if clean_match and clean_match.endswith('.pdf'):
                    sources.append(clean_match)
        
        # Also extract from SOURCE(S): lists
        source_list_pattern = r'SOURCE\(S\):\s*([^.]+)'
        source_list_matches = re.findall(source_list_pattern, response, re.IGNORECASE)
        for match in source_list_matches:
            individual_sources = re.findall(r'([^,]+\.pdf)', match)
            sources.extend([s.strip() for s in individual_sources])
        
        # Remove duplicates and filter valid sources
        unique_sources = []
        for source in sources:
            if source not in unique_sources and len(source) > 5:
                unique_sources.append(source)
        
        return unique_sources
    
    def query(self, question: str) -> Dict[str, Any]:
        """Main query interface - simplified without LangGraph"""
        logger.info(f"Processing agentic query: {question[:100]}...")
        
        try:
            # Run the simple pipeline
            with using_project("agentic-rag-pipeline") if self.tracer else nullcontext():
                logger.info("Running agentic pipeline...")
                answer = self._run_pipeline(question)
                logger.info("Pipeline completed successfully")
            
            # Return in the format expected by the test
            return {
                'answer': answer.answer,
                'context': ', '.join(answer.sources) if answer.sources else 'No context retrieved',
                'retrieval_mode': answer.retrieval_mode,
                'query_classification': {
                    'intent': answer.query_classification.intent,
                    'complexity': answer.query_classification.complexity
                }
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return error response
            return {
                'answer': f"Error: {str(e)}",
                'context': "No context due to error",
                'retrieval_mode': 'error',
                'query_classification': {
                    'intent': 'error',
                    'complexity': 'error'
                }
            }

# Utility function for nullcontext (Python < 3.7 compatibility)
try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager
    
    @contextmanager
    def nullcontext():
        yield

def load_and_test_questions(pipeline: AgenticRAGPipeline, csv_path: str, num_questions: int = 3):
    """Load questions from CSV and test the pipeline"""
    try:
        df = pd.read_csv(csv_path)
        questions = df['Question'].tolist()[:num_questions]
        
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"🧪 Testing Question {i}: {question}")
            print('='*60)
            
            try:
                answer = pipeline.query(question)
                
                print(f"\n📝 Answer ({answer.language}):")
                print(answer.answer)
                
                print(f"\n📚 Sources ({len(answer.sources)}):")
                for source in answer.sources:
                    print(f"  • {source}")
                
                print(f"\n🎯 Metrics:")
                print(f"  • Intent: {answer.query_classification.intent}")
                print(f"  • Complexity: {answer.query_classification.complexity}")
                print(f"  • Retrieval Mode: {answer.retrieval_mode}")
                print(f"  • Confidence: {answer.confidence:.3f}")
                print(f"  • Language: {answer.language}")
                
                results.append({
                    'question': question,
                    'answer': answer.answer,
                    'sources_count': len(answer.sources),
                    'confidence': answer.confidence,
                    'retrieval_mode': answer.retrieval_mode
                })
                
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                results.append({
                    'question': question,
                    'error': str(e)
                })
        
        return results
        
    except Exception as e:
        print(f"❌ Error loading questions: {e}")
        return []

def main():
    """Main function to demonstrate the agentic RAG pipeline"""
    print("🤖 Initializing Agentic RAG Pipeline")
    print("="*50)
    
    # Initialize pipeline
    try:
        pipeline = AgenticRAGPipeline(
            lightrag_api_url="https://lightrag-h7h2g9atabdjg3af.canadacentral-01.azurewebsites.net",
        )
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Test multilingual support
    multilingual_queries = [
        "What is Apple's net sales for Q3 2022?",  # English
        "¿Cuáles fueron las ventas netas de Apple en Q3 2022?",  # Spanish
        "Quelles étaient les ventes nettes d'Apple au Q3 2022?",  # French
    ]
    
    print(f"\n🌍 Testing Multilingual Support")
    print("-" * 40)
    
    for query in multilingual_queries:
        print(f"\n🔍 Query: {query}")
        try:
            answer = pipeline.query(query)
            print(f"📝 Answer ({answer.language}): {answer.answer[:200]}...")
            print(f"🎯 Confidence: {answer.confidence:.3f}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test with CSV questions
    print(f"\n📋 Testing with CSV Questions")
    print("-" * 40)
    
    results = load_and_test_questions(
        pipeline, 
        '/Users/bhavay/Desktop/Aparavi/questions_with_partial_answers.csv',
        num_questions=2
    )
    
    print(f"\n📊 Summary: Processed {len(results)} questions")
    print("🔗 Check Phoenix UI for detailed traces and evaluations")

if __name__ == "__main__":
    main()
