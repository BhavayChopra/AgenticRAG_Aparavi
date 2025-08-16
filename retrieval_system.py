#!/usr/bin/env python3
"""
Multi-modal RAG system with LangGraph and Phoenix tracing
Handles text, tables, and citations for financial document queries
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

# Core dependencies
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# LangGraph for workflow orchestration
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

# Phoenix for tracing and evaluation
import phoenix as px
from phoenix.otel import register
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalResult(BaseModel):
    """Structure for retrieval results"""
    content: str = Field(description="Retrieved content")
    source_doc: str = Field(description="Source document name")
    page_number: int = Field(description="Page number")
    chunk_id: str = Field(description="Unique chunk identifier")
    relevance_score: float = Field(description="Relevance score")
    chunk_type: str = Field(description="Type of content (text, table, etc.)")

class QueryAnswer(BaseModel):
    """Structure for final answer"""
    answer: str = Field(description="Final answer to the question")
    sources: List[str] = Field(description="List of source citations")
    confidence: float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="Explanation of how answer was derived")

class RAGState(BaseModel):
    """State for the RAG workflow"""
    question: str
    retrieved_docs: List[RetrievalResult] = []
    answer: Optional[QueryAnswer] = None
    messages: Annotated[list, add_messages] = []

class MultiModalRetriever:
    """Multi-modal retriever for text and tables with Phoenix tracing"""
    
    def __init__(self, lightrag_api_url: str = None, openai_api_key: str = None):
        self.lightrag_api_url = lightrag_api_url
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize Phoenix tracing
        self._setup_phoenix()
        
        # Initialize components
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=self.openai_api_key
        )
        
        self.embeddings = OpenAIEmbeddings(
            api_key=self.openai_api_key
        )
        
        # Load processed documents
        self.documents = self._load_documents()
        self.vectorstore = self._create_vectorstore()
        
        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()
        
    def _setup_phoenix(self):
        """Initialize Phoenix tracing"""
        try:
            # Start Phoenix locally
            px.launch_app()
            
            # Register OpenTelemetry tracing
            tracer_provider = register(
                project_name="lightrag-retrieval",
                endpoint="http://localhost:6006/v1/traces"
            )
            
            logger.info("Phoenix tracing initialized successfully")
        except Exception as e:
            logger.warning(f"Phoenix setup failed: {e}. Continuing without tracing.")
    
    def _load_documents(self) -> List[Document]:
        """Load all documents from lightrag_style directory"""
        docs = []
        lightrag_dir = Path('/Users/bhavay/Desktop/Aparavi/parsed_results/lightrag_style')
        
        for json_file in lightrag_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                
                for chunk in doc_data:
                    # Determine chunk type (text vs table)
                    content = chunk.get('content', '')
                    chunk_type = self._classify_content_type(content)
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source_doc': chunk['doc_id'],
                            'chunk_id': chunk['chunk_id'],
                            'page_number': chunk['metadata']['page_number'],
                            'parsing_method': chunk['metadata']['parsing_method'],
                            'chunk_type': chunk_type,
                            'element_count': chunk['metadata']['element_count'],
                            'text_length': chunk['metadata']['text_length']
                        }
                    )
                    docs.append(doc)
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {len(docs)} document chunks")
        return docs
    
    def _classify_content_type(self, content: str) -> str:
        """Classify content as text, table, or other"""
        # Simple heuristics for content classification
        if any(indicator in content.lower() for indicator in [
            'consolidated statements', 'balance sheet', 'cash flows',
            'operations', 'comprehensive income', 'table', '$', 'million'
        ]):
            # Check for table-like patterns
            lines = content.split('\n')
            numeric_lines = sum(1 for line in lines if any(char.isdigit() for char in line))
            if numeric_lines / len(lines) > 0.3:  # 30% of lines contain numbers
                return 'table'
        
        return 'text'
    
    def _create_vectorstore(self) -> FAISS:
        """Create FAISS vectorstore from documents"""
        if not self.documents:
            raise ValueError("No documents loaded")
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(
            documents=self.documents,
            embedding=self.embeddings
        )
        
        logger.info("Vector store created successfully")
        return vectorstore
    
    def retrieve_documents(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query"""
        # Get similar documents
        similar_docs = self.vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in similar_docs:
            result = RetrievalResult(
                content=doc.page_content,
                source_doc=doc.metadata['source_doc'],
                page_number=doc.metadata['page_number'],
                chunk_id=doc.metadata['chunk_id'],
                relevance_score=1 - score,  # Convert distance to similarity
                chunk_type=doc.metadata['chunk_type']
            )
            results.append(result)
        
        return results
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for RAG"""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("evaluate", self._evaluate_node)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "evaluate")
        workflow.add_edge("evaluate", END)
        
        return workflow.compile()
    
    def _retrieve_node(self, state: RAGState) -> RAGState:
        """Retrieval node in the workflow"""
        retrieved_docs = self.retrieve_documents(state.question, k=10)
        state.retrieved_docs = retrieved_docs
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {state.question[:50]}...")
        return state
    
    def _generate_node(self, state: RAGState) -> RAGState:
        """Generation node in the workflow"""
        # Prepare context from retrieved documents
        context_parts = []
        sources = []
        
        for doc in state.retrieved_docs:
            context_parts.append(
                f"[SOURCE: {doc.source_doc}, Page {doc.page_number}, Type: {doc.chunk_type}]\n"
                f"{doc.content}\n"
            )
            sources.append(f"{doc.source_doc} (Page {doc.page_number})")
        
        context = "\n---\n".join(context_parts)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template("""
You are a financial analyst assistant. Answer the question based on the provided context.

IMPORTANT RULES:
1. Always cite your sources using the format: (SOURCE: document_name, Page X)
2. If the information comes from tables, mention that explicitly
3. Be precise with numbers and dates
4. If you cannot find specific information, say so clearly
5. Provide a comprehensive answer that addresses all aspects of the question

Context:
{context}

Question: {question}

Provide your answer with proper source citations:
        """)
        
        # Generate response
        messages = prompt.format_messages(context=context, question=state.question)
        response = self.llm.invoke(messages)
        
        # Extract unique sources
        unique_sources = list(set(sources))
        
        answer = QueryAnswer(
            answer=response.content,
            sources=unique_sources,
            confidence=0.8,  # TODO: Implement confidence scoring
            reasoning="Answer generated from retrieved document chunks with source citations"
        )
        
        state.answer = answer
        return state
    
    def _evaluate_node(self, state: RAGState) -> RAGState:
        """Evaluation node using Phoenix evaluators"""
        try:
            # Prepare data for evaluation
            context = "\n".join([doc.content for doc in state.retrieved_docs])
            
            # Relevance evaluation
            relevance_evaluator = RelevanceEvaluator()
            relevance_score = relevance_evaluator.evaluate(
                input=state.question,
                retrieval_context=context
            )
            
            # QA evaluation
            qa_evaluator = QAEvaluator()
            qa_score = qa_evaluator.evaluate(
                input=state.question,
                context=context,
                response=state.answer.answer if state.answer else ""
            )
            
            # Hallucination evaluation
            hallucination_evaluator = HallucinationEvaluator()
            hallucination_score = hallucination_evaluator.evaluate(
                input=state.question,
                context=context,
                response=state.answer.answer if state.answer else ""
            )
            
            logger.info(f"Evaluation scores - Relevance: {relevance_score}, QA: {qa_score}, Hallucination: {hallucination_score}")
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
        
        return state
    
    def query(self, question: str) -> QueryAnswer:
        """Main query interface"""
        initial_state = RAGState(question=question)
        final_state = self.workflow.invoke(initial_state)
        return final_state.answer

def load_test_questions(csv_path: str) -> List[Dict[str, str]]:
    """Load questions from CSV file"""
    df = pd.read_csv(csv_path)
    questions = []
    
    for _, row in df.iterrows():
        questions.append({
            'question': row['Question'],
            'expected_sources': row['Source Docs'],
            'question_type': row['Question Type'],
            'chunk_type': row['Source Chunk Type']
        })
    
    return questions

def main():
    """Main function to test the retrieval system"""
    # Initialize retriever
    retriever = MultiModalRetriever()
    
    # Load test questions
    questions = load_test_questions('/Users/bhavay/Desktop/Aparavi/questions_with_partial_answers.csv')
    
    # Test with first few questions
    for i, q in enumerate(questions[:3]):  # Test first 3 questions
        print(f"\n{'='*50}")
        print(f"Question {i+1}: {q['question']}")
        print(f"Expected sources: {q['expected_sources']}")
        print(f"Question type: {q['question_type']}")
        
        try:
            answer = retriever.query(q['question'])
            
            print(f"\nAnswer: {answer.answer}")
            print(f"\nSources found: {', '.join(answer.sources)}")
            print(f"Confidence: {answer.confidence}")
            print(f"Reasoning: {answer.reasoning}")
            
        except Exception as e:
            print(f"Error processing question: {e}")

if __name__ == "__main__":
    main()
