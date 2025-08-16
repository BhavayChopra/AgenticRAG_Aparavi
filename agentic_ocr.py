"""
Agentic OCR Framework using AutoGen for intelligent parsing decisions.
Decides between standard parsing (Unstructured.io) and OCR parsing (LlamaParse).
"""

import json
import os
import tempfile
from typing import Dict, Any, Optional, Tuple
from dataclasses import asdict
import autogen
from unstructured.partition.pdf import partition_pdf
from llama_parse import LlamaParse
from config import load_config
from page_assessment import PageSignals, analyze_page_with_unstructured


class AgenticOCRFramework:
    """Main framework for agentic OCR processing."""
    
    def __init__(self):
        """Initialize the framework with configuration and agents."""
        self.config = load_config()
        self._setup_agents()
        self._setup_parsers()
    
    def _setup_agents(self):
        """Set up AutoGen agents for decision making."""
        
        # Configuration for the LLM using OpenAI GPT-4.1-mini
        llm_config = {
            "config_list": [
                {
                    "model": "gpt-4.1-mini",
                    "api_key": self.config.openai_api_key,
                    "api_type": "openai"
                }
            ],
            "temperature": 0.1,
            "timeout": 30,
        }
        
        # System prompt for the extraction mode selector
        system_prompt = """You are an extraction mode selector for multi-modal documents. Your job: Decide if a document page should be processed with fast text extraction or OCR-based extraction. Choose ONLY one tool: 'standard_parse' for machine-readable PDFs with good text density and minimal images. 'ocr_parse' for scanned/image-based pages, or low-density text pages with heavy tables/images. You will be given page signals and must respond with a single JSON: {"decision": "standard_parse", "reason": "<brief reason>"} or {"decision": "ocr_parse", "reason": "<brief reason>"}."""
        
        # Assistant agent for making parsing decisions
        self.assistant = autogen.AssistantAgent(
            name="extraction_selector",
            system_message=system_prompt,
            llm_config=llm_config,
        )
        
        # User proxy agent for managing the conversation
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
        )
        
        # Register the parsing functions as tools
        self.user_proxy.register_function(
            function_map={
                "standard_parse": self._standard_parse,
                "ocr_parse": self._ocr_parse,
            }
        )
    
    def _setup_parsers(self):
        """Set up the parsing engines."""
        # LlamaParse for OCR parsing (optional)
        if self.config.llama_parse.api_key:
            try:
                self.llama_parser = LlamaParse(
                    api_key=self.config.llama_parse.api_key,
                    result_type="text",
                    num_workers=1,
                    verbose=False,
                    language="en",
                    parsing_instruction="Extract all text content including tables, preserving structure."
                )
                print("✓ LlamaParse initialized successfully")
            except Exception as e:
                print(f"⚠️  LlamaParse initialization failed: {e}")
                self.llama_parser = None
        else:
            print("⚠️  LlamaParse API key not provided - OCR parsing will not be available")
            self.llama_parser = None
    
    def _standard_parse(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """
        Standard parsing using Unstructured.io (no OCR).
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to parse (0-indexed)
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Use Unstructured.io for standard parsing
            elements = partition_pdf(
                filename=pdf_path,
                strategy="fast",
                extract_images_in_pdf=False,
                infer_table_structure=True,
                chunking_strategy=None,
                include_page_breaks=True
            )
            
            # Filter elements for the specific page
            page_elements = []
            for element in elements:
                if hasattr(element, 'metadata') and element.metadata:
                    element_page = element.metadata.get('page_number', 0)
                    if element_page == page_num + 1 or element_page == page_num:
                        page_elements.append(element)
                elif page_num == 0:  # If no page metadata, assume it's from page 0
                    page_elements.append(element)
            
            # If we didn't get page-specific elements, fall back to all elements (for single page docs)
            if not page_elements and page_num == 0:
                page_elements = elements
            
            # Extract text content
            text_content = "\n".join([elem.text for elem in page_elements if hasattr(elem, 'text') and elem.text])
            
            # Extract metadata
            metadata = {
                "parsing_method": "standard_parse",
                "element_count": len(page_elements),
                "total_elements": len(elements),
                "text_length": len(text_content),
                "elements": [{"type": getattr(elem, 'category', 'unknown'), "text": elem.text[:100]} for elem in page_elements[:5] if hasattr(elem, 'text') and elem.text]  # First 5 elements preview
            }
            
            return {
                "text": text_content,
                "metadata": metadata,
                "success": True,
                "error": None
            }
        
        except Exception as e:
            return {
                "text": "",
                "metadata": {"parsing_method": "standard_parse"},
                "success": False,
                "error": str(e)
            }
    
    def _ocr_parse(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """
        OCR parsing using LlamaParse.
        Note: LlamaParse processes the entire document, so we extract the relevant page content.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to parse (0-indexed)
            
        Returns:
            Dictionary with extracted text and metadata
        """
        # Check if LlamaParse is available
        if not self.llama_parser:
            return {
                "text": "",
                "metadata": {"parsing_method": "ocr_parse"},
                "success": False,
                "error": "LlamaParse not available - API key not configured"
            }
        
        try:
            # Use LlamaParse for OCR parsing (processes entire document)
            documents = self.llama_parser.load_data(pdf_path)
            
            # Extract text content from all documents
            all_text = "\n".join([doc.text for doc in documents])
            
            # For now, we'll return all text since LlamaParse doesn't easily separate by page
            # In a production system, you might want to implement page splitting logic
            text_content = all_text
            
            # If we have multiple pages and this isn't page 0, try to estimate page content
            if page_num > 0 and len(all_text) > 1000:
                # Simple heuristic: split text roughly by page number
                # This is approximate since we don't have exact page boundaries
                lines = all_text.split('\n')
                lines_per_page = max(1, len(lines) // (page_num + 1))
                start_line = page_num * lines_per_page
                end_line = (page_num + 1) * lines_per_page
                text_content = '\n'.join(lines[start_line:end_line])
            
            # Extract metadata
            metadata = {
                "parsing_method": "ocr_parse",
                "document_count": len(documents),
                "text_length": len(text_content),
                "total_text_length": len(all_text),
                "page_estimation": "approximate" if page_num > 0 else "full_document",
                "llama_parse_metadata": documents[0].metadata if documents else {}
            }
            
            return {
                "text": text_content,
                "metadata": metadata,
                "success": True,
                "error": None
            }
        
        except Exception as e:
            return {
                "text": "",
                "metadata": {"parsing_method": "ocr_parse"},
                "success": False,
                "error": str(e)
            }
    
    def decide_parsing_method(self, page_signals: PageSignals) -> Tuple[str, str]:
        """
        Use AutoGen agent to decide parsing method based on page signals.
        
        Args:
            page_signals: Page analysis signals
            
        Returns:
            Tuple of (decision, reason)
        """
        # User prompt template
        user_prompt = f"""You are given document page analysis:
- Characters: {page_signals.char_count}
- Text blocks: {page_signals.block_count}
- Images: {page_signals.image_count}
- Text density: {page_signals.density:.2f}
- Language: {page_signals.language}
- Heuristic reason: {page_signals.heuristic_reason}

Decide extraction method:
- 'standard_parse' → Unstructured.io Python library (no OCR)
- 'ocr_parse' → LlamaParse 10cred Balanced OCR mode

Output ONLY JSON as per system instructions."""
        
        # Get decision from the assistant agent
        chat_result = self.user_proxy.initiate_chat(
            self.assistant,
            message=user_prompt,
            max_turns=1,
        )
        
        # Parse the response
        try:
            # Get the last message from assistant
            last_message = chat_result.chat_history[-1]["content"]
            
            # Try to extract JSON from the message
            decision_data = json.loads(last_message)
            decision = decision_data.get("decision", "standard_parse")
            reason = decision_data.get("reason", "default decision")
            
            # Validate decision
            if decision not in ["standard_parse", "ocr_parse"]:
                decision = "standard_parse"
                reason = f"invalid decision '{decision}', defaulting to standard_parse"
            
            return decision, reason
        
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # Fallback decision
            return "standard_parse", f"failed to parse agent response: {str(e)}"
    
    def process_page(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """
        Process a single page: assess, decide, and parse.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to process (0-indexed)
            
        Returns:
            Complete processing result with signals, decision, and parsed content
        """
        # Step 1: Assess page signals
        try:
            page_signals = analyze_page_with_unstructured(pdf_path, page_num)
        except Exception as e:
            return {
                "pdf_path": pdf_path,
                "page_num": page_num,
                "signals": None,
                "decision": "standard_parse",
                "reason": f"assessment failed: {str(e)}",
                "parsed_content": {"text": "", "metadata": {}, "success": False, "error": str(e)},
                "processing_error": str(e)
            }
        
        # Step 2: Get decision from agent
        try:
            decision, reason = self.decide_parsing_method(page_signals)
        except Exception as e:
            decision, reason = "standard_parse", f"decision failed: {str(e)}"
        
        # Step 3: Execute parsing
        try:
            if decision == "ocr_parse":
                parsed_content = self._ocr_parse(pdf_path, page_num)
            else:
                parsed_content = self._standard_parse(pdf_path, page_num)
        except Exception as e:
            parsed_content = {
                "text": "",
                "metadata": {"parsing_method": decision},
                "success": False,
                "error": str(e)
            }
        
        return {
            "pdf_path": pdf_path,
            "page_num": page_num,
            "signals": asdict(page_signals),
            "decision": decision,
            "reason": reason,
            "parsed_content": parsed_content,
            "processing_error": None
        }
