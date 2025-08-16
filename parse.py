"""
Simple parsing functions for standard and OCR parsing.
"""

import json
import os
from typing import List, Dict, Any
from unstructured.partition.pdf import partition_pdf
from llama_parse import LlamaParse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key with fallback
LLAMA_API_KEY = os.getenv("LLAMA_PARSE_API_KEY") or "YOUR_LLAMA_PARSE_API_KEY"

# Initialize LlamaParse
try:
    llama_parser = LlamaParse(
        api_key=LLAMA_API_KEY,
        result_type="text",
        num_workers=1,
        verbose=False,
        language="en"
    )
    print("✅ LlamaParse initialized successfully")
except Exception as e:
    print(f"⚠️  LlamaParse initialization failed: {e}")
    llama_parser = None


def standard_parse_page(filename: str, page_numbers: List[int]) -> Dict[str, Any]:
    """
    Parse PDF pages using Unstructured.io.
    
    Args:
        filename: PDF file path
        page_numbers: list of pages to parse (1-indexed)
        
    Returns:
        JSON object with parsed content
    """
    elements = partition_pdf(
        filename=filename,
        strategy="fast",
        extract_images_in_pdf=False,
        infer_table_structure=True,
        include_page_breaks=True
    )
    
    parsed_pages = []
    
    for page_num in page_numbers:
        page_elements = []
        
        for element in elements:
            if hasattr(element, 'metadata') and element.metadata:
                if hasattr(element.metadata, 'page_number'):
                    element_page = element.metadata.page_number
                    if element_page == page_num:
                        page_elements.append(element)
        
        # Extract text and metadata
        elements_data = []
        raw_text_parts = []
        
        for element in page_elements:
            if hasattr(element, 'text') and element.text:
                elements_data.append({
                    "type": getattr(element, 'category', 'unknown'),
                    "text": element.text
                })
                raw_text_parts.append(element.text)
        
        parsed_pages.append({
            "page_number": page_num,
            "elements": elements_data,
            "raw_text": "\n".join(raw_text_parts)
        })
    
    return {
        "filename": filename,
        "parsed_pages": parsed_pages
    }


def ocr_parse_page(filename: str, page_numbers: List[int]) -> Dict[str, Any]:
    """
    Parse PDF pages using LlamaParse OCR.
    
    Args:
        filename: PDF file path
        page_numbers: list of pages to parse (1-indexed)
        
    Returns:
        JSON object with parsed content
    """
    if not llama_parser:
        # Fallback to standard parsing if LlamaParse not available
        print("⚠️  LlamaParse not available, falling back to standard parsing")
        return standard_parse_page(filename, page_numbers)
    
    documents = llama_parser.load_data(filename)
    all_text = "\n".join([doc.text for doc in documents])
    
    parsed_pages = []
    
    for page_num in page_numbers:
        # Simple page estimation
        lines = all_text.split('\n')
        lines_per_page = max(1, len(lines) // max(page_numbers))
        start_line = (page_num - 1) * lines_per_page
        end_line = page_num * lines_per_page
        page_text = '\n'.join(lines[start_line:end_line])
        
        # Create elements
        elements_data = []
        if page_text.strip():
            paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
            for paragraph in paragraphs:
                elements_data.append({
                    "type": "NarrativeText",
                    "text": paragraph
                })
        
        parsed_pages.append({
            "page_number": page_num,
            "elements": elements_data,
            "raw_text": page_text
        })
    
    return {
        "filename": filename,
        "parsed_pages": parsed_pages
    }
