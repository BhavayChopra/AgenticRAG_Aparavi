"""
Page assessment module using Unstructured.io partition for analyzing PDF pages.
Collects signals to help decide between standard parsing and OCR parsing.
"""

import os
import tempfile
from typing import Dict, Any, List
from dataclasses import dataclass
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element


@dataclass
class PageSignals:
    """Page analysis signals for extraction method decision."""
    char_count: int
    block_count: int
    image_count: int
    density: float
    language: str
    heuristic_reason: str


def analyze_page_with_unstructured(pdf_path: str, page_num: int) -> PageSignals:
    """
    Analyze a single PDF page using Unstructured.io partition to collect signals.
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to analyze (0-indexed)
        
    Returns:
        PageSignals object with analysis results
    """
    try:
        # Use Unstructured.io to partition the entire PDF with page numbers
        elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",  # Fast strategy for assessment
            extract_images_in_pdf=False,  # Don't extract images for assessment
            infer_table_structure=False,  # Skip table inference for speed
            chunking_strategy=None,  # No chunking needed for assessment
            include_page_breaks=True,  # Include page break information
        )
        
        # Filter elements for the specific page
        page_elements = []
        for element in elements:
            # Check if element has page_number metadata
            if hasattr(element, 'metadata') and element.metadata:
                try:
                    element_page = element.metadata.get('page_number', 0) if hasattr(element.metadata, 'get') else getattr(element.metadata, 'page_number', 0)
                    # Convert to 0-indexed if it's 1-indexed
                    if element_page == page_num + 1 or element_page == page_num:
                        page_elements.append(element)
                except:
                    # If we can't get page number, include for page 0
                    if page_num == 0:
                        page_elements.append(element)
            elif page_num == 0:  # If no page metadata, assume it's from page 0
                page_elements.append(element)
        
        # If we didn't get page-specific elements, fall back to analyzing all
        if not page_elements and page_num == 0:
            page_elements = elements
        
        # Analyze the elements for the specific page
        char_count = 0
        text_blocks = 0
        image_count = 0
        all_text = ""
        
        for element in page_elements:
            if hasattr(element, 'text') and element.text:
                char_count += len(element.text)
                text_blocks += 1
                all_text += element.text + " "
            
            # Count image elements
            if hasattr(element, 'category') and element.category == "Image":
                image_count += 1
        
        # Calculate text density (characters per text block)
        density = char_count / max(text_blocks, 1)
        
        # Detect language
        language = "unknown"
        if all_text.strip():
            try:
                language = detect(all_text.strip())
            except LangDetectException:
                language = "unknown"
        
        # Generate heuristic reasoning
        heuristic_reason = _generate_heuristic_reason(
            char_count, text_blocks, image_count, density
        )
        
        return PageSignals(
            char_count=char_count,
            block_count=text_blocks,
            image_count=image_count,
            density=density,
            language=language,
            heuristic_reason=heuristic_reason
        )
    
    except Exception as e:
        # Return default signals if analysis fails
        return PageSignals(
            char_count=0,
            block_count=0,
            image_count=0,
            density=0.0,
            language="unknown",
            heuristic_reason=f"analysis failed: {str(e)}"
        )


def _generate_heuristic_reason(char_count: int, block_count: int, image_count: int, density: float) -> str:
    """Generate heuristic reasoning based on page signals."""
    reasons = []
    
    # Character count analysis
    if char_count < 100:
        reasons.append("very low text content")
    elif char_count < 500:
        reasons.append("low text content")
    elif char_count > 2000:
        reasons.append("high text content")
    
    # Block count analysis
    if block_count < 3:
        reasons.append("few text blocks")
    elif block_count > 20:
        reasons.append("many text blocks")
    
    # Image analysis
    if image_count > 0:
        reasons.append(f"{image_count} images detected")
    
    # Density analysis
    if density < 20:
        reasons.append("low text density")
    elif density > 100:
        reasons.append("high text density")
    
    # Default reasoning
    if not reasons:
        reasons.append("standard document structure")
    
    return ", ".join(reasons)


def get_pdf_page_count(pdf_path: str) -> int:
    """Get the total number of pages in a PDF using Unstructured.io."""
    try:
        # Use Unstructured.io to get page count
        elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",
            include_page_breaks=True,
            extract_images_in_pdf=False,
            infer_table_structure=False,
            chunking_strategy=None
        )
        
        # Find the maximum page number
        max_page = 0
        for element in elements:
            if hasattr(element, 'metadata') and element.metadata:
                try:
                    page_num = element.metadata.get('page_number', 1) if hasattr(element.metadata, 'get') else getattr(element.metadata, 'page_number', 1)
                    max_page = max(max_page, page_num)
                except:
                    # If we can't get page number, assume 1 page
                    max_page = max(max_page, 1)
        
        # Return the page count (convert from 1-indexed to count)
        return max_page if max_page > 0 else 1
    
    except Exception as e:
        # If we can't determine page count, assume 1 page
        print(f"Warning: Could not determine page count for {pdf_path}: {e}")
        return 1


def assess_all_pages(pdf_path: str) -> List[PageSignals]:
    """Assess all pages in a PDF file."""
    page_count = get_pdf_page_count(pdf_path)
    signals = []
    
    for page_num in range(page_count):
        try:
            page_signals = analyze_page_with_unstructured(pdf_path, page_num)
            signals.append(page_signals)
        except Exception as e:
            # If analysis fails, create default signals
            signals.append(PageSignals(
                char_count=0,
                block_count=0,
                image_count=0,
                density=0.0,
                language="unknown",
                heuristic_reason=f"analysis failed: {str(e)}"
            ))
    
    return signals
