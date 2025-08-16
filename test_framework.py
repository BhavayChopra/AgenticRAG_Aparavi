"""
Test script for the Agentic OCR Framework.
Tests the framework with a single PDF to verify functionality.
"""

import os
import sys
from pathlib import Path

from config import load_config, validate_config
from agentic_ocr import AgenticOCRFramework
from trace_logger import TraceLogger
from page_assessment import analyze_page_with_unstructured, get_pdf_page_count


def test_configuration():
    """Test that configuration is properly loaded."""
    print("Testing configuration...")
    
    try:
        config = load_config()
        print(f"✓ Configuration loaded successfully")
        print(f"  - OpenAI API Key: {'*' * 10 if config.openai_api_key else 'NOT SET'}")
        print(f"  - LlamaParse API Key: {'*' * 10 if config.llama_parse.api_key else 'NOT SET'}")
        
        # Note: We don't validate here to allow testing with minimal config
        return True
    
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_page_assessment(pdf_path: str):
    """Test page assessment functionality."""
    print(f"\nTesting page assessment with: {Path(pdf_path).name}")
    
    try:
        # Get page count
        page_count = get_pdf_page_count(pdf_path)
        print(f"✓ PDF has {page_count} pages")
        
        # Test assessment on first page
        if page_count > 0:
            signals = analyze_page_with_unstructured(pdf_path, 0)
            print(f"✓ Page 0 assessment successful:")
            print(f"  - Characters: {signals.char_count}")
            print(f"  - Text blocks: {signals.block_count}")
            print(f"  - Images: {signals.image_count}")
            print(f"  - Density: {signals.density:.2f}")
            print(f"  - Language: {signals.language}")
            print(f"  - Heuristic: {signals.heuristic_reason}")
            return True
        else:
            print("✗ PDF has no pages")
            return False
    
    except Exception as e:
        print(f"✗ Page assessment failed: {e}")
        return False


def test_framework_initialization():
    """Test framework initialization."""
    print("\nTesting framework initialization...")
    
    try:
        framework = AgenticOCRFramework()
        print("✓ Framework initialized successfully")
        print(f"  - Assistant agent: {framework.assistant.name}")
        print(f"  - User proxy: {framework.user_proxy.name}")
        return framework
    
    except Exception as e:
        print(f"✗ Framework initialization failed: {e}")
        return None


def test_single_page_processing(framework: AgenticOCRFramework, pdf_path: str):
    """Test processing a single page."""
    print(f"\nTesting single page processing...")
    
    try:
        # Process page 0
        result = framework.process_page(pdf_path, 0)
        
        print(f"✓ Page processing completed:")
        print(f"  - Decision: {result.get('decision', 'unknown')}")
        print(f"  - Reason: {result.get('reason', 'unknown')}")
        print(f"  - Success: {result.get('parsed_content', {}).get('success', False)}")
        
        text_length = len(result.get('parsed_content', {}).get('text', ''))
        print(f"  - Extracted text length: {text_length}")
        
        if result.get('processing_error'):
            print(f"  - Processing error: {result['processing_error']}")
        
        return result
    
    except Exception as e:
        print(f"✗ Single page processing failed: {e}")
        return None


def test_trace_logging():
    """Test trace logging functionality."""
    print("\nTesting trace logging...")
    
    try:
        logger = TraceLogger("test_trace.jsonl")
        
        # Test logging
        logger.log_session_start({"test": "session"})
        logger.log_assessment("test.pdf", 0, {"char_count": 100})
        logger.log_decision("test.pdf", 0, "standard_parse", "test reason", {})
        logger.log_extraction("test.pdf", 0, "standard_parse", True, 100)
        logger.log_session_end({"test": "complete"})
        
        print("✓ Trace logging successful")
        
        # Clean up test file
        if os.path.exists("test_trace.jsonl"):
            os.unlink("test_trace.jsonl")
        
        return True
    
    except Exception as e:
        print(f"✗ Trace logging failed: {e}")
        return False


def find_test_pdf():
    """Find a suitable PDF for testing."""
    sample_dir = Path("SampleDataSet")
    
    if not sample_dir.exists():
        print(f"✗ SampleDataSet directory not found")
        return None
    
    # Look for a small PDF file (invoices are usually smaller)
    pdf_files = list(sample_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"✗ No PDF files found in SampleDataSet")
        return None
    
    # Prefer invoice files for testing (they're usually simpler)
    invoice_files = [f for f in pdf_files if "invoice" in f.name.lower()]
    if invoice_files:
        return str(invoice_files[0])
    
    # Otherwise use any PDF
    return str(pdf_files[0])


def main():
    """Main test function."""
    print("Agentic OCR Framework - Test Suite")
    print("=" * 50)
    
    # Test 1: Configuration
    if not test_configuration():
        print("\n⚠️  Configuration test failed - some features may not work")
    
    # Test 2: Find test PDF
    test_pdf = find_test_pdf()
    if not test_pdf:
        print("\n✗ No test PDF found - cannot continue with tests")
        return
    
    print(f"\nUsing test PDF: {Path(test_pdf).name}")
    
    # Test 3: Page assessment
    if not test_page_assessment(test_pdf):
        print("\n✗ Page assessment test failed")
        return
    
    # Test 4: Framework initialization
    framework = test_framework_initialization()
    if not framework:
        print("\n✗ Framework initialization failed")
        return
    
    # Test 5: Trace logging
    if not test_trace_logging():
        print("\n⚠️  Trace logging test failed")
    
    # Test 6: Single page processing (only if we have proper API keys)
    try:
        config = load_config()
        validate_config(config)
        
        result = test_single_page_processing(framework, test_pdf)
        if result:
            print("\n✓ All tests completed successfully!")
        else:
            print("\n✗ Single page processing test failed")
    
    except Exception as e:
        print(f"\n⚠️  Skipping single page processing test due to configuration: {e}")
        print("✓ Basic functionality tests completed!")
    
    print(f"\nTo run full processing, use:")
    print(f"python runner.py --single-file \"{test_pdf}\" --max-pages 2")


if __name__ == "__main__":
    main()
