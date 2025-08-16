"""
Main runner script for the Agentic OCR Framework.
Processes all PDFs in the SampleDataSet folder using intelligent parsing decisions.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict
import argparse

from agentic_ocr import AgenticOCRFramework
from trace_logger import TraceLogger
from page_assessment import get_pdf_page_count, analyze_page_with_unstructured


class OCRRunner:
    """Main runner for processing PDFs with the Agentic OCR Framework."""
    
    def __init__(self, output_dir: str = "parsed_docs", log_file: str = "trace.jsonl"):
        """
        Initialize the OCR runner.
        
        Args:
            output_dir: Directory to store parsed outputs
            log_file: Path to the trace log file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = TraceLogger(log_file)
        self.framework = AgenticOCRFramework()
        
        self.session_start_time = datetime.utcnow().isoformat()
        
    def save_parsed_content(self, pdf_path: str, page_num: int, processing_result: Dict[str, Any]) -> str:
        """
        Save parsed content to a local file.
        
        Args:
            pdf_path: Original PDF path
            page_num: Page number
            processing_result: Complete processing result
            
        Returns:
            Path to the saved file
        """
        # Create a safe filename
        pdf_name = Path(pdf_path).stem
        output_filename = f"{pdf_name}_page_{page_num:03d}.json"
        output_path = self.output_dir / output_filename
        
        # Prepare content for saving
        save_content = {
            "source_pdf": pdf_path,
            "page_number": page_num,
            "processing_timestamp": datetime.utcnow().isoformat(),
            "signals": processing_result.get("signals", {}),
            "decision": processing_result.get("decision", ""),
            "reason": processing_result.get("reason", ""),
            "parsing_method": processing_result.get("parsed_content", {}).get("metadata", {}).get("parsing_method", ""),
            "text_content": processing_result.get("parsed_content", {}).get("text", ""),
            "extraction_metadata": processing_result.get("parsed_content", {}).get("metadata", {}),
            "success": processing_result.get("parsed_content", {}).get("success", False),
            "error": processing_result.get("parsed_content", {}).get("error", None),
            "processing_error": processing_result.get("processing_error", None)
        }
        
        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_content, f, indent=2, ensure_ascii=False)
            return str(output_path)
        except Exception as e:
            self.logger.log_error(pdf_path, page_num, "file_save_error", str(e))
            return ""
    
    def save_decisions_only(self, pdf_path: str, page_num: int, signals: Dict[str, Any], decision: str, reason: str) -> str:
        """
        Save just the agent decisions (when parsing fails but we want to preserve decisions).
        
        Args:
            pdf_path: Original PDF path
            page_num: Page number
            signals: Page assessment signals
            decision: Agent decision
            reason: Decision reasoning
            
        Returns:
            Path to the saved decisions file
        """
        # Create a safe filename for decisions
        pdf_name = Path(pdf_path).stem
        decisions_filename = f"{pdf_name}_page_{page_num:03d}_DECISIONS_ONLY.json"
        decisions_path = self.output_dir / decisions_filename
        
        # Prepare decisions for saving
        decisions_content = {
            "source_pdf": pdf_path,
            "page_number": page_num,
            "assessment_timestamp": datetime.utcnow().isoformat(),
            "signals": signals,
            "agent_decision": decision,
            "agent_reasoning": reason,
            "status": "decision_only",
            "note": "Parsing failed but agent decision preserved for later execution"
        }
        
        # Save decisions file
        try:
            with open(decisions_path, 'w', encoding='utf-8') as f:
                json.dump(decisions_content, f, indent=2, ensure_ascii=False)
            return str(decisions_path)
        except Exception as e:
            print(f"Warning: Could not save decisions for {pdf_path} page {page_num}: {e}")
            return ""
    
    def process_single_pdf(self, pdf_path: str, max_pages: int = None) -> Dict[str, Any]:
        """
        Process all pages of a single PDF.
        
        Args:
            pdf_path: Path to the PDF file
            max_pages: Maximum number of pages to process (None for all)
            
        Returns:
            Processing summary
        """
        print(f"\nProcessing PDF: {pdf_path}")
        
        try:
            total_pages = get_pdf_page_count(pdf_path)
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
            
            print(f"Total pages: {total_pages}, Processing: {pages_to_process}")
            
            results = []
            successful_pages = 0
            failed_pages = 0
            
            for page_num in range(pages_to_process):
                print(f"  Processing page {page_num + 1}/{pages_to_process}...")
                
                # Initialize variables for this page
                page_signals = None
                agent_decision = "unknown"
                agent_reason = "not processed"
                saved_path = ""
                
                try:
                    # Step 1: Always try to get page assessment and agent decision first
                    try:
                        page_signals = analyze_page_with_unstructured(pdf_path, page_num)
                        print(f"    📊 Assessment: {page_signals.char_count} chars, {page_signals.block_count} blocks, {page_signals.image_count} images")
                        
                        # Get agent decision
                        agent_decision, agent_reason = self.framework.decide_parsing_method(page_signals)
                        print(f"    🤖 Decision: {agent_decision} - {agent_reason}")
                        
                    except Exception as assess_error:
                        print(f"    ⚠️  Assessment failed: {assess_error}")
                        # Continue with default signals
                        page_signals = {"char_count": 0, "block_count": 0, "image_count": 0, "density": 0.0, "language": "unknown", "heuristic_reason": f"assessment failed: {assess_error}"}
                        agent_decision = "standard_parse"  # Default fallback
                        agent_reason = f"assessment failed, defaulting to standard_parse: {assess_error}"
                    
                    # Step 2: Try to execute the parsing
                    try:
                        # Process the page with the framework
                        processing_result = self.framework.process_page(pdf_path, page_num)
                        
                        # Log the complete processing result
                        self.logger.log_page_processing(processing_result)
                        
                        # Save parsed content locally
                        saved_path = self.save_parsed_content(pdf_path, page_num, processing_result)
                        processing_result["saved_to"] = saved_path
                        
                        results.append(processing_result)
                        
                        # Update counters
                        if processing_result.get("parsed_content", {}).get("success", False):
                            successful_pages += 1
                            print(f"    ✅ Complete Success: Parsing worked!")
                        else:
                            failed_pages += 1
                            error_msg = processing_result.get("parsed_content", {}).get("error", "Unknown parsing error")
                            print(f"    ⚠️  Parsing failed but decision saved: {error_msg}")
                    
                    except Exception as parsing_error:
                        # Parsing failed, but save the decisions anyway
                        failed_pages += 1
                        error_msg = str(parsing_error)
                        print(f"    ❌ Parsing failed: {error_msg}")
                        print(f"    💾 Saving decisions for later execution...")
                        
                        # Save just the decisions
                        decisions_path = self.save_decisions_only(pdf_path, page_num, asdict(page_signals) if page_signals else {}, agent_decision, agent_reason)
                        
                        # Log the error
                        self.logger.log_error(pdf_path, page_num, "parsing_exception", error_msg)
                        
                        # Add failed result with preserved decisions
                        results.append({
                            "pdf_path": pdf_path,
                            "page_num": page_num,
                            "signals": asdict(page_signals) if page_signals else None,
                            "decision": agent_decision,
                            "reason": agent_reason,
                            "parsed_content": {"success": False, "error": error_msg},
                            "processing_error": error_msg,
                            "saved_to": decisions_path,
                            "decision_preserved": True
                        })
                
                except Exception as e:
                    # Complete failure - even assessment failed
                    failed_pages += 1
                    error_msg = str(e)
                    print(f"    ❌ Complete failure: {error_msg}")
                    
                    self.logger.log_error(pdf_path, page_num, "complete_failure", error_msg)
                    
                    # Add completely failed result
                    results.append({
                        "pdf_path": pdf_path,
                        "page_num": page_num,
                        "signals": None,
                        "decision": "failed",
                        "reason": f"complete processing failure: {error_msg}",
                        "parsed_content": {"success": False, "error": error_msg},
                        "processing_error": error_msg,
                        "saved_to": "",
                        "decision_preserved": False
                    })
            
            summary = {
                "pdf_path": pdf_path,
                "total_pages": total_pages,
                "processed_pages": pages_to_process,
                "successful_pages": successful_pages,
                "failed_pages": failed_pages,
                "results": results
            }
            
            print(f"  Summary: {successful_pages} successful, {failed_pages} failed")
            return summary
        
        except Exception as e:
            error_msg = f"Failed to process PDF {pdf_path}: {str(e)}"
            print(f"  ✗ {error_msg}")
            self.logger.log_error(pdf_path, 0, "pdf_processing_error", error_msg)
            
            return {
                "pdf_path": pdf_path,
                "total_pages": 0,
                "processed_pages": 0,
                "successful_pages": 0,
                "failed_pages": 0,
                "error": error_msg,
                "results": []
            }
    
    def process_pdf_directory(self, directory_path: str, max_files: int = None, max_pages_per_file: int = None) -> Dict[str, Any]:
        """
        Process all PDFs in a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            max_files: Maximum number of files to process (None for all)
            max_pages_per_file: Maximum pages per file (None for all)
            
        Returns:
            Overall processing summary
        """
        directory_path = Path(directory_path)
        
        # Find all PDF files
        pdf_files = list(directory_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return {"error": "No PDF files found", "processed_files": []}
        
        files_to_process = pdf_files[:max_files] if max_files else pdf_files
        
        print(f"Found {len(pdf_files)} PDF files, processing {len(files_to_process)}")
        
        # Log session start
        session_info = {
            "directory": str(directory_path),
            "total_pdf_files": len(pdf_files),
            "files_to_process": len(files_to_process),
            "max_pages_per_file": max_pages_per_file,
            "output_directory": str(self.output_dir)
        }
        self.logger.log_session_start(session_info)
        
        # Process each PDF
        all_results = []
        total_successful_pages = 0
        total_failed_pages = 0
        
        for i, pdf_file in enumerate(files_to_process, 1):
            print(f"\n[{i}/{len(files_to_process)}] Processing: {pdf_file.name}")
            
            result = self.process_single_pdf(str(pdf_file), max_pages_per_file)
            all_results.append(result)
            
            total_successful_pages += result.get("successful_pages", 0)
            total_failed_pages += result.get("failed_pages", 0)
        
        # Final summary
        session_summary = {
            "session_start_time": self.session_start_time,
            "session_end_time": datetime.utcnow().isoformat(),
            "directory": str(directory_path),
            "total_pdf_files_found": len(pdf_files),
            "processed_files": len(files_to_process),
            "total_successful_pages": total_successful_pages,
            "total_failed_pages": total_failed_pages,
            "output_directory": str(self.output_dir),
            "log_file": self.logger.log_file_path
        }
        
        self.logger.log_session_end(session_summary)
        
        print(f"\n{'='*60}")
        print(f"SESSION COMPLETE")
        print(f"{'='*60}")
        print(f"Files processed: {len(files_to_process)}")
        print(f"Total successful pages: {total_successful_pages}")
        print(f"Total failed pages: {total_failed_pages}")
        print(f"Output directory: {self.output_dir}")
        print(f"Trace log: {self.logger.log_file_path}")
        
        return {
            "session_summary": session_summary,
            "processed_files": all_results
        }


def main():
    """Main entry point for the runner script."""
    parser = argparse.ArgumentParser(description="Agentic OCR Framework Runner")
    parser.add_argument("--input-dir", default="SampleDataSet", 
                       help="Directory containing PDF files to process")
    parser.add_argument("--output-dir", default="parsed_docs",
                       help="Directory to store parsed outputs")
    parser.add_argument("--log-file", default="trace.jsonl",
                       help="Path to trace log file")
    parser.add_argument("--max-files", type=int, default=None,
                       help="Maximum number of PDF files to process")
    parser.add_argument("--max-pages", type=int, default=None,
                       help="Maximum number of pages per PDF to process")
    parser.add_argument("--single-file", type=str, default=None,
                       help="Process only a specific PDF file")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = OCRRunner(output_dir=args.output_dir, log_file=args.log_file)
    
    print("Agentic OCR Framework")
    print("=" * 60)
    
    try:
        if args.single_file:
            # Process single file
            if not os.path.exists(args.single_file):
                print(f"Error: File {args.single_file} not found")
                return
            
            result = runner.process_single_pdf(args.single_file, args.max_pages)
            print(f"\nSingle file processing complete: {result.get('successful_pages', 0)} successful, {result.get('failed_pages', 0)} failed")
        
        else:
            # Process directory
            if not os.path.exists(args.input_dir):
                print(f"Error: Directory {args.input_dir} not found")
                return
            
            result = runner.process_pdf_directory(args.input_dir, args.max_files, args.max_pages)
    
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
