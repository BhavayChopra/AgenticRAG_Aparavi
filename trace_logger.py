"""
Trace logging system for the Agentic OCR Framework.
Logs all assessments, decisions, and extraction results to JSONL format.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import jsonlines


class TraceLogger:
    """Logger for tracing all OCR framework operations."""
    
    def __init__(self, log_file_path: str = "trace.jsonl"):
        """
        Initialize the trace logger.
        
        Args:
            log_file_path: Path to the JSONL log file
        """
        self.log_file_path = log_file_path
        self._ensure_log_file_exists()
    
    def _ensure_log_file_exists(self):
        """Ensure the log file exists and is accessible."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        except (OSError, TypeError):
            # Handle case where log_file_path has no directory component
            pass
    
    def log_page_processing(self, processing_result: Dict[str, Any]) -> None:
        """
        Log the complete page processing result.
        
        Args:
            processing_result: Complete result from process_page method
        """
        # Add timestamp and processing metadata
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "page_processing",
            **processing_result
        }
        
        self._write_log_entry(log_entry)
    
    def log_assessment(self, pdf_path: str, page_num: int, signals: Dict[str, Any]) -> None:
        """
        Log page assessment results.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number assessed
            signals: Page signals from assessment
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "page_assessment",
            "pdf_path": pdf_path,
            "page_num": page_num,
            "signals": signals
        }
        
        self._write_log_entry(log_entry)
    
    def log_decision(self, pdf_path: str, page_num: int, decision: str, reason: str, signals: Dict[str, Any]) -> None:
        """
        Log parsing decision results.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number
            decision: Parsing decision (standard_parse or ocr_parse)
            reason: Reason for the decision
            signals: Page signals that led to decision
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "parsing_decision",
            "pdf_path": pdf_path,
            "page_num": page_num,
            "decision": decision,
            "reason": reason,
            "signals": signals
        }
        
        self._write_log_entry(log_entry)
    
    def log_extraction(self, pdf_path: str, page_num: int, method: str, success: bool, 
                      text_length: int, error: Optional[str] = None, metadata: Optional[Dict] = None) -> None:
        """
        Log extraction results.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number
            method: Extraction method used (standard_parse or ocr_parse)
            success: Whether extraction was successful
            text_length: Length of extracted text
            error: Error message if extraction failed
            metadata: Additional metadata from extraction
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "text_extraction",
            "pdf_path": pdf_path,
            "page_num": page_num,
            "method": method,
            "success": success,
            "text_length": text_length,
            "error": error,
            "metadata": metadata or {}
        }
        
        self._write_log_entry(log_entry)
    
    def log_error(self, pdf_path: str, page_num: int, error_type: str, error_message: str, 
                  context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log error events.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number
            error_type: Type of error
            error_message: Error message
            context: Additional context about the error
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "error",
            "pdf_path": pdf_path,
            "page_num": page_num,
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }
        
        self._write_log_entry(log_entry)
    
    def log_session_start(self, session_info: Dict[str, Any]) -> None:
        """
        Log the start of a processing session.
        
        Args:
            session_info: Information about the processing session
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "session_start",
            **session_info
        }
        
        self._write_log_entry(log_entry)
    
    def log_session_end(self, session_info: Dict[str, Any]) -> None:
        """
        Log the end of a processing session.
        
        Args:
            session_info: Summary information about the completed session
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "session_end",
            **session_info
        }
        
        self._write_log_entry(log_entry)
    
    def _write_log_entry(self, log_entry: Dict[str, Any]) -> None:
        """
        Write a log entry to the JSONL file.
        
        Args:
            log_entry: Log entry to write
        """
        try:
            with jsonlines.open(self.log_file_path, mode='a') as writer:
                writer.write(log_entry)
        except Exception as e:
            # Fallback to standard JSON writing
            try:
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            except Exception as fallback_error:
                print(f"Failed to write log entry: {fallback_error}")
    
    def get_session_stats(self, session_start_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a processing session.
        
        Args:
            session_start_time: ISO timestamp of session start (optional)
            
        Returns:
            Dictionary with session statistics
        """
        stats = {
            "total_pages": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "standard_parse_count": 0,
            "ocr_parse_count": 0,
            "errors": 0,
            "total_text_length": 0
        }
        
        try:
            with jsonlines.open(self.log_file_path, mode='r') as reader:
                for entry in reader:
                    # Filter by session if start time provided
                    if session_start_time and entry.get("timestamp", "") < session_start_time:
                        continue
                    
                    event_type = entry.get("event_type", "")
                    
                    if event_type == "page_processing":
                        stats["total_pages"] += 1
                        
                        if entry.get("parsed_content", {}).get("success", False):
                            stats["successful_extractions"] += 1
                            stats["total_text_length"] += entry.get("parsed_content", {}).get("metadata", {}).get("text_length", 0)
                        else:
                            stats["failed_extractions"] += 1
                        
                        decision = entry.get("decision", "")
                        if decision == "standard_parse":
                            stats["standard_parse_count"] += 1
                        elif decision == "ocr_parse":
                            stats["ocr_parse_count"] += 1
                    
                    elif event_type == "error":
                        stats["errors"] += 1
        
        except Exception as e:
            stats["error"] = f"Failed to read log file: {str(e)}"
        
        return stats
    
    def clear_log(self) -> None:
        """Clear the log file."""
        try:
            with open(self.log_file_path, 'w') as f:
                pass  # Just truncate the file
        except Exception as e:
            print(f"Failed to clear log file: {e}")


def create_logger(log_file_path: str = "trace.jsonl") -> TraceLogger:
    """
    Factory function to create a trace logger.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        TraceLogger instance
    """
    return TraceLogger(log_file_path)
