"""
Script to execute parsing based on decisions in parsed_docs folder.
Outputs in structured format for ETL, embedding pipelines, and LightRAG.
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from parse import standard_parse_page, ocr_parse_page

def sanitize_filename(filename):
    """Convert filename to safe directory name."""
    # Remove extension and path
    name = os.path.splitext(os.path.basename(filename))[0]
    # Replace spaces and special characters with underscores
    name = re.sub(r'[^\w\-_]', '_', name)
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    return name.strip('_')

def infer_doc_type(filename):
    """Infer document type from filename."""
    filename_lower = filename.lower()
    if 'invoice' in filename_lower:
        return 'invoice'
    elif any(q in filename_lower for q in ['q1', 'q2', 'q3', 'q4', 'quarter']):
        return 'financial_report'
    elif 'energieausweis' in filename_lower:
        return 'energy_certificate'
    elif 'receipt' in filename_lower:
        return 'receipt'
    elif 'license' in filename_lower:
        return 'license'
    else:
        return 'document'

def process_decisions():
    """
    Process all decision files and execute parsing accordingly.
    """
    decisions_dir = "parsed_docs"
    
    # Group decisions by file and method
    file_decisions = defaultdict(lambda: {"standard_parse": [], "ocr_parse": []})
    file_metadata = {}
    
    # Read all decision files
    print("📖 Reading decision files...")
    for filename in os.listdir(decisions_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(decisions_dir, filename)
            
            with open(filepath, 'r') as f:
                decision_data = json.load(f)
            
            source_pdf = decision_data["source_pdf"]
            page_number = decision_data["page_number"] + 1  # Convert to 1-indexed
            decision = decision_data["decision"]
            
            # Store metadata for this file
            if source_pdf not in file_metadata:
                file_metadata[source_pdf] = {
                    "source_pdf": source_pdf,
                    "total_pages": 0,
                    "standard_pages": [],
                    "ocr_pages": [],
                    "processing_timestamp": datetime.now().isoformat()
                }
            
            file_decisions[source_pdf][decision].append(page_number)
            file_metadata[source_pdf]["total_pages"] = max(
                file_metadata[source_pdf]["total_pages"], 
                page_number
            )
            
            if decision == "standard_parse":
                file_metadata[source_pdf]["standard_pages"].append(page_number)
            else:
                file_metadata[source_pdf]["ocr_pages"].append(page_number)
    
    # Create output directories
    os.makedirs("parsed_results/by_document", exist_ok=True)
    os.makedirs("parsed_results/flat_pages", exist_ok=True)
    os.makedirs("parsed_results/lightrag_input", exist_ok=True)
    
    # Execute parsing for each file
    for source_pdf, decisions in file_decisions.items():
        doc_name = sanitize_filename(source_pdf)
        doc_type = infer_doc_type(source_pdf)
        print(f"\n📄 Processing: {source_pdf}")
        print(f"   Directory: {doc_name}")
        print(f"   Document type: {doc_type}")
        
        # Create document directory
        doc_dir = f"parsed_results/by_document/{doc_name}"
        os.makedirs(doc_dir, exist_ok=True)
        
        # Prepare LightRAG output file
        lightrag_file = f"parsed_results/lightrag_input/{doc_name}_chunks.jsonl"
        lightrag_chunks = []
        
        # Parse standard pages
        standard_result = None
        if decisions["standard_parse"]:
            pages = sorted(decisions["standard_parse"])
            print(f"   📝 Standard parsing pages: {pages}")
            
            try:
                standard_result = standard_parse_page(source_pdf, pages)
                
                # Save to by_document structure
                with open(f"{doc_dir}/standard_pages.json", 'w') as f:
                    json.dump(standard_result, f, indent=2)
                
                # Process for flat_pages and LightRAG
                for page_data in standard_result["parsed_pages"]:
                    page_num = page_data["page_number"]
                    
                    # Flat page file
                    page_filename = f"parsed_results/flat_pages/{doc_name}_page_{page_num:03d}.json"
                    page_output = {
                        "source_pdf": source_pdf,
                        "page_number": page_num,
                        "parsing_method": "standard_parse",
                        "timestamp": datetime.now().isoformat(),
                        **page_data
                    }
                    
                    with open(page_filename, 'w') as f:
                        json.dump(page_output, f, indent=2)
                    
                    # LightRAG chunk
                    lightrag_chunk = {
                        "doc_id": doc_name,
                        "chunk_id": f"{doc_name}_page_{page_num:03d}",
                        "content": page_data["raw_text"],
                        "metadata": {
                            "source_pdf": source_pdf,
                            "page_number": page_num,
                            "parsing_method": "standard_parse",
                            "element_count": len(page_data.get("elements", [])),
                            "doc_type": doc_type,
                            "text_length": len(page_data["raw_text"]),
                            "elements": page_data.get("elements", [])
                        }
                    }
                    lightrag_chunks.append(lightrag_chunk)
                
                print(f"      ✅ Standard pages saved")
                
            except Exception as e:
                print(f"      ❌ Standard parsing error: {e}")
        
        # Parse OCR pages
        ocr_result = None
        if decisions["ocr_parse"]:
            pages = sorted(decisions["ocr_parse"])
            print(f"   🔍 OCR parsing pages: {pages}")
            
            try:
                ocr_result = ocr_parse_page(source_pdf, pages)
                
                # Save to by_document structure
                with open(f"{doc_dir}/ocr_pages.json", 'w') as f:
                    json.dump(ocr_result, f, indent=2)
                
                # Process for flat_pages and LightRAG
                for page_data in ocr_result["parsed_pages"]:
                    page_num = page_data["page_number"]
                    
                    # Flat page file
                    page_filename = f"parsed_results/flat_pages/{doc_name}_page_{page_num:03d}.json"
                    page_output = {
                        "source_pdf": source_pdf,
                        "page_number": page_num,
                        "parsing_method": "ocr_parse",
                        "timestamp": datetime.now().isoformat(),
                        **page_data
                    }
                    
                    with open(page_filename, 'w') as f:
                        json.dump(page_output, f, indent=2)
                    
                    # LightRAG chunk
                    lightrag_chunk = {
                        "doc_id": doc_name,
                        "chunk_id": f"{doc_name}_page_{page_num:03d}",
                        "content": page_data["raw_text"],
                        "metadata": {
                            "source_pdf": source_pdf,
                            "page_number": page_num,
                            "parsing_method": "ocr_parse",
                            "element_count": len(page_data.get("elements", [])),
                            "doc_type": doc_type,
                            "text_length": len(page_data["raw_text"]),
                            "elements": page_data.get("elements", [])
                        }
                    }
                    lightrag_chunks.append(lightrag_chunk)
                
                print(f"      ✅ OCR pages saved")
                
            except Exception as e:
                print(f"      ❌ OCR parsing error: {e}")
        
        # Save LightRAG chunks (JSONL format)
        if lightrag_chunks:
            with open(lightrag_file, 'w') as f:
                for chunk in lightrag_chunks:
                    f.write(json.dumps(chunk) + '\n')
            print(f"      🚀 LightRAG chunks saved: {len(lightrag_chunks)} chunks")
        
        # Create metadata file
        metadata = file_metadata[source_pdf].copy()
        metadata.update({
            "document_name": doc_name,
            "document_type": doc_type,
            "standard_pages_count": len(decisions["standard_parse"]),
            "ocr_pages_count": len(decisions["ocr_parse"]),
            "standard_success": standard_result is not None and standard_result.get("filename") is not None,
            "ocr_success": ocr_result is not None and ocr_result.get("filename") is not None,
            "lightrag_chunks": len(lightrag_chunks)
        })
        
        with open(f"{doc_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"      📊 Metadata saved")
    
    # Create summary
    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "total_documents": len(file_decisions),
        "total_pages": sum(len(decisions["standard_parse"]) + len(decisions["ocr_parse"]) 
                          for decisions in file_decisions.values()),
        "output_formats": {
            "by_document": "Document-grouped JSON files",
            "flat_pages": "Individual page JSON files", 
            "lightrag_input": "JSONL chunks ready for LightRAG"
        },
        "documents": {sanitize_filename(pdf): {
            "standard_pages": len(decisions["standard_parse"]),
            "ocr_pages": len(decisions["ocr_parse"]),
            "doc_type": infer_doc_type(pdf)
        } for pdf, decisions in file_decisions.items()}
    }
    
    with open("parsed_results/processing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary:")
    print(f"   Total documents: {summary['total_documents']}")
    print(f"   Total pages: {summary['total_pages']}")
    print(f"   Output formats: 3 (by_document, flat_pages, lightrag_input)")
    print(f"   Structure created: parsed_results/")

if __name__ == "__main__":
    print("🚀 Running structured parsing with LightRAG optimization...")
    process_decisions()
    print("\n✅ Done! Check parsed_results/ for organized output including LightRAG-ready chunks.")
