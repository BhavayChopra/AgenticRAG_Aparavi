#!/usr/bin/env python3
"""
Gradio Chat Interface for Agentic RAG Pipeline
Beautiful chat interface with formatted financial responses
"""

import gradio as gr
import json
import logging
from typing import List, Dict, Any
from agentic_rag_pipeline import AgenticRAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatInterface:
    """Professional chat interface for financial RAG queries"""
    
    def __init__(self):
        self.pipeline = None
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Initialize the RAG pipeline"""
        try:
            lightrag_url = "https://lightrag-h7h2g9atabdjg3af.canadacentral-01.azurewebsites.net"
            self.pipeline = AgenticRAGPipeline(lightrag_api_url=lightrag_url)
            logger.info("✅ RAG Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG pipeline: {e}")
            self.pipeline = None
    
    def format_response(self, raw_response: str) -> str:
        """Pure content display - no modifications to pipeline output"""
        try:
            # Return exactly what comes from the pipeline
            # Only clean up source citations for better chat display
            formatted = raw_response
            
            if "SOURCE(S):" in formatted:
                parts = formatted.split("SOURCE(S):")
                if len(parts) == 2:
                    main_content = parts[0].strip()
                    source_part = parts[1].strip()
                    formatted = f"{main_content}\n\n📄 **Sources:** {source_part}"
            
            return formatted
            
        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return raw_response
    
    def chat_with_rag(self, message: str, history) -> str:
        """Main chat function that interfaces with RAG pipeline"""
        
        if not self.pipeline:
            return "❌ **System Error**: RAG pipeline not initialized. Please check your configuration."
        
        if not message.strip():
            return "💡 **Tip**: Ask me specific financial questions about Apple, Microsoft, NVIDIA, Intel, or Amazon!"
        
        try:
            # Get response from RAG pipeline
            rag_response = self.pipeline.query(message)
            
            if not rag_response or 'answer' not in rag_response:
                return "⚠️ **No Results**: I couldn't find relevant information. Try rephrasing your question."
            
            # Format the response
            raw_answer = rag_response['answer']
            formatted_answer = self.format_response(raw_answer)
            
            # Add minimal metadata if available (no hybrid mode display)
            metadata = []
            if 'sources' in rag_response and len(rag_response['sources']) > 0:
                metadata.append(f"📚 {len(rag_response['sources'])} documents")
            
            # Final formatted response - just the pure content with minimal metadata
            final_response = formatted_answer
            if metadata:
                final_response += f"\n\n*{' • '.join(metadata)}*"
            
            return final_response
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"❌ **Error**: {str(e)}\n\nPlease try again or rephrase your question."

def create_chat_interface():
    """Create and configure the Gradio chat interface"""
    
    # Initialize the RAG chat interface
    rag_chat = RAGChatInterface()
    
    # Define example queries
    examples = [
        "How has Apple's total net sales changed over time?",
        "What are the major factors contributing to Apple's gross margin changes?",
        "Has there been any significant change in Apple's operating expenses?", 
        "How has Microsoft's revenue from cloud services evolved?",
        "What are NVIDIA's revenue trends from GPU sales?",
        "How has Intel's R&D spending changed over quarters?",
        "What is Amazon's revenue from AWS services?",
        "Show me trends in Apple's iPhone sales across quarters"
    ]
    
    # Create the main chat interface using gr.ChatInterface
    chat_interface = gr.ChatInterface(
        fn=rag_chat.chat_with_rag,
        type="messages",
        title="💬 AI Financial Assistant",
        description="""
        Ask questions about financial data from major tech companies:
        Apple • Microsoft • NVIDIA • Intel • Amazon
        """,
        examples=examples,
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.emerald,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            text_size=gr.themes.sizes.text_md,
            spacing_size=gr.themes.sizes.spacing_md,
            radius_size=gr.themes.sizes.radius_lg
        ).set(
            body_background_fill="*neutral_50",
            body_background_fill_dark="*neutral_900",
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_hover="*primary_600"
        ),
        chatbot=gr.Chatbot(
            height=650,
            bubble_full_width=False,
            show_copy_button=True,
            avatar_images=(
                None,  # User avatar (will use default)
                "💎"   # Bot avatar - clean diamond icon
            ),
            render_markdown=True,
            show_label=False,
            type="messages",
            layout="bubble"
        ),
        textbox=gr.Textbox(
            placeholder="Ask about financial metrics, trends, or performance...",
            container=False,
            scale=7,
            max_lines=3
        ),
        submit_btn="Send",
        retry_btn="↻ Retry",
        undo_btn="← Undo",
        clear_btn="Clear"
    )
    
    return chat_interface

def main():
    """Launch the Gradio chat interface"""
    
    # Create the interface
    demo = create_chat_interface()
    
    # Add modern, clean CSS styling
    demo.css = """
    /* Main container styling */
    .gradio-container {
        max-width: 1100px !important;
        margin: 0 auto !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Chat message styling */
    .message {
        line-height: 1.6 !important;
        padding: 12px 16px !important;
        margin: 8px 0 !important;
        border-radius: 16px !important;
        max-width: 80% !important;
    }
    
    /* User message styling */
    .user .message {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        margin-left: auto !important;
        margin-right: 0 !important;
    }
    
    /* Bot message styling */
    .bot .message {
        background: #f8fafc !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
        margin-left: 0 !important;
        margin-right: auto !important;
    }
    
    /* Dark mode bot messages */
    .dark .bot .message {
        background: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #374151 !important;
    }
    
    /* Avatar styling */
    .avatar {
        width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 16px !important;
    }
    
    /* Input area styling */
    .input-container {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        transition: border-color 0.2s ease !important;
    }
    
    .input-container:focus-within {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
    }
    
    /* Button styling */
    .primary-button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .primary-button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Code blocks in messages */
    pre, code {
        background: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace !important;
    }
    
    .dark pre, .dark code {
        background: #1e293b !important;
        border: 1px solid #374151 !important;
        color: #f1f5f9 !important;
    }
    
    /* Hide unnecessary elements */
    .chatbot .wrap.svelte-1k8lxgj {
        border: none !important;
        box-shadow: none !important;
    }
    """
    
    # Launch with configuration
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Standard Gradio port
        share=False,            # Set to True if you want a public link
        debug=True,             # Enable debug mode
        show_error=True,        # Show detailed errors
        favicon_path=None,      # You can add a custom favicon
        auth=None              # Add authentication if needed
    )

if __name__ == "__main__":
    main()
