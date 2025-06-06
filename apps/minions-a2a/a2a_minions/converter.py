import tempfile
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

import PyPDF2

logger = logging.getLogger(__name__)


class DocumentConverter:
    """Handles document conversion for A2A tasks."""
    
    def __init__(self):
        self.temp_files = []
        # Create thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pdf_worker")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup_temp_files()
        self.executor.shutdown(wait=False)
    
    async def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file asynchronously."""
        
        def _extract_pdf_sync(path: str) -> str:
            """Synchronous PDF extraction to run in thread pool."""
            try:
                text = ""
                with open(path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    num_pages = len(pdf_reader.pages)
                    
                    for page_num in range(num_pages):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n"
                
                return text.strip()
            except Exception as e:
                logger.error(f"Failed to extract PDF text from {path}: {e}")
                return f"[Error extracting PDF: {e}]"
        
        # Run PDF extraction in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _extract_pdf_sync, pdf_path)