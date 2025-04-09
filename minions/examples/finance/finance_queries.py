import sys
import os
import time
import logging
import pdf2image  # For converting PDF to images
import pytesseract  # For OCR
from PIL import Image
import tempfile
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from minions.utils.minion_mcp import _make_mcp_minion

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==== HELPER FUNCTIONS =====
def validate_pdf_path(pdf_path):
    """Validate that the PDF file exists and is accessible."""
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return False
        if not pdf_path.is_file():
            logger.error(f"Path is not a file: {pdf_path}")
            return False
        if not os.access(pdf_path, os.R_OK):
            logger.error(f"PDF file is not readable: {pdf_path}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating PDF path {pdf_path}: {str(e)}")
        return False

def convert_pdf_to_text(pdf_path):
    """Convert PDF to text, handling both text-based and image-based PDFs."""
    if not validate_pdf_path(pdf_path):
        return None

    # First try standard PDF text extraction
    try:
        import PyPDF2
        logger.info(f"Attempting standard text extraction for {pdf_path}")
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text.strip():
                    text += f"\n--- Page {page_num} ---\n{page_text}"
                else:
                    logger.warning(f"No text extracted from page {page_num}")
            
            # If we got meaningful text, return it
            if len(text.strip()) > 100:  # Arbitrary threshold
                logger.info(f"Successfully extracted text from {pdf_path}")
                return text
            else:
                logger.warning(f"Extracted text too short from {pdf_path}, trying OCR")
    except Exception as e:
        logger.error(f"Standard PDF text extraction failed for {pdf_path}: {str(e)}")
    
    # If standard extraction didn't work, use OCR
    try:
        logger.info(f"Attempting OCR extraction for {pdf_path}")
        # Convert PDF to images
        images = pdf2image.convert_from_path(pdf_path)
        logger.info(f"Converted {pdf_path} to {len(images)} images")
        
        # Create a temporary directory for the images
        with tempfile.TemporaryDirectory() as temp_dir:
            text = ""
            
            # Process each page
            for i, image in enumerate(images, 1):
                logger.info(f"Processing page {i}/{len(images)}...")
                
                # Save the image temporarily
                image_path = os.path.join(temp_dir, f'page_{i}.png')
                image.save(image_path, 'PNG')
                
                # Perform OCR
                page_text = pytesseract.image_to_string(Image.open(image_path))
                if page_text.strip():
                    text += f"\n--- Page {i} ---\n{page_text}"
                else:
                    logger.warning(f"No text extracted from page {i} via OCR")
        
        if text.strip():
            logger.info(f"Successfully extracted text via OCR from {pdf_path}")
            return text
        else:
            logger.error(f"OCR extraction failed to produce text for {pdf_path}")
            return None
    except Exception as e:
        logger.error(f"OCR extraction failed for {pdf_path}: {str(e)}")
        return None

def ensure_text_file(pdf_path):
    """Ensure a text file exists for the given PDF path."""
    try:
        pdf_path = Path(pdf_path)
        text_path = pdf_path.with_suffix('.txt')
        
        # If text file already exists and is recent, return its path
        if text_path.exists():
            # Check if text file is newer than PDF
            if text_path.stat().st_mtime > pdf_path.stat().st_mtime:
                logger.info(f"Using existing text file: {text_path}")
                return str(text_path)
            else:
                logger.info(f"Text file exists but is older than PDF, regenerating")
        
        # Convert PDF to text
        text = convert_pdf_to_text(pdf_path)
        if not text:
            logger.error(f"Failed to convert {pdf_path} to text")
            return None
        
        # Save the text
        try:
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Converted PDF to text and saved to: {text_path}")
            return str(text_path)
        except Exception as e:
            logger.error(f"Error saving text file {text_path}: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error in ensure_text_file for {pdf_path}: {str(e)}")
        return None

def process_pdf_files(pdf_paths):
    """Process multiple PDF files and return their text file paths."""
    text_paths = []
    for pdf_path in pdf_paths:
        text_path = ensure_text_file(pdf_path)
        if not text_path:
            logger.error(f"Could not process {pdf_path}")
            return None
        text_paths.append(text_path)
    return text_paths

# ===== QUERIES =====
def analyze_amd_10k():
    # Create a filesystem minion
    minion = _make_mcp_minion("filesystem")
    
    # Set up the task and context
    pdf_path = os.path.join(project_root, "minions/examples/finance/pdfs/AMD_2015_10K.pdf")
    
    # Convert PDF to text if needed
    text_path = ensure_text_file(pdf_path)
    if not text_path:
        return
    
    context = f"""I have AMD's 2015 10-K filing at {pdf_path}. This is an annual report filed with the SEC.
    The text has been extracted and saved to {text_path}."""
    
    # Define the analysis task
    task = f"""
    Please analyze the AMD 2015 10-K filing and provide:
    1. Key financial metrics (revenue, net income, etc.)
    2. Major business risks mentioned
    3. Significant business developments or changes
    4. Market position and competitive landscape
    
    The file is located at: {text_path}
    """
    
    # Run the minion
    output = minion(
        task=task,
        context=[context],
        max_rounds=10,
        logging_id=int(time.time()),
    )
    
    print("\nAnalysis Results:")
    print("=" * 50)
    print(output['final_answer'])

def compare_amd_years():
    """Compare AMD's financial performance across multiple years using read_multiple_files."""
    # Create a filesystem minion
    minion = _make_mcp_minion("filesystem")
    
    # Set up the task and context
    pdf_paths = [
        os.path.join(project_root, "minions/examples/finance/pdfs/AMD_2015_10K.pdf"),
        os.path.join(project_root, "minions/examples/finance/pdfs/AMD_2016_10K.pdf"),
        os.path.join(project_root, "minions/examples/finance/pdfs/AMD_2019_10K.pdf")
    ]
    
    # Convert PDFs to text and verify they exist
    text_paths = []
    for pdf_path in pdf_paths:
        text_path = ensure_text_file(pdf_path)
        if not text_path:
            print(f"Error: Could not convert {pdf_path} to text")
            return
        text_paths.append(text_path)
    
    context = f"""I have AMD's 10-K filings from 2015, 2016, and 2019 at {text_paths}. 
    These are annual reports filed with the SEC."""
    
    # Define the analysis task
    task = f"""
    Please analyze AMD's financial performance across these years and provide:
    1. Revenue growth trends
    2. Changes in net income/loss
    3. Key business developments in each year
    4. Market position changes over time
    
    The files are located at: {text_paths}
    """
    
    # Run the minion
    output = minion(
        task=task,
        context=[context],
        max_rounds=10,
        logging_id=int(time.time()),
    )
    
    print("\nAnalysis Results:")
    print("=" * 50)
    print(output['final_answer'])

def analyze_tech_sector():
    """Analyze multiple tech companies using search_files and read_multiple_files."""
    # Create a filesystem minion
    minion = _make_mcp_minion("filesystem")
    
    # Set up the task and context
    pdf_paths = [
        os.path.join(project_root, "minions/examples/finance/pdfs/AMD_2015_10K.pdf"),
        os.path.join(project_root, "minions/examples/finance/pdfs/MICROSOFT_2017_10K.pdf"),
        os.path.join(project_root, "minions/examples/finance/pdfs/APPLE_2022_10K.pdf")
    ]
    
    # Convert PDFs to text and verify they exist
    text_paths = []
    for pdf_path in pdf_paths:
        text_path = ensure_text_file(pdf_path)
        if not text_path:
            print(f"Error: Could not convert {pdf_path} to text")
            return
        text_paths.append(text_path)
    
    context = f"""I have tech company 10-K filings at {text_paths}. 
    These are annual reports filed with the SEC."""
    
    # Define the analysis task
    task = f"""
    Please analyze the tech sector by comparing:
    1. AMD's 2015 10-K
    2. Microsoft's 2017 10-K
    3. Apple's 2022 10-K
    
    For each company, provide:
    1. Key financial metrics
    2. Market position
    3. Competitive advantages
    4. Growth strategies
    
    The files are located at: {text_paths}
    """
    
    # Run the minion
    output = minion(
        task=task,
        context=[context],
        max_rounds=10,
        logging_id=int(time.time()),
    )
    
    print("\nAnalysis Results:")
    print("=" * 50)
    print(output['final_answer'])

def analyze_retail_sector():
    """Analyze retail companies using list_directory and read_multiple_files."""
    # Create a filesystem minion
    minion = _make_mcp_minion("filesystem")
    
    # Set up the task and context
    pdf_paths = [
        os.path.join(project_root, "minions/examples/finance/pdfs/WALMART_2017_10K.pdf"),
        os.path.join(project_root, "minions/examples/finance/pdfs/BESTBUY_2022_10K.pdf"),
        os.path.join(project_root, "minions/examples/finance/pdfs/ULTABEAUTY_2023_10K.pdf")
    ]
    
    # Convert PDFs to text and verify they exist
    text_paths = []
    for pdf_path in pdf_paths:
        text_path = ensure_text_file(pdf_path)
        if not text_path:
            print(f"Error: Could not convert {pdf_path} to text")
            return
        text_paths.append(text_path)
    
    context = f"""I have retail company 10-K filings at {text_paths}. 
    These are annual reports filed with the SEC."""
    
    # Define the analysis task
    task = f"""
    Please analyze the retail sector by comparing:
    1. Walmart's 2017 10-K
    2. Best Buy's 2022 10-K
    3. Ulta Beauty's 2023 10-K
    
    For each company, provide:
    1. Revenue breakdown by segment
    2. Store count and expansion plans
    3. E-commerce strategy
    4. Competitive positioning
    
    The files are located at: {text_paths}
    """
    
    # Run the minion
    output = minion(
        task=task,
        context=[context],
        max_rounds=10,
        logging_id=int(time.time()),
    )
    
    print("\nAnalysis Results:")
    print("=" * 50)
    print(output['final_answer'])

def analyze_earnings_trends():
    """Analyze earnings trends across multiple companies using search_files and read_multiple_files."""
    # Create a filesystem minion
    minion = _make_mcp_minion("filesystem")
    
    # Set up the task and context
    pdf_paths = [
        os.path.join("minions/examples/finance/pdfs/PEPSICO_2023Q2_EARNINGS.pdf"),
        os.path.join("minions/examples/finance/pdfs/SALESFORCE_2024Q2_EARNINGS.pdf"),
        os.path.join("minions/examples/finance/pdfs/ULTABEAUTY_2024Q2_EARNINGS.pdf")
    ]
    
    # Convert PDFs to text and verify they exist
    text_paths = []
    for pdf_path in pdf_paths:
        text_path = ensure_text_file(pdf_path)
        if not text_path:
            print(f"Error: Could not convert {pdf_path} to text")
            return
        text_paths.append(text_path)
    
    context = f"""I have quarterly earnings reports at {text_paths}. 
    These are quarterly reports filed with the SEC."""
    
    # Define the analysis task
    task = f"""
    Please analyze earnings trends by comparing:
    1. PepsiCo's 2023 Q2 earnings
    2. Salesforce's 2024 Q2 earnings
    3. Ulta Beauty's 2024 Q2 earnings
    
    For each company, provide:
    1. Revenue growth vs previous quarter
    2. Key drivers of performance
    3. Forward guidance
    4. Market reaction
    
    The files are located at: {text_paths}
    """
    
    # Run the minion
    output = minion(
        task=task,
        context=[context],
        max_rounds=10,
        logging_id=int(time.time()),
    )
    
    print("\nAnalysis Results:")
    print("=" * 50)
    print(output['final_answer'])

# ==== MAIN =====
if __name__ == "__main__":
    # Uncomment the function you want to run
    # analyze_amd_10k()
    # compare_amd_years()
    # analyze_tech_sector()
    # analyze_retail_sector()
    analyze_earnings_trends() 