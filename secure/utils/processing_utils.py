import os
import glob
import logging
import fitz  # PyMuPDF
from typing import Optional, Tuple, Any

# Define default logger
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str, custom_logger: Any = None) -> Optional[str]:
    """Extract text content from a PDF file"""
    # Use custom logger if provided, otherwise use module logger
    log = custom_logger if custom_logger else logger

    try:
        if not os.path.exists(pdf_path):
            log.error(f"PDF file not found: {pdf_path}")
            return None

        log.info(f"📄 Extracting text from PDF: {pdf_path}")

        pdf_content = ""
        try:
            # Open the PDF file
            with fitz.open(pdf_path) as doc:
                # Iterate through each page
                for page_num in range(len(doc)):
                    # Get the page
                    page = doc[page_num]
                    # Extract text from the page
                    pdf_content += page.get_text()
                    # Add a separator between pages
                    if page_num < len(doc) - 1:
                        pdf_content += "\n\n"
        except Exception as e:
            log.error(f"Error extracting text from PDF: {str(e)}")
            return None

        log.info(f"✅ PDF text extraction successful: {len(pdf_content)} characters")
        return pdf_content

    except Exception as e:
        log.error(f"❌ Error processing PDF: {str(e)}")
        return None


def extract_text_from_txt(txt_path: str, custom_logger: Any = None) -> Optional[str]:
    """Extract text content from a .txt file"""
    # Use custom logger if provided, otherwise use module logger
    log = custom_logger if custom_logger else logger

    try:
        if not os.path.exists(txt_path):
            log.error(f"Text file not found: {txt_path}")
            return None

        log.info(f"📝 Reading text from file: {txt_path}")

        try:
            with open(txt_path, "r", encoding="utf-8") as file:
                txt_content = file.read()
        except UnicodeDecodeError:
            # Try other encodings if UTF-8 fails
            try:
                with open(txt_path, "r", encoding="latin-1") as file:
                    txt_content = file.read()
            except Exception as e:
                log.error(f"Error reading text file with latin-1 encoding: {str(e)}")
                return None

        log.info(f"✅ Text file read successful: {len(txt_content)} characters")
        return txt_content

    except Exception as e:
        log.error(f"❌ Error processing text file: {str(e)}")
        return None


def process_folder(
    folder_path: str, custom_logger: Any = None
) -> Tuple[str, Optional[str]]:
    """Process a folder containing text, PDF, and image files.

    Returns:
        Tuple containing:
        - Concatenated text content from all text and PDF files
        - Path to a selected image (or None if no images found)
    """
    # Use custom logger if provided, otherwise use module logger
    log = custom_logger if custom_logger else logger

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        log.error(f"Folder not found or not a directory: {folder_path}")
        return "", None

    log.info(f"📂 Processing folder: {folder_path}")

    # Find all text, PDF, and image files in the folder
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    # Process text files
    all_text_content = []
    for txt_file in txt_files:
        txt_content = extract_text_from_txt(txt_file, custom_logger)
        if txt_content:
            all_text_content.append(
                f"--- Content from {os.path.basename(txt_file)} ---\n{txt_content}"
            )

    # Process PDF files
    for pdf_file in pdf_files:
        pdf_content = extract_text_from_pdf(pdf_file, custom_logger)
        if pdf_content:
            all_text_content.append(
                f"--- Content from {os.path.basename(pdf_file)} ---\n{pdf_content}"
            )

    # Select one image (if available)
    selected_image = None
    if image_files:
        selected_image = image_files[0]  # Select the first image
        log.info(f"🖼️ Selected image: {selected_image}")

    # Combine all text content
    combined_text = "\n\n".join(all_text_content)

    file_summary = f"Processed {len(txt_files)} text files, {len(pdf_files)} PDF files, and found {len(image_files)} images."
    log.info(f"✅ Folder processing complete. {file_summary}")

    return combined_text, selected_image
