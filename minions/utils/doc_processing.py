import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import requests
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
from docling_core.types.doc import ImageRefMode
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, stream_generate


def pdf_to_images(
    pdf_data: Union[str, Path, bytes], dpi: int = 300
) -> List[Image.Image]:
    """
    Convert a PDF file to a list of PIL Image objects.

    Args:
        pdf_data: Path to the PDF file or PDF data as bytes
        dpi: Resolution for the conversion (higher means better quality but larger images)

    Returns:
        List of PIL Image objects, one per page
    """
    if isinstance(pdf_data, (str, Path)):
        return convert_from_path(pdf_data, dpi=dpi)
    elif isinstance(pdf_data, bytes):
        return convert_from_bytes(pdf_data, dpi=dpi)
    else:
        raise TypeError("pdf_data must be a string path, Path object, or bytes")


def img_to_markdown_smoldocling(
    image_data: Union[str, Path, Image.Image, bytes],
    prompt: str = "Convert this page to docling.",
    model_path: str = "ds4sd/SmolDocling-256M-preview-mlx-bf16",
    verbose: bool = False,
    max_tokens: int = 4096,
    model_and_processor=None,
    config=None,
) -> str:
    """
    Convert an image or PDF to markdown using SmolDocling.

    Args:
        image_data: Path to image file, URL, PIL Image object, image/PDF data as bytes,
                   or base64-encoded image string
        prompt: Prompt to guide the conversion
        model_path: Path to the SmolDocling model
        verbose: Whether to print progress
        max_tokens: Maximum number of tokens to generate
        model_and_processor: Optional tuple of (model, processor) to avoid reloading
        config: Optional model config to avoid reloading

    Returns:
        Markdown representation of the document
    """
    # Load the model if not provided
    if model_and_processor is None or config is None:
        model, processor = load(model_path)
        config = load_config(model_path)
    else:
        model, processor = model_and_processor

    # Handle different input types
    if isinstance(image_data, (str, Path)):
        # Check if it's a base64 encoded string
        if (
            isinstance(image_data, str)
            and image_data.startswith(("data:image", "data:application/pdf"))
            or (len(image_data) > 100 and "," in image_data[:100])
        ):
            # Extract the base64 data after the comma if it's a data URL
            if "," in image_data:
                base64_data = image_data.split(",", 1)[1]
            else:
                base64_data = image_data

            import base64

            image_bytes = base64.b64decode(base64_data)

            try:
                # Try to open as an image
                pil_image = Image.open(BytesIO(image_bytes))
            except Exception:
                # If that fails, try to process as PDF
                try:
                    images = pdf_to_images(image_bytes)
                    if not images:
                        raise ValueError("Could not extract images from PDF bytes")
                    pil_image = images[0]  # Use first page
                except Exception as e:
                    raise ValueError(
                        f"Could not process base64 data as image or PDF: {e}"
                    )
        # Check if it's a PDF
        elif str(image_data).lower().endswith(".pdf"):
            # Convert first page of PDF to image
            images = pdf_to_images(image_data)
            if not images:
                raise ValueError(f"Could not extract images from PDF: {image_data}")
            pil_image = images[0]  # Use first page
        elif urlparse(str(image_data)).scheme != "":  # it's a URL
            response = requests.get(image_data, stream=True, timeout=10)
            response.raise_for_status()
            pil_image = Image.open(BytesIO(response.content))
        else:  # Local image file
            pil_image = Image.open(image_data)
    elif isinstance(image_data, Image.Image):
        pil_image = image_data
    elif isinstance(image_data, bytes):
        # Try to determine if it's a PDF or image
        try:
            # First try to open as an image
            pil_image = Image.open(BytesIO(image_data))
        except Exception:
            # If that fails, try to process as PDF
            try:
                images = pdf_to_images(image_data)
                if not images:
                    raise ValueError("Could not extract images from PDF bytes")
                pil_image = images[0]  # Use first page
            except Exception as e:
                raise ValueError(f"Could not process bytes as image or PDF: {e}")
    else:
        raise TypeError(
            "image_data must be a string path, Path object, PIL Image, bytes, or base64-encoded string"
        )

    # Apply chat template
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)

    # Generate output
    output = ""
    for token in stream_generate(
        model,
        processor,
        formatted_prompt,
        [pil_image],
        max_tokens=max_tokens,
        verbose=verbose,
    ):
        output += token.text
        if verbose:
            print(token.text, end="")
        if "</doctag>" in token.text:
            break

    # Populate document
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([output], [pil_image])
    doc = DoclingDocument(name="Document")
    doc.load_from_doctags(doctags_doc)

    # Export as markdown
    return doc.export_to_markdown()


def process_pdf_to_markdown(
    pdf_data: Union[str, Path, bytes],
    prompt: str = "Convert this page to docling.",
    model_path: str = "ds4sd/SmolDocling-256M-preview-mlx-bf16",
    verbose: bool = False,
    max_tokens: int = 4096,
    return_type: str = "string",
) -> List[str]:
    """
    Process all pages of a PDF and convert each to markdown.

    Args:
        pdf_data: Path to the PDF file or PDF data as bytes
        prompt: Prompt to guide the conversion
        model_path: Path to the SmolDocling model
        verbose: Whether to print progress
        max_tokens: Maximum number of tokens to generate
        return_type: Whether to return a concatenated string or a list of strings
    Returns:
        List of markdown strings, one per page
    """
    images = pdf_to_images(pdf_data)
    markdown_pages = []

    # Load model once for all pages
    model, processor = load(model_path)
    config = load_config(model_path)

    for i, img in enumerate(images):
        if verbose:
            print(f"Processing page {i+1}/{len(images)}...")
        markdown = img_to_markdown_smoldocling(
            img,
            prompt,
            model_path,
            verbose,
            max_tokens,
            model_and_processor=(model, processor),
            config=config,
        )
        markdown_pages.append(markdown)

    if return_type == "string":
        return "\n".join(markdown_pages)
    else:
        return markdown_pages
