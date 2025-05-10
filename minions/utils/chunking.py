from typing import List, Optional
import re
import json
import ast


def chunk_by_section(
    doc: str, max_chunk_size: int = 3000, overlap: int = 20
) -> List[str]:
    sections = []
    start = 0
    while start < len(doc):
        end = start + max_chunk_size
        sections.append(doc[start:end])
        start += max_chunk_size - overlap
    return sections


def chunk_by_page(doc: str, page_markers: Optional[List[str]] = None) -> List[str]:
    if page_markers is None:
        page_markers = [
            r"\f",  # form feed character
            r"^page\s+\d+(\s+of\s+\d+)?\s*$",  # "Page X" or "Page X of Y"
            r"^[\s_\-()]*\d+[\s_\-()]*$",  # standalone numbers or decorated numbers (e.g. - 3 -)
            r"^[-=#]{3,}\s*.*page.*[-=#]{3,}\s*$",  # lines like --- page ---, === pg ===, etc.
            r"^\s*[\[<\(]\s*page(?:\s+\d+)?\s*[\]>)]\s*$",  # lines like [page] or [page 3] (or any bracket variant)
        ]
    pattern = "|".join(page_markers)
    compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    matches = list(re.finditer(compiled_pattern, doc))
    if not matches:
        return [doc]
    pages = []
    start = 0
    for match in matches:
        chunk = doc[start : match.start()]
        if chunk:
            pages.append(chunk)
        start = match.end()
    last_chunk = doc[start:]
    if last_chunk:
        pages.append(last_chunk)
    print(pages)
    return pages


def chunk_sentences(
    sentences: List[str], max_chunk_size: int, overlap_sentences: int
) -> List[str]:
    """
    Helper to group sentences into chunks with overlap.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sep = " " if current_chunk else ""
        new_length = current_length + len(sep) + len(sentence)
        if new_length > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap = current_chunk[-overlap_sentences:] if overlap_sentences else []
            current_chunk = overlap + [sentence]
            current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
        else:
            current_chunk.append(sentence)
            current_length = new_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def chunk_by_paragraph(
    doc: str, max_chunk_size: int = 1500, overlap_sentences: int = 0
) -> List[str]:
    sentence_regex = re.compile(r"(?<=[.!?])\s+")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc) if p.strip()]

    chunks = []
    current_paragraphs = []
    current_length = 0

    for paragraph in paragraphs:
        if (
            len(paragraph) > max_chunk_size
        ):  # if the paragraph is too large, break it into sentences
            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))
                current_paragraphs = []
                current_length = 0
            sentences = sentence_regex.split(paragraph)
            sentence_chunks = chunk_sentences(
                sentences, max_chunk_size, overlap_sentences
            )
            chunks.extend(sentence_chunks)
        else:  # normal paragraph that fits within max_chunk_size
            sep_len = 2 if current_paragraphs else 0  # "\n\n" separator length
            candidate_length = current_length + sep_len + len(paragraph)
            if candidate_length > max_chunk_size and current_paragraphs:
                chunk_text = "\n\n".join(current_paragraphs)
                chunks.append(chunk_text)

                if overlap_sentences:
                    sentences = sentence_regex.split(current_paragraphs[-1])
                    overlap = " ".join(
                        sentences[-min(overlap_sentences, len(sentences)) :]
                    )
                    current_paragraphs = [overlap, paragraph]
                    current_length = len(overlap) + 2 + len(paragraph)
                else:
                    current_paragraphs = [paragraph]
                    current_length = len(paragraph)
            else:
                current_paragraphs.append(paragraph)
                current_length = candidate_length

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))
    return chunks


def extract_imports(lines: List[str], tree: ast.AST) -> str:
    import_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_lines.add(lines[node.lineno - 1])
    if import_lines:
        return "\n".join(import_lines) + "\n\n"
    return ""


def extract_function_header(lines: List[str], start_line: int) -> List[str]:
    header_lines = []
    paren_count = 0
    for i in range(start_line, len(lines)):
        line = lines[i]
        header_lines.append(line)
        paren_count += line.count("(") - line.count(")")
        if ":" in line and paren_count == 0:  # header is complete
            break
    return header_lines


def extract_function(lines: List[str], node: ast.FunctionDef) -> str:
    if node.decorator_list:
        start_line = min(dec.lineno for dec in node.decorator_list) - 1
    else:
        start_line = node.lineno - 1
    end_line = getattr(node, "end_lineno", node.lineno)
    function_chunk = "\n".join(lines[start_line:end_line])
    return function_chunk


def chunk_by_code(doc: str, functions_per_chunk: int = 1) -> List[str]:
    """
    Splits Python code into chunks by function (with decorators).
    Optionally specify number of functions per chunk
    """
    try:
        tree = ast.parse(doc)
    except SyntaxError:
        return [doc]
    lines = doc.splitlines()
    functions = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):  # stand-alone functions
            functions.append(extract_function(lines, node))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    functions.append(extract_function(lines, item))
    chunks = []
    for i in range(0, len(functions), functions_per_chunk):
        batch = functions[i : i + functions_per_chunk]
        chunk = "\n\n".join(batch)
        chunks.append(chunk)

    return chunks


def chunk_by_function_and_class(doc: str) -> List[str]:
    """
    Splits Python code into chunks by function and class definitions.

    For stand-alone functions, the chunk is the entire function (with decorators).
    For classes, two types of chunks:
      1) Each method (prefixed with the class header)
      2) A class structure chunk with the class header, class variables, and method signatures

    Each chunk is prepended with import statements.
    """
    try:
        tree = ast.parse(doc)
    except SyntaxError:
        return [doc]
    lines = doc.splitlines()
    import_lines = extract_imports(lines, tree)
    chunks = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):  # stand-alone functions
            chunks.append(import_lines + extract_function(lines, node))
        elif isinstance(node, ast.ClassDef):
            class_header = lines[node.lineno - 1].strip()
            class_lines = [class_header]  # chunk for class structure

            for item in node.body:
                start_line = item.lineno - 1
                if isinstance(item, ast.Assign):  # add class variables to class chunk
                    end_line = getattr(item, "end_lineno", item.lineno)
                    for i in range(start_line, end_line):
                        class_lines.append("    " + lines[i])
                elif isinstance(item, ast.FunctionDef):
                    # add function chunk
                    chunks.append(
                        f"{import_lines}{class_header}\n{extract_function(lines, item)}"
                    )

                    # add function header to class chunk
                    for line in extract_function_header(lines, start_line):
                        class_lines.append("    " + line)

            class_structure = "\n".join(class_lines)
            chunks.append(import_lines + class_structure)
    return chunks
