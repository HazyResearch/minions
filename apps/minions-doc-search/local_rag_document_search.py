#!/usr/bin/env python3
"""
Document Search Script with Multiple Retrievers

This script loads all text files from a specified directory and uses different retrieval methods
to retrieve the top k most relevant documents based on a query.

Supported retrievers:
- bm25: Traditional keyword-based retrieval
- embedding: Dense vector embeddings with SentenceTransformers + FAISS
- mlx: Apple Silicon optimized embeddings with MLX
- gemini: Google Gemini embeddings via API
- embeddinggemma: Google EmbeddingGemma-300m model via SentenceTransformers
- multimodal: ChromaDB + Ollama embeddings
- qdrant: Qdrant + Ollama embeddings

Usage:
  python local_rag_document_search.py --retriever bm25
  python local_rag_document_search.py --retriever embedding
  python local_rag_document_search.py --retriever mlx
  python local_rag_document_search.py --retriever gemini
  python local_rag_document_search.py --retriever embeddinggemma
  python local_rag_document_search.py --retriever multimodal
  python local_rag_document_search.py --retriever qdrant
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

# Add the minions directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../minions'))

from minions.utils.retrievers import bm25_retrieve_top_k_chunks, embedding_retrieve_top_k_chunks, SentenceTransformerEmbeddings, MLXEmbeddings, GeminiEmbeddings
from minions.utils.multimodal_retrievers import retrieve_chunks_from_chroma, retrieve_chunks_from_qdrant
from minions.clients.ollama import OllamaClient
from pydantic import BaseModel


class DocumentAnswer(BaseModel):
    """Structured output for answering questions from documents using RAG"""
    answer: str
    citation: str
    confidence: str  # high, medium, low


def generate_keywords_with_local_model(user_query: str, local_client: OllamaClient) -> Tuple[List[str], Dict[str, float]]:
    """
    Use a local model to generate keywords and weights for BM25 search based on user query.
    
    Args:
        user_query: The user's natural language question
        local_client: Ollama client for generating keywords
        
    Returns:
        Tuple of (keywords_list, weights_dict) where:
        - keywords_list: List of keywords suitable for BM25 search
        - weights_dict: Dictionary mapping keywords to their importance weights
    """
    system_prompt = """You are a keyword extraction and weighting assistant for document search. 
    
Your task is to analyze a user's question and generate the most relevant keywords for BM25 document search, along with importance weights for each keyword.

Guidelines:
1. Extract 2-5 keywords that best capture the essence of the user's question
2. Focus on nouns, technical terms, and specific concepts
3. Consider synonyms and variations (e.g., "FTE" and "headcount" for staffing questions)
4. Avoid common words like "the", "and", "how", "many", "what"
5. Think about what words would appear in documents that answer this question
6. Assign weights from 1.0 to 5.0 based on importance:
   - 5.0: Critical terms that must appear (e.g., "FTE", "approved")
   - 3.0-4.0: Important terms that strongly indicate relevance
   - 1.0-2.0: Supporting terms that add context

Return a JSON object with "keywords" (array of strings) and "weights" (object mapping keywords to weights).

Examples:
User: "How many additional FTEs were approved in the executive team call?"
Response: {"keywords": ["additional", "FTE", "headcount", "approved", "executive"], "weights": {"additional": 3.0, "FTE": 5.0, "headcount": 4.0, "approved": 5.0, "executive": 3.0}}

User: "What was discussed about the marketing budget?"
Response: {"keywords": ["marketing", "budget", "discussion"], "weights": {"marketing": 4.0, "budget": 5.0, "discussion": 2.0}}

User: "Which vendor was selected for the website redesign?"
Response: {"keywords": ["vendor", "website", "redesign", "selected"], "weights": {"vendor": 4.0, "website": 3.0, "redesign": 4.0, "selected": 5.0}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User question: {user_query}"}
    ]
    
    print(f"Generating keywords and weights for query: '{user_query}'")
    print("Asking local model for keyword suggestions...")
    
    try:
        response, usage, done_reason = local_client.chat(messages)
        
        response_text = response[0].strip()
        print(f"Local model response: '{response_text}'")
        
        # Try to parse JSON response
        import json
        try:
            if response_text.strip().startswith('{'):
                parsed = json.loads(response_text)
                keywords = parsed.get("keywords", [])
                weights = parsed.get("weights", {})
                print(f"Parsed keywords: {keywords}")
                print(f"Parsed weights: {weights}")
                return keywords, weights
            else:
                # Fallback: treat as simple keywords
                keywords = response_text.split()
                weights = {keyword: 1.0 for keyword in keywords}
                print(f"Using fallback weights: {weights}")
                return keywords, weights
        except json.JSONDecodeError:
            # Fallback: treat as simple keywords
            keywords = response_text.split()
            weights = {keyword: 1.0 for keyword in keywords}
            print(f"JSON parsing failed, using fallback weights: {weights}")
            return keywords, weights
            
    except Exception as e:
        print(f"Error getting keywords from local model: {e}")
        # Fallback to simple keyword extraction
        simple_keywords = [word for word in user_query.split() 
                          if word.lower() not in ['how', 'many', 'what', 'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for']]
        simple_weights = {keyword: 1.0 for keyword in simple_keywords}
        print(f"Using fallback keywords: {simple_keywords}")
        print(f"Using fallback weights: {simple_weights}")
        return simple_keywords, simple_weights


def load_text_files(directory_path: str, file_extensions: List[str] = None) -> Tuple[List[str], List[str]]:
    """
    Load all text files from the specified directory.
    
    Args:
        directory_path: Path to the directory containing text files
        file_extensions: List of file extensions to include (e.g., ['.md', '.txt', '.py'])
                        If None, loads common text file types
        
    Returns:
        Tuple of (file_contents, file_paths) where:
        - file_contents: List of file contents as strings
        - file_paths: List of corresponding file paths
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Default text file extensions if none specified
    if file_extensions is None:
        file_extensions = [
            '.md', '.txt', '.py', '.js', '.ts', '.jsx', '.tsx', 
            '.html', '.css', '.json', '.yaml', '.yml', '.xml',
            '.rst', '.org', '.tex', '.log', '.cfg', '.ini',
            '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat',
            '.sql', '.r', '.rb', '.go', '.rs', '.cpp', '.c',
            '.h', '.hpp', '.java', '.kt', '.swift', '.php',
            '.pl', '.lua', '.vim', '.dockerfile', '.gitignore'
        ]
    
    # Find all files with specified extensions
    all_files = []
    for ext in file_extensions:
        pattern = f"*{ext}" if ext.startswith('.') else f"*.{ext}"
        all_files.extend(directory.glob(pattern))
    
    # Also include files without extensions (often text files)
    for file_path in directory.iterdir():
        if file_path.is_file() and not file_path.suffix and file_path not in all_files:
            # Try to determine if it's a text file by attempting to read it
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(100)  # Try to read first 100 chars
                all_files.append(file_path)
            except (UnicodeDecodeError, PermissionError):
                # Skip binary files or files we can't read
                continue
    
    if not all_files:
        raise ValueError(f"No readable text files found in {directory_path}")
    
    file_contents = []
    file_paths = []
    skipped_files = []
    
    print(f"Loading text files from: {directory_path}")
    print(f"Found {len(all_files)} potential text files")
    print(f"Supported extensions: {', '.join(file_extensions[:10])}{'...' if len(file_extensions) > 10 else ''}")
    print("-" * 50)
    
    for file_path in sorted(all_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Skip empty files
                if not content.strip():
                    skipped_files.append((file_path.name, "empty file"))
                    continue
                
                # Skip very large files (>10MB) to avoid memory issues
                if len(content) > 10 * 1024 * 1024:
                    skipped_files.append((file_path.name, f"too large ({len(content)//1024//1024}MB)"))
                    continue
                
                file_contents.append(content)
                file_paths.append(str(file_path))
                print(f"✓ Loaded: {file_path.name} ({len(content)} chars)")
                
        except UnicodeDecodeError:
            skipped_files.append((file_path.name, "binary/non-text file"))
        except PermissionError:
            skipped_files.append((file_path.name, "permission denied"))
        except Exception as e:
            skipped_files.append((file_path.name, str(e)))
    
    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files:")
        for filename, reason in skipped_files[:5]:  # Show first 5 skipped files
            print(f"  ✗ {filename}: {reason}")
        if len(skipped_files) > 5:
            print(f"  ... and {len(skipped_files) - 5} more")
    
    print(f"\nSuccessfully loaded {len(file_contents)} documents")
    return file_contents, file_paths


def search_documents(query, documents: List[str], file_paths: List[str], k: int = 5, retriever_type: str = "bm25", weights: Dict[str, float] = None) -> List[Tuple[str, str, float]]:
    """
    Search documents using the specified retriever and return top k results.
    
    Args:
        query: Search query (string for semantic retrievers, list of keywords for BM25)
        documents: List of document contents (treated as chunks)
        file_paths: List of corresponding file paths
        k: Number of top results to return
        retriever_type: Type of retriever to use ("bm25", "embedding", "mlx", "multimodal", "qdrant")
        weights: Dictionary of keyword weights for BM25 retrieval
        
    Returns:
        List of tuples (file_path, content_preview, score)
    """
    print(f"\nUsing retriever type: {retriever_type}")
    if isinstance(query, list):
        print(f"Keywords: {query}")
    else:
        print(f"Query: '{query}'")
    if weights:
        print(f"Weights: {weights}")
    print("-" * 50)
    
    # Factory pattern for retrievers
    retrievers = {
        "bm25": _retrieve_bm25,
        "embedding": _retrieve_embedding,
        "mlx": _retrieve_mlx,
        "gemini": _retrieve_gemini,
        "embeddinggemma": _retrieve_embeddinggemma,
        "multimodal": _retrieve_multimodal,
        "qdrant": _retrieve_qdrant
    }
    
    if retriever_type not in retrievers:
        raise ValueError(f"Unknown retriever type: {retriever_type}. Supported types: {list(retrievers.keys())}")
    
    # Get relevant chunks using the selected retriever
    if retriever_type == "bm25":
        relevant_chunks = retrievers[retriever_type](query, documents, k, weights)
    else:
        relevant_chunks = retrievers[retriever_type](query, documents, k)
    
    # Convert chunks back to results format
    results = []
    for i, chunk in enumerate(relevant_chunks):
        # Find the original document index
        doc_index = documents.index(chunk)
        file_path = file_paths[doc_index]
        
        # Create a preview of the content (first 200 chars)
        content_preview = chunk[:200].replace('\n', ' ').strip()
        if len(chunk) > 200:
            content_preview += "..."
        
        # Use rank as score (higher rank = higher score)
        score = float(k - i)
        
        filename = os.path.basename(file_path)
        print(f"Rank {i+1}: {filename} - {retriever_type.upper()} Score={score:.3f}")
        
        results.append((file_path, content_preview, score))
    
    return results


def _retrieve_bm25(query: List[str], documents: List[str], k: int, weights: Dict[str, float] = None) -> List[str]:
    """BM25 retrieval using keywords and weights."""
    print(f"Searching for keywords: {query}")
    if weights:
        print(f"Using weights: {weights}")
    
    # Use the minions BM25 retriever with weights
    return bm25_retrieve_top_k_chunks(query, documents, weights=weights, k=k)


def _retrieve_embedding(query: str, documents: List[str], k: int) -> List[str]:
    """Dense embedding retrieval using SentenceTransformers + FAISS."""
    print("Using dense embedding retrieval with SentenceTransformers + FAISS")
    
    try:
        return embedding_retrieve_top_k_chunks([query], documents, k=k)
    except ImportError as e:
        print(f"Error: {e}")
        print("Falling back to BM25...")
        return _retrieve_bm25(query.split(), documents, k)


def _retrieve_mlx(query: str, documents: List[str], k: int) -> List[str]:
    """MLX embedding retrieval (Apple Silicon optimized)."""
    print("Using MLX embedding retrieval (Apple Silicon optimized)")
    
    try:
        mlx_model = MLXEmbeddings()
        return embedding_retrieve_top_k_chunks([query], documents, k=k, embedding_model=mlx_model)
    except ImportError as e:
        print(f"Error: {e}")
        print("Falling back to BM25...")
        return _retrieve_bm25(query.split(), documents, k)


def _retrieve_gemini(query: str, documents: List[str], k: int) -> List[str]:
    """Gemini embedding retrieval using Google's Gemini embeddings."""
    print("Using Gemini embedding retrieval with Google's Gemini API")
    
    try:
        gemini_model = GeminiEmbeddings()
        return embedding_retrieve_top_k_chunks([query], documents, k=k, embedding_model=gemini_model)
    except ImportError as e:
        print(f"Error: {e}")
        print("Falling back to BM25...")
        return _retrieve_bm25(query.split(), documents, k)
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure to set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        print("Falling back to BM25...")
        return _retrieve_bm25(query.split(), documents, k)


def _retrieve_embeddinggemma(query: str, documents: List[str], k: int) -> List[str]:
    """EmbeddingGemma retrieval using Google's EmbeddingGemma-300m model."""
    print("Using EmbeddingGemma-300m retrieval with SentenceTransformers")
    
    try:
        # Use SentenceTransformerEmbeddings with EmbeddingGemma model
        embeddinggemma_model = SentenceTransformerEmbeddings(model_name="google/embeddinggemma-300m")
        return embedding_retrieve_top_k_chunks([query], documents, k=k, embedding_model=embeddinggemma_model)
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure sentence-transformers is installed.")
        print("Falling back to BM25...")
        return _retrieve_bm25(query.split(), documents, k)
    except Exception as e:
        print(f"Error loading EmbeddingGemma model: {e}")
        print("This might be due to model download or compatibility issues.")
        print("Falling back to BM25...")
        return _retrieve_bm25(query.split(), documents, k)


def _retrieve_multimodal(query: str, documents: List[str], k: int) -> List[str]:
    """Multimodal retrieval using ChromaDB + Ollama."""
    print("Using multimodal retrieval with ChromaDB + Ollama")
    
    try:
        keywords = query.split()
        return retrieve_chunks_from_chroma(documents, keywords, embedding_model="llama3.2", k=k)
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to BM25...")
        return _retrieve_bm25(query.split(), documents, k)


def _retrieve_qdrant(query: str, documents: List[str], k: int) -> List[str]:
    """Qdrant retrieval using Qdrant + Ollama."""
    print("Using Qdrant retrieval with Qdrant + Ollama")
    
    try:
        keywords = query.split()
        return retrieve_chunks_from_qdrant(documents, keywords, embedding_model="llama3.2", k=k)
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to BM25...")
        return _retrieve_bm25(query.split(), documents, k)


def answer_question_with_ollama(user_query: str, documents: List[str], file_paths: List[str], ollama_client: OllamaClient) -> DocumentAnswer:
    """
    Answer a question using Ollama client with concatenated documents as context using RAG.
    
    Args:
        user_query: The user's question
        documents: List of document contents (top k from retrieval)
        file_paths: List of corresponding file paths
        ollama_client: Ollama client for generating the answer
        
    Returns:
        DocumentAnswer with structured response
    """
    # Concatenate documents with clear separators
    context_parts = []
    for i, (doc, path) in enumerate(zip(documents, file_paths), 1):
        filename = os.path.basename(path)
        context_parts.append(f"=== DOCUMENT {i}: {filename} ===\n{doc}\n")
    
    concatenated_context = "\n".join(context_parts)
    
    system_prompt = """You are an expert at analyzing documents and extracting specific information using retrieval-augmented generation.

Your task is to answer questions based on the provided documents. Read through all the documents carefully and provide:

1. A direct, specific answer to the question
2. An exact citation (quote) from the relevant document that supports your answer
3. Your confidence level (high, medium, low) in the answer

Guidelines:
- If the information is explicitly stated in the documents, use "high" confidence
- If you need to infer from context or combine information from multiple sources, use "medium" confidence  
- If the information is not clearly available in the provided documents, use "low" confidence and state what you found
- For citations, use exact quotes from the documents with clear attribution
- Be precise with numbers, dates, and specific details when available
- If the answer requires information not present in the documents, clearly state this limitation"""

    user_message = f"""Question: {user_query}

Documents:
{concatenated_context}

Please provide a structured answer with the specific information requested, an exact citation from the documents, and your confidence level."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    print(f"\nAsking Ollama to answer: '{user_query}'")
    print(f"Context length: {len(concatenated_context)} characters")
    print("Processing with local model...")
    
    try:
        response, usage, done_reason = ollama_client.chat(messages)
        
        # Parse the structured output
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], DocumentAnswer):
                return response[0]
            else:
                # If not structured, try to parse the text response
                answer_text = response[0]
                
                # Try to parse JSON-like response from the model
                import json
                try:
                    if answer_text.strip().startswith('{'):
                        parsed = json.loads(answer_text)
                        return DocumentAnswer(
                            answer=parsed.get("answer", answer_text),
                            citation=parsed.get("citation", "No citation provided"),
                            confidence=parsed.get("confidence", "medium")
                        )
                except:
                    pass
                
                return DocumentAnswer(
                    answer=answer_text,
                    citation="Raw response - structured parsing failed",
                    confidence="medium"
                )
        else:
            return DocumentAnswer(
                answer="No response generated",
                citation="N/A",
                confidence="low"
            )
            
    except Exception as e:
        print(f"Error getting answer from Ollama: {e}")
        return DocumentAnswer(
            answer=f"Error: {str(e)}",
            citation="N/A",
            confidence="low"
        )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Document search with different retriever types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use BM25 with all supported text file types
  python local_rag_document_search.py --retriever bm25 --documents-path ./docs
  
  # Use EmbeddingGemma with only Python and Markdown files
  python local_rag_document_search.py --retriever embeddinggemma --documents-path ./src --file-extensions .py .md
  
  # Use embedding retriever with specific file types
  python local_rag_document_search.py --retriever embedding --documents-path ./project --file-extensions .txt .json .yaml
  
  # Other retriever examples
  python local_rag_document_search.py --retriever mlx
  python local_rag_document_search.py --retriever gemini
  python local_rag_document_search.py --retriever multimodal
  python local_rag_document_search.py --retriever qdrant
        """
    )
    
    parser.add_argument(
        "--retriever", 
        type=str, 
        default="bm25",
        choices=["bm25", "embedding", "mlx", "gemini", "embeddinggemma", "multimodal", "qdrant"],
        help="Type of retriever to use (default: bm25)"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="gemma3:4b",
        help="Model to use for keyword generation and question answering (default: gemma3:4b)"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default="What are the main topics discussed in these documents?",
        help="Search query (default: generic document question)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top results to return (default: 3)"
    )
    
    parser.add_argument(
        "--documents-path",
        type=str,
        help="Path to directory containing text files"
    )
    
    parser.add_argument(
        "--file-extensions",
        type=str,
        nargs='+',
        help="File extensions to include (e.g., --file-extensions .md .txt .py). If not specified, loads common text file types"
    )
    
    return parser.parse_args()


def main():
    """Main function to demonstrate the document search with different retriever types."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"=== Document Search with {args.retriever.upper()} Retriever ===")
    print(f"User Query: '{args.query}'")
    print("=" * 70)
    
    try:
        # Initialize local client for keyword generation (only for BM25)
        keyword_client = None
        if args.retriever == "bm25":
            print("\nInitializing Ollama client for keyword generation...")
            keyword_client = OllamaClient(
                model_name=args.model_name,
                temperature=0.0,
                max_tokens=50,  # Short response for keywords
                num_ctx=2048,   # Small context for keyword generation
                use_async=False
            )
            print("✓ Keyword generation client initialized")
        
        # Load all text files
        documents, file_paths = load_text_files(args.documents_path, args.file_extensions)
        
        # Generate keywords using local model (only for BM25, other retrievers use the full query)
        if args.retriever == "bm25":
            search_query, search_weights = generate_keywords_with_local_model(args.query, keyword_client)
        else:
            search_query = args.query  # Use full query for semantic retrievers
            search_weights = None
        
        # Perform search with specified retriever
        results = search_documents(search_query, documents, file_paths, args.top_k, args.retriever, search_weights)
        
        # Display results
        print(f"\nTop {len(results)} results for user query:")
        print(f"Original Query: '{args.query}'")
        if args.retriever == "bm25":
            print(f"Generated Keywords: {search_query}")
        print("=" * 70)
        
        for i, (file_path, preview, score) in enumerate(results, 1):
            filename = os.path.basename(file_path)
            print(f"\n{i}. {filename}")
            print(f"   {args.retriever.upper()} Score: {score:.3f}")
            print(f"   Path: {file_path}")
            print(f"   Preview: {preview}")
        
        # Show search results summary
        print(f"\n{'='*70}")
        print(f"Found {len(results)} relevant documents")
        
        # Show search results summary
        print(f"\n{'='*70}")
        print("SEARCH RESULTS SUMMARY:")
        print(f"- User query: '{args.query}'")
        if args.retriever == "bm25":
            print(f"- Local AI-generated keywords: {search_query}")
        print(f"- Retriever type: {args.retriever}")
        print(f"- Documents found: {len(results)}")
        
        # Initialize Ollama client for answering the question (can reuse the same client)
        print(f"\n{'='*70}")
        print("ANSWERING QUESTION WITH OLLAMA")
        print("="*70)
        
        print("Initializing Ollama client for question answering...")
        answer_client = OllamaClient(
            model_name=args.model_name,  # Same model
            temperature=0.0,
            max_tokens=500,
            num_ctx=4096,  # Optimized context window (we know we need ~2400 tokens)
            structured_output_schema=DocumentAnswer,
            use_async=False
        )
        print("✓ Answer generation client initialized")
        
        # Extract top 5 documents for context
        top_k_documents = []
        top_k_paths = []
        for file_path, _, _ in results[:args.top_k]:
            # Find the full document content
            for i, path in enumerate(file_paths):
                if path == file_path:
                    top_k_documents.append(documents[i])
                    top_k_paths.append(path)
                    break
        
        print(f"Using top {len(top_k_documents)} documents as context...")
        
        # Calculate actual character and token counts
        total_chars = sum(len(doc) for doc in top_k_documents)
        estimated_tokens = total_chars // 4  # Rough estimate: 4 chars per token
        
        print(f"Character count analysis:")
        for i, (doc, path) in enumerate(zip(top_k_documents, top_k_paths), 1):
            filename = os.path.basename(path)
            print(f"  Document {i} ({filename}): {len(doc)} chars")
        
        print(f"Total characters: {total_chars}")
        print(f"Estimated tokens: {estimated_tokens} (÷4 rule)")
        print(f"Context window used: 4,096 tokens")
        print(f"Utilization: {(estimated_tokens/4096)*100:.1f}%")
        
        # Get answer from Ollama
        answer = answer_question_with_ollama(args.query, top_k_documents, top_k_paths, answer_client)
        
        # Display the answer
        print(f"\n{'='*70}")
        print("OLLAMA ANSWER:")
        print("="*70)
        print(f"Question: {args.query}")
        print(f"\nAnswer: {answer.answer}")
        print(f"\nCitation: {answer.citation}")
        print(f"Confidence: {answer.confidence}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
