#!/usr/bin/env python3
"""
Document Search Script with Multiple Retrievers

This script loads all .md files from a specified directory and uses different retrieval methods
to retrieve the top k most relevant documents based on a query.

Supported retrievers:
- bm25: Traditional keyword-based retrieval
- embedding: Dense vector embeddings with SentenceTransformers + FAISS
- mlx: Apple Silicon optimized embeddings with MLX
- multimodal: ChromaDB + Ollama embeddings

Usage:
  python local_rag_document_search.py --retriever bm25
  python local_rag_document_search.py --retriever embedding
  python local_rag_document_search.py --retriever mlx
  python local_rag_document_search.py --retriever multimodal
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

# Add the minions directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../minions'))

from minions.utils.retrievers import bm25_retrieve_top_k_chunks, embedding_retrieve_top_k_chunks, EmbeddingModel
from minions.utils.mlx_embeddings import MLXEmbeddings
from minions.utils.multimodal_retrievers import retrieve_chunks_from_chroma
from minions.clients.ollama import OllamaClient
from pydantic import BaseModel


class MeetingAnswer(BaseModel):
    """Structured output for answering questions from meeting summaries"""
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


def load_markdown_files(directory_path: str) -> Tuple[List[str], List[str]]:
    """
    Load all .md files from the specified directory.
    
    Args:
        directory_path: Path to the directory containing markdown files
        
    Returns:
        Tuple of (file_contents, file_paths) where:
        - file_contents: List of file contents as strings
        - file_paths: List of corresponding file paths
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Find all .md files in the directory
    md_files = list(directory.glob("*.md"))
    
    if not md_files:
        raise ValueError(f"No .md files found in {directory_path}")
    
    file_contents = []
    file_paths = []
    
    print(f"Loading markdown files from: {directory_path}")
    print(f"Found {len(md_files)} .md files")
    print("-" * 50)
    
    for file_path in sorted(md_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                file_contents.append(content)
                file_paths.append(str(file_path))
                print(f"✓ Loaded: {file_path.name} ({len(content)} chars)")
        except Exception as e:
            print(f"✗ Error loading {file_path.name}: {e}")
    
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
        retriever_type: Type of retriever to use ("bm25", "embedding", "mlx", "multimodal")
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
        "multimodal": _retrieve_multimodal
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


def answer_question_with_ollama(user_query: str, documents: List[str], file_paths: List[str], ollama_client: OllamaClient) -> MeetingAnswer:
    """
    Answer a question using Ollama client with concatenated documents as context.
    
    Args:
        user_query: The user's question
        documents: List of document contents (top k from BM25 search)
        file_paths: List of corresponding file paths
        ollama_client: Ollama client for generating the answer
        
    Returns:
        MeetingAnswer with structured response
    """
    # Concatenate documents with clear separators
    context_parts = []
    for i, (doc, path) in enumerate(zip(documents, file_paths), 1):
        filename = os.path.basename(path)
        context_parts.append(f"=== DOCUMENT {i}: {filename} ===\n{doc}\n")
    
    concatenated_context = "\n".join(context_parts)
    
    system_prompt = """You are an expert at analyzing meeting summaries and extracting specific information.

Your task is to answer questions based on the provided meeting summaries. Read through all the documents carefully and provide:

1. A direct, specific answer to the question
2. An exact citation (quote) from the relevant meeting summary that supports your answer
3. Your confidence level (high, medium, low) in the answer

Guidelines:
- If the information is explicitly stated, use "high" confidence
- If you need to infer from context, use "medium" confidence  
- If the information is not clearly available, use "low" confidence and state what you found
- For citations, use exact quotes from the documents
- Be precise with numbers, dates, and specific details"""

    user_message = f"""Question: {user_query}

Meeting Summaries:
{concatenated_context}

Please provide a structured answer with the specific information requested, an exact citation from the meetings, and your confidence level."""

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
            if isinstance(response[0], MeetingAnswer):
                return response[0]
            else:
                # If not structured, try to parse the text response
                answer_text = response[0]
                
                # Try to parse JSON-like response from the model
                import json
                try:
                    if answer_text.strip().startswith('{'):
                        parsed = json.loads(answer_text)
                        return MeetingAnswer(
                            answer=parsed.get("answer", answer_text),
                            citation=parsed.get("citation", "No citation provided"),
                            confidence=parsed.get("confidence", "medium")
                        )
                except:
                    pass
                
                return MeetingAnswer(
                    answer=answer_text,
                    citation="Raw response - structured parsing failed",
                    confidence="medium"
                )
        else:
            return MeetingAnswer(
                answer="No response generated",
                citation="N/A",
                confidence="low"
            )
            
    except Exception as e:
        print(f"Error getting answer from Ollama: {e}")
        return MeetingAnswer(
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
  python local_rag_document_search.py --retriever bm25
  python local_rag_document_search.py --retriever embedding
  python local_rag_document_search.py --retriever mlx
  python local_rag_document_search.py --retriever multimodal
        """
    )
    
    parser.add_argument(
        "--retriever", 
        type=str, 
        default="bm25",
        choices=["bm25", "embedding", "mlx", "multimodal"],
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
        default="how many additional FTEs were approved in the executive team call?",
        help="Search query (default: FTE approval question)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top results to return (default: 5)"
    )
    
    parser.add_argument(
        "--documents-path",
        type=str,
        default="/Users/biderman/Dropbox/Stanford/Dan/minions/data/meeting_summaries",
        help="Path to directory containing markdown files"
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
        
        # Load all markdown files
        documents, file_paths = load_markdown_files(args.documents_path)
        
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
        
        # Check if the expected file is in the results
        expected_file = "6_executive_leadership_team_monthly_update.md"
        found_expected = any(expected_file in result[0] for result in results)
        
        print(f"\n{'='*70}")
        print(f"Expected file '{expected_file}' found in results: {found_expected}")
        
        if found_expected:
            # Find the rank of the expected file
            for i, (file_path, _, score) in enumerate(results, 1):
                if expected_file in file_path:
                    print(f"Expected file ranked #{i} with {args.retriever.upper()} score: {score:.3f}")
                    break
        
        # Show improvement from local model keyword generation
        print(f"\n{'='*70}")
        print("SEARCH RESULTS SUMMARY:")
        print(f"- User query: '{args.query}'")
        if args.retriever == "bm25":
            print(f"- Local AI-generated keywords: {search_query}")
        print(f"- Retriever type: {args.retriever}")
        print(f"- Search success: {'Yes' if found_expected else 'No'}")
        
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
            structured_output_schema=MeetingAnswer,
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