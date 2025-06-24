#!/usr/bin/env python3
"""
BM25 Document Search Script

This script loads all .md files from a specified directory and uses BM25 keyword search 
to retrieve the top k most relevant documents based on a query.
"""

import os
import sys
import glob
from pathlib import Path
from typing import List, Tuple

# Add the minions directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'minions'))

from minions.utils.retrievers import bm25_retrieve_top_k_chunks
from minions.clients.ollama import OllamaClient
from pydantic import BaseModel


class MeetingAnswer(BaseModel):
    """Structured output for answering questions from meeting summaries"""
    answer: str
    citation: str
    confidence: str  # high, medium, low


def generate_keywords_with_local_model(user_query: str, local_client: OllamaClient) -> str:
    """
    Use a local model to generate keywords for BM25 search based on user query.
    
    Args:
        user_query: The user's natural language question
        local_client: Ollama client for generating keywords
        
    Returns:
        String of keywords suitable for BM25 search
    """
    system_prompt = """You are a keyword extraction assistant for document search. 
    
Your task is to analyze a user's question and generate the most relevant keywords for BM25 document search.

Guidelines:
1. Extract 2-5 keywords that best capture the essence of the user's question
2. Focus on nouns, technical terms, and specific concepts
3. Consider synonyms and variations (e.g., "FTE" and "headcount" for staffing questions)
4. Avoid common words like "the", "and", "how", "many", "what"
5. Think about what words would appear in documents that answer this question

Return only the keywords separated by spaces, no explanation or formatting.

Examples:
User: "How many additional FTEs were approved in the executive team call?"
Keywords: additional FTE headcount approved executive

User: "What was discussed about the marketing budget?"
Keywords: marketing budget discussion

User: "Which vendor was selected for the website redesign?"
Keywords: vendor website redesign selected"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User question: {user_query}"}
    ]
    
    print(f"Generating keywords for query: '{user_query}'")
    print("Asking local model for keyword suggestions...")
    
    try:
        response, usage, done_reason = local_client.chat(messages)
        
        keywords = response[0].strip()
        print(f"Local model suggested keywords: '{keywords}'")
        return keywords
    except Exception as e:
        print(f"Error getting keywords from local model: {e}")
        # Fallback to simple keyword extraction
        simple_keywords = " ".join([word for word in user_query.split() 
                                  if word.lower() not in ['how', 'many', 'what', 'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for']])
        print(f"Using fallback keywords: '{simple_keywords}'")
        return simple_keywords


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


def search_documents_bm25(query: str, documents: List[str], file_paths: List[str], k: int = 5) -> List[Tuple[str, str, float]]:
    """
    Search documents using BM25 and return top k results.
    
    Args:
        query: Search query string
        documents: List of document contents
        file_paths: List of corresponding file paths
        k: Number of top results to return
        
    Returns:
        List of tuples (file_path, content_preview, score)
    """
    # Convert query to keywords (split by spaces)
    keywords = query.split()
    
    print(f"\nSearching for keywords: {keywords}")
    print(f"Query: '{query}'")
    print("-" * 50)
    
    # Debug: Let's manually check BM25 scores for each document
    from rank_bm25 import BM25Plus
    import numpy as np
    
    # Tokenize documents for BM25 (split by spaces and lowercase)
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Plus(tokenized_docs)
    
    # Calculate BM25 scores for each keyword and aggregate
    print("BM25 Score Analysis:")
    print("-" * 30)
    
    final_scores = np.zeros(len(documents))
    for keyword in keywords:
        keyword_scores = bm25.get_scores(keyword.lower())
        final_scores += keyword_scores
        print(f"Keyword '{keyword}' scores: {keyword_scores}")
    
    print(f"Final aggregated BM25 scores: {final_scores}")
    
    # Get top k indices by BM25 score
    top_k_indices = np.argsort(final_scores)[::-1][:k]
    
    print(f"Top {k} document indices by BM25 score: {top_k_indices}")
    
    # Build results using the BM25 ranking
    results = []
    for rank, doc_index in enumerate(top_k_indices):
        file_path = file_paths[doc_index]
        doc = documents[doc_index]
        bm25_score = final_scores[doc_index]
        
        # Create a preview of the content (first 200 chars)
        content_preview = doc[:200].replace('\n', ' ').strip()
        if len(doc) > 200:
            content_preview += "..."
        
        # Calculate keyword frequency for comparison
        keyword_count = sum(doc.lower().count(keyword.lower()) for keyword in keywords)
        
        filename = os.path.basename(file_path)
        print(f"Rank {rank+1}: {filename} - BM25={bm25_score:.3f}, KeywordCount={keyword_count}")
        
        results.append((file_path, content_preview, bm25_score))
    
    return results


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


def main():
    """Main function to demonstrate the BM25 document search with local model query processing."""
    
    # Configuration
    documents_path = "/Users/biderman/Dropbox/Stanford/Dan/minions/data/meeting_summaries"
    user_query = "how many additional FTEs were approved in the executive team call?"
    top_k = 5
    
    print("=== BM25 Document Search with Local Model Query Processing ===")
    print(f"User Query: '{user_query}'")
    print("=" * 70)
    
    try:
        # Initialize local client for keyword generation (using lighter model)
        print("\nInitializing Ollama client for keyword generation...")
        keyword_client = OllamaClient(
            model_name="llama3.2",
            temperature=0.0,
            max_tokens=50,  # Short response for keywords
            num_ctx=2048,   # Small context for keyword generation
            use_async=False
        )
        print("✓ Keyword generation client initialized")
        
        # Load all markdown files
        documents, file_paths = load_markdown_files(documents_path)
        
        # Generate keywords using local model
        search_keywords = generate_keywords_with_local_model(user_query, keyword_client)
        
        # Perform BM25 search
        results = search_documents_bm25(search_keywords, documents, file_paths, top_k)
        
        # Display results
        print(f"\nTop {len(results)} results for user query:")
        print(f"Original Query: '{user_query}'")
        print(f"Generated Keywords: '{search_keywords}'")
        print("=" * 70)
        
        for i, (file_path, preview, score) in enumerate(results, 1):
            filename = os.path.basename(file_path)
            print(f"\n{i}. {filename}")
            print(f"   BM25 Score: {score:.3f}")
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
                    print(f"Expected file ranked #{i} with BM25 score: {score:.3f}")
                    break
        
        # Show improvement from local model keyword generation
        print(f"\n{'='*70}")
        print("SEARCH RESULTS SUMMARY:")
        print(f"- User query: '{user_query}'")
        print(f"- Local AI-generated keywords: '{search_keywords}'")
        print(f"- Search success: {'Yes' if found_expected else 'No'}")
        
        # Initialize Ollama client for answering the question (can reuse the same client)
        print(f"\n{'='*70}")
        print("ANSWERING QUESTION WITH OLLAMA")
        print("="*70)
        
        print("Initializing Ollama client for question answering...")
        answer_client = OllamaClient(
            model_name="llama3.2",  # Same model
            temperature=0.0,
            max_tokens=500,
            num_ctx=4096,  # Optimized context window (we know we need ~2400 tokens)
            structured_output_schema=MeetingAnswer,
            use_async=False
        )
        print("✓ Answer generation client initialized")
        
        # Extract top 5 documents for context
        top_5_documents = []
        top_5_paths = []
        for file_path, _, _ in results[:5]:
            # Find the full document content
            for i, path in enumerate(file_paths):
                if path == file_path:
                    top_5_documents.append(documents[i])
                    top_5_paths.append(path)
                    break
        
        print(f"Using top {len(top_5_documents)} documents as context...")
        
        # Calculate actual character and token counts
        total_chars = sum(len(doc) for doc in top_5_documents)
        estimated_tokens = total_chars // 4  # Rough estimate: 4 chars per token
        
        print(f"Character count analysis:")
        for i, (doc, path) in enumerate(zip(top_5_documents, top_5_paths), 1):
            filename = os.path.basename(path)
            print(f"  Document {i} ({filename}): {len(doc)} chars")
        
        print(f"Total characters: {total_chars}")
        print(f"Estimated tokens: {estimated_tokens} (÷4 rule)")
        print(f"Context window used: 4,096 tokens")
        print(f"Utilization: {(estimated_tokens/4096)*100:.1f}%")
        
        # Get answer from Ollama
        answer = answer_question_with_ollama(user_query, top_5_documents, top_5_paths, answer_client)
        
        # Display the answer
        print(f"\n{'='*70}")
        print("OLLAMA ANSWER:")
        print("="*70)
        print(f"Question: {user_query}")
        print(f"\nAnswer: {answer.answer}")
        print(f"\nCitation: {answer.citation}")
        print(f"Confidence: {answer.confidence}")
        
        # Verify against expected result
        expected_result = "### Resource Requests Approved\n- Additional headcount: 3 FTEs"
        print(f"\n{'='*70}")
        print("VERIFICATION:")
        print("="*70)
        print(f"Expected result contains: '{expected_result}'")
        
        # Check if the answer contains the key information
        answer_lower = answer.answer.lower()
        contains_3_ftes = "3" in answer.answer and ("fte" in answer_lower or "headcount" in answer_lower)
        contains_additional = "additional" in answer_lower
        
        print(f"Answer contains '3 FTEs/headcount': {contains_3_ftes}")
        print(f"Answer contains 'additional': {contains_additional}")
        print(f"Overall match: {'✅ PASS' if (contains_3_ftes and contains_additional) else '❌ FAIL'}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())