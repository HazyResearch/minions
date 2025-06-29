# Minions Document Search

A powerful document search and question-answering tool that supports multiple retrieval methods for finding relevant information in markdown documents.

## Features

### Multiple Retrieval Methods
- **BM25**: Traditional keyword-based retrieval with AI-generated keywords and weights
- **Embedding**: Dense vector embeddings using SentenceTransformers + FAISS
- **MLX**: Apple Silicon optimized embeddings with MLX
- **Multimodal**: ChromaDB + Ollama embeddings

### AI-Powered Features
- **Smart Keyword Generation**: Uses local LLMs to generate optimized keywords for BM25 search
- **Question Answering**: Structured answers with citations and confidence levels
- **Context-Aware Search**: Optimized context windows for efficient processing

## Installation

1. Install the main minions package from the root directory:
```bash
cd ../../
pip install -e .
```

2. Install additional dependencies for this app:
```bash
cd apps/minions-doc-search
pip install -r requirements.txt
```

## Usage

### Basic Usage

Search for documents using different retrieval methods:

```bash
# Use BM25 with AI-generated keywords (default)
python local_rag_document_search.py --retriever bm25

# Use dense embeddings
python local_rag_document_search.py --retriever embedding

# Use MLX embeddings (Apple Silicon)
python local_rag_document_search.py --retriever mlx

# Use multimodal retrieval
python local_rag_document_search.py --retriever multimodal
```

### Advanced Options

```bash
# Custom query and settings
python local_rag_document_search.py \
  --retriever bm25 \
  --query "How many FTEs were approved?" \
  --top-k 5 \
  --model_name "gemma3:4b" \
  --documents-path "/path/to/documents"
```

### Command Line Arguments

- `--retriever`: Retrieval method (`bm25`, `embedding`, `mlx`, `multimodal`)
- `--query`: Search query (default: FTE approval question)
- `--top-k`: Number of results to return (default: 3)
- `--model_name`: Ollama model for keyword generation and QA (default: `gemma3:4b`)
- `--documents-path`: Path to directory containing .md files

## How It Works

### 1. Document Loading
The script loads all `.md` files from the specified directory and prepares them for search.

### 2. Retrieval Process

#### BM25 Retrieval
1. Uses a local LLM to generate optimized keywords from natural language queries
2. Assigns importance weights to each keyword (1.0-5.0 scale)
3. Performs BM25 search with weighted keywords

#### Other Retrievers
- **Embedding**: Creates dense vector representations using SentenceTransformers
- **MLX**: Optimized for Apple Silicon using MLX framework
- **Multimodal**: Uses ChromaDB with Ollama embeddings

### 3. Question Answering
1. Takes top-k retrieved documents as context
2. Uses structured output with Pydantic models
3. Returns answer, citation, and confidence level

## Output Format

The tool provides:
- **Ranked search results** with scores and previews
- **Structured answers** with:
  - Direct answer to the question
  - Exact citation from source documents
  - Confidence level (high/medium/low)

## Example Output

```
=== Document Search with BM25 Retriever ===
User Query: 'How many additional FTEs were approved in the executive team call?'

Top 3 results for user query:
1. executive_leadership_team_monthly_update.md
   BM25 Score: 3.000
   Preview: In the executive leadership team meeting, we discussed...

OLLAMA ANSWER:
Question: How many additional FTEs were approved in the executive team call?

Answer: 5 additional FTEs were approved for the engineering team.
Citation: "The executive team approved 5 additional FTE positions for Q4 hiring"
Confidence: high
```

## Dependencies

The app requires:
- **Core**: pydantic, argparse, pathlib
- **Optional**: sentence-transformers, faiss-cpu, chromadb, mlx
- **Main minions package** (install from root directory)

## Prerequisites

- **Ollama**: Must be installed and running for local LLM functionality
- **Model**: Default model (`gemma3:4b`) should be available in Ollama
- **Documents**: Directory containing `.md` files to search

## Troubleshooting

### Common Issues
1. **Model not found**: Ensure the specified model is installed in Ollama
2. **Import errors**: Install optional dependencies based on chosen retriever
3. **MLX issues**: MLX only works on Apple Silicon Macs
4. **No documents found**: Check that the documents path contains `.md` files

### Performance Tips
- Use BM25 for keyword-focused searches
- Use embedding methods for semantic similarity
- Adjust `--top-k` based on document collection size
- MLX provides best performance on Apple Silicon

## Contributing

This app is part of the larger Minions project. See the main project README for contribution guidelines. 