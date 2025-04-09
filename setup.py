from setuptools import setup, find_packages

setup(
    name="minions",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ollama>=0.1.6",  # for local LLM
        "streamlit==1.42.2",  # for the UI
        "openai>=1.12.0",  # for OpenAI client
        "anthropic>=0.18.1",  # for Anthropic client
        "together>=0.2.0",  # for Together client
        "groq>=0.4.0",  # for Groq client
        "requests>=2.31.0",  # for API calls
        "tiktoken>=0.5.2",  # for token counting
        "pymupdf>=1.23.8",  # for PDF processing
        "st-theme>=0.1.0",
        "mcp>=0.1.0",
        "spacy>=3.7.2",  # for PII extraction
        "rank_bm25>=0.2.2",  # for smart retrieval
        "PyMuPDF>=1.23.8",  # for PDF handling
        "firecrawl-py>=0.1.0",  # for scraping urls
        "google-genai>=0.3.0",  # for Gemini client
        "pandas>=2.0.0",  # for data manipulation
        "numpy>=1.24.0",  # for numerical operations
        "python-dotenv>=1.0.0",  # for environment variables
        "tqdm>=4.65.0",  # for progress bars
        "pytest>=7.4.0",  # for testing
        "pytest-cov>=4.1.0",  # for test coverage
        "PyPDF2>=3.0.0",  # for PDF processing
        "pdfplumber>=0.10.3",  # for PDF text extraction
        "pdfminer.six>=20221105",  # for PDF parsing
    ],
    extras_require={
        "mlx": ["mlx-lm>=0.1.0"],
        "csm-mlx": ["csm-mlx @ git+https://github.com/senstella/csm-mlx.git"],
        "embeddings": [
            "faiss-cpu>=1.7.4",  # for embedding search
            "sentence-transformers>=2.2.2",  # for pretrained embedding models
            "torch>=2.1.0"  # for running embedding models on CUDA
        ]
    },
    author="Sabri, Avanika, and Dan",
    description="A package for running minion protocols with local and remote LLMs",
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "minions=minions_cli:main",
        ],
    },
)
