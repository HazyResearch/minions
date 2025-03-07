from setuptools import setup, find_packages

setup(
    name="minions",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ollama",  # for local LLM
        "streamlit",  # for the UI
        "openai",  # for OpenAI client
        "anthropic",  # for Anthropic client
        "together",  # for Together client
        "groq",  # for Groq client
        "requests",  # for API calls
        "tiktoken",  # for token counting
        "pymupdf",  # for PDF processing
        "st-theme",
        "mcp",
        "spacy",  # for PII extraction, worked on python 3.11 and not 3.13
        "rank_bm25",  # for smart retrieval
        "PyMuPDF",  # for PDF handling
    ],
    extras_require={
        "mlx": ["mlx-lm"],
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
