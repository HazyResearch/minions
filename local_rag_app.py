#!/usr/bin/env python3
"""
Simple Local RAG Streamlit App

A streamlined interface for document selection and loading from markdown files.
This app focuses on simplicity and step-by-step functionality.
"""

import streamlit as st
import os
from pathlib import Path
from typing import List, Tuple

# Import the core functionality from local_rag_document_search.py
from local_rag_document_search import load_markdown_files, search_documents, answer_question_with_ollama, MeetingAnswer
from minions.clients.ollama import OllamaClient

# Set page config
st.set_page_config(
    page_title="Local RAG App", 
    page_icon="📚",
    layout="wide"
)

def main():
    st.title("🤖 Local RAG Assistant")
    st.markdown("Ask questions about your documents using local AI models")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Setup")
        
        # Default path
        default_path = "data/meeting_summaries"
        
        # Path input with default
        docs_path = st.text_input(
            "Documents Path:", 
            value=default_path,
            help="Enter the path to the directory containing markdown files"
        )
        
        # Load documents button
        if st.button("Load Documents"):
            try:
                # Load markdown files
                with st.spinner("Loading markdown files..."):
                    file_contents, file_paths = load_markdown_files(docs_path)
                
                # Store in session state
                st.session_state.file_contents = file_contents
                st.session_state.file_paths = file_paths
                st.session_state.docs_loaded = True
                
                st.success(f"✅ Loaded {len(file_contents)} files")
                
            except FileNotFoundError:
                st.error(f"❌ Directory not found: {docs_path}")
            except ValueError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        # Show loaded documents status
        if hasattr(st.session_state, 'docs_loaded') and st.session_state.docs_loaded:
            st.info(f"📚 {len(st.session_state.file_paths)} documents loaded")
            
            # Optional: Show document list in expander
            with st.expander("View Documents"):
                for i, file_path in enumerate(st.session_state.file_paths):
                    filename = os.path.basename(file_path)
                    content = st.session_state.file_contents[i]
                    
                    st.markdown(f"**📄 {filename}**")
                    st.markdown(content[:500] + "..." if len(content) > 500 else content)
                    if i < len(st.session_state.file_paths) - 1:
                        st.markdown("---")
    
    # Main interface - only show if documents are loaded
    if hasattr(st.session_state, 'docs_loaded') and st.session_state.docs_loaded:
        
        # Query input (full width)
        query = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., How many additional FTEs were approved in the executive team call?",
            help="Enter your question here"
        )
        
        # Settings in columns
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            model_name = st.selectbox(
                "Model:",
                options=["gemma3:4b", "gemma3:2b", "gemma3:9b", "llama3.2:3b", "llama3.2:1b", "qwen2.5:3b"],
                index=0,  # Default to gemma3:4b
                help="Select the local Ollama model"
            )
        
        with col2:
            retriever_type = st.selectbox(
                "Retriever:",
                options=["mlx", "embedding", "bm25"],
                index=0,  # Default to MLX
                help="Select retrieval method"
            )
        
        with col3:
            st.write("")  # Spacing
            search_button = st.button("🔍 Ask", disabled=not query.strip(), use_container_width=True)
        
        # Search and answer
        if search_button and query.strip():
            try:
                with st.spinner("🔍 Processing your question..."):
                    # Get documents from session state
                    file_contents = st.session_state.file_contents
                    file_paths = st.session_state.file_paths
                    
                    # Retrieval
                    search_results = search_documents(
                        query=query,
                        documents=file_contents,
                        file_paths=file_paths,
                        k=3,
                        retriever_type=retriever_type
                    )
                    
                    # Extract retrieved documents
                    retrieved_docs = []
                    retrieved_paths = []
                    
                    for file_path, preview, score in search_results:
                        # Find full document content for this result
                        for j, path in enumerate(file_paths):
                            if path == file_path:
                                retrieved_docs.append(file_contents[j])
                                retrieved_paths.append(path)
                                break
                    
                    # Answer generation with Ollama
                    answer_client = OllamaClient(
                        model_name=model_name,
                        temperature=0.0,
                        max_tokens=500,
                        num_ctx=4096,
                        structured_output_schema=MeetingAnswer,
                        use_async=False
                    )
                    
                    # Get answer
                    answer = answer_question_with_ollama(
                        user_query=query,
                        documents=retrieved_docs,
                        file_paths=retrieved_paths,
                        ollama_client=answer_client
                    )
                    
                    # Display the final answer
                    st.markdown("---")
                    
                    # Answer section
                    st.markdown(f"**Q:** {query}")
                    st.markdown(f"**A:** {answer.answer}")
                    
                    # Citation
                    if answer.citation and answer.citation != "N/A":
                        st.markdown("**Source:**")
                        st.info(answer.citation)
                    
                    # Metadata in small text
                    confidence_emoji = {
                        "high": "🟢",
                        "medium": "🟡", 
                        "low": "🔴"
                    }.get(answer.confidence.lower(), "⚪")
                    
                    st.caption(f"{confidence_emoji} {answer.confidence.title()} confidence • {model_name} • {retriever_type.upper()} retrieval")
                    
                    # Optional: Show retrieved documents in expander
                    with st.expander("🔍 Retrieved Documents (for debugging)"):
                        for i, (file_path, preview, score) in enumerate(search_results, 1):
                            filename = os.path.basename(file_path)
                            st.write(f"**{i}. {filename}** (Score: {score:.3f})")
                            st.write(f"{preview}")
                            if i < len(search_results):
                                st.write("---")
                    
                    # Store results in session state
                    st.session_state.last_query = query
                    st.session_state.last_answer = answer
                    st.session_state.last_results = search_results
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    else:
        # Show instructions when no documents loaded
        st.info("👈 Please load documents from the sidebar to get started")
        
        st.markdown("""
        ### How to use:
        1. **Load Documents**: Enter a path to your markdown files in the sidebar
        2. **Ask Questions**: Type your question in the text box
        3. **Get Answers**: The AI will search your documents and provide answers with citations
        
        ### Example questions:
        - "How many FTEs were approved?"
        - "What was discussed about the marketing budget?"
        - "Which vendor was selected for the website redesign?"
        """)

if __name__ == "__main__":
    main()