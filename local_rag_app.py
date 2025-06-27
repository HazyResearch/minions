#!/usr/bin/env python3
"""
Simple Local RAG Streamlit App

A streamlined interface for document selection and loading from markdown files.
This app focuses on simplicity and step-by-step functionality.
"""

import streamlit as st
import os
import time
from pathlib import Path
from typing import List, Tuple

# Import the core functionality from local_rag_document_search.py
from local_rag_document_search import load_markdown_files, search_documents, answer_question_with_ollama, MeetingAnswer
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minions import Minions, Document
from pydantic import BaseModel
import json

# Set page config
st.set_page_config(
    page_title="Local RAG + Minions App", 
    page_icon="📚",
    layout="wide"
)

class StructuredLocalOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None

def message_callback(role, message, is_final=True):
    """Show messages for both Local RAG and Minions protocols"""
    # Initialize placeholder_messages in session state if not present
    if "placeholder_messages" not in st.session_state:
        st.session_state.placeholder_messages = {}

    # Map supervisor -> Remote, worker -> Local
    chat_role = "Remote" if role == "supervisor" else "Local"
    
    if role == "supervisor":
        chat_role = "Remote"
        avatar_path = "assets/gru.jpg"
    else:
        chat_role = "Local"
        avatar_path = "assets/minion.png"

    # If we are not final, handle intermediate content or show working state
    if not is_final:
        # Clear any existing placeholder for this role
        if role in st.session_state.placeholder_messages:
            try:
                st.session_state.placeholder_messages[role].empty()
            except Exception as e:
                print(f"Warning: Could not clear existing placeholder for {role}: {e}")

        # Create a placeholder container and store it
        placeholder = st.empty()

        # Check if we have actual content to show
        has_content = message is not None and message != ""

        if has_content:
            # Show intermediate content
            with placeholder.chat_message(chat_role, avatar=avatar_path):
                # Display the intermediate content
                if role == "worker" and isinstance(message, list):
                    # For Minions protocol, messages are a list of jobs (intermediate results)
                    st.markdown("#### Intermediate results from minions...")
                    tasks = {}
                    for job in message:
                        task_id = job.manifest.task_id
                        if task_id not in tasks:
                            tasks[task_id] = {"task": job.manifest.task, "jobs": []}
                        tasks[task_id]["jobs"].append(job)

                    for task_id, task_info in tasks.items():
                        task_info["jobs"] = sorted(
                            task_info["jobs"], key=lambda x: x.manifest.job_id
                        )
                        include_jobs = [
                            job
                            for job in task_info["jobs"]
                            if job.output.answer
                            and job.output.answer.lower().strip() != "none"
                        ]

                        st.markdown(
                            f"_Processing: {len(task_info['jobs']) - len(include_jobs)} chunks pending, {len(include_jobs)} completed_"
                        )
                        # Show a sample of completed jobs
                        for job in include_jobs[:3]:  # Show first 3 completed jobs
                            st.markdown(
                                f"**✅ Job {job.manifest.job_id + 1} (Chunk {job.manifest.chunk_id + 1})**"
                            )
                            answer = job.output.answer.replace("$", "\\$")
                            st.markdown(f"Answer: {answer}")

                elif isinstance(message, dict):
                    if "content" in message and isinstance(
                        message["content"], (dict, str)
                    ):
                        try:
                            # Try to parse as JSON if it's a string
                            content = (
                                message["content"]
                                if isinstance(message["content"], dict)
                                else json.loads(message["content"])
                            )
                            st.json(content)
                        except json.JSONDecodeError:
                            st.write(message["content"])
                    else:
                        st.write(message)
                elif isinstance(message, str):
                    message = message.replace("$", "\\$")
                    st.markdown(message)
                else:
                    st.write(str(message))
        else:
            # Show working state when no content
            with placeholder.chat_message(chat_role, avatar=avatar_path):
                st.markdown("**Working...**")

        st.session_state.placeholder_messages[role] = placeholder
    else:
        # Handle final message - clear placeholder for this role
        if role in st.session_state.placeholder_messages:
            try:
                st.session_state.placeholder_messages[role].empty()
                del st.session_state.placeholder_messages[role]
            except Exception as e:
                print(f"Warning: Could not clear placeholder for {role}: {e}")
                if role in st.session_state.placeholder_messages:
                    del st.session_state.placeholder_messages[role]
        
        with st.chat_message(chat_role, avatar=avatar_path):
            if role == "worker" and isinstance(message, list):
                # For Minions protocol, messages are a list of jobs
                st.markdown("#### Here are the outputs from all the minions!")
                tasks = {}
                for job in message:
                    task_id = job.manifest.task_id
                    if task_id not in tasks:
                        tasks[task_id] = {"task": job.manifest.task, "jobs": []}
                    tasks[task_id]["jobs"].append(job)

                for task_id, task_info in tasks.items():
                    # first sort task_info[jobs] by job_id
                    task_info["jobs"] = sorted(
                        task_info["jobs"], key=lambda x: x.manifest.job_id
                    )
                    include_jobs = [
                        job
                        for job in task_info["jobs"]
                        if job.output.answer
                        and job.output.answer.lower().strip() != "none"
                    ]

                    st.markdown(
                        f"_Note: {len(task_info['jobs']) - len(include_jobs)} jobs did not have relevant information._"
                    )
                    st.markdown(f"**Jobs with relevant information:**")
                    # print all the relevant information
                    for job in include_jobs:
                        st.markdown(
                            f"**✅ Job {job.manifest.job_id + 1} (Chunk {job.manifest.chunk_id + 1})**"
                        )
                        answer = job.output.answer.replace("$", "\\$")
                        st.markdown(f"Answer: {answer}")

            elif isinstance(message, dict):
                if "content" in message and isinstance(message["content"], (dict, str)):
                    try:
                        # Try to parse as JSON if it's a string
                        content = (
                            message["content"]
                            if isinstance(message["content"], dict)
                            else json.loads(message["content"])
                        )
                        st.json(content)

                    except json.JSONDecodeError:
                        st.write(message["content"])
                else:
                    st.write(message)
            else:
                if isinstance(message, str):
                    message = message.replace("$", "\\$")
                st.markdown(message)

def main():
    st.title("🤖 Local QA over Zoom AI Companion Summaries")
    st.markdown("Ask questions about your meeting summaries!")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("⚙️ Protocol Selection")
        
        # Protocol selection
        protocol_choice = st.radio(
            "Choose processing method:",
            options=["Local RAG", "Minions"],
            index=0,
            help="Local RAG: Fast local-only processing with retrieval\nMinions: Hybrid local + cloud processing with intermediate steps"
        )
        
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
        
        # Show different settings based on protocol choice
        if protocol_choice == "Local RAG":
            # Settings in columns for Local RAG
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                model_name = st.selectbox(
                    "Model:",
                    options=["gemma3n:e2b", "gemma3n:e4b" ,"gemma3:4b", "gemma3:2b", "gemma3:9b", "llama3.2:3b", "llama3.2:1b", "qwen2.5:3b"],
                    index=0,  # Default to gemma3n:e2b
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
        else:
            # Simple button for Minions
            search_button = st.button("🔍 Ask", disabled=not query.strip(), use_container_width=True)
            # Set defaults for Minions
            model_name = "qwen2.5:3b"  # Default local model for Minions
            retriever_type = "bm25"    # Default retriever for Minions
        
        # Search and answer
        if search_button and query.strip():
            try:
                # Clear any previous chat messages
                if "placeholder_messages" in st.session_state:
                    st.session_state.placeholder_messages = {}
                
                if protocol_choice == "Local RAG":
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
                
                else:  # Minions protocol
                    with st.spinner("🤖 Processing with Minions protocol..."):
                        # Get documents from session state
                        file_contents = st.session_state.file_contents
                        file_paths = st.session_state.file_paths
                        
                        # Convert to Document objects for Minions
                        document_list = [
                            Document(content=content, filename=filename) 
                            for filename, content in zip(file_paths, file_contents)
                        ]
                        
                        # Setup Minions protocol clients
                        local_client = OllamaClient(
                            model_name=model_name,
                            temperature=0.0,
                            max_tokens=4096,
                            num_ctx=4096,
                            use_async=True,
                            structured_output_schema=StructuredLocalOutput
                        )
                        
                        remote_client = OpenAIClient(
                            model_name="gpt-4o-mini",
                            temperature=0.0,
                            max_tokens=4096
                        )
                        
                        protocol = Minions(local_client, remote_client, callback=message_callback)
                        
                        # Run Minions protocol
                        st.markdown("---")
                        st.markdown(f"**Q:** {query}")
                        
                        start_time = time.time()
                        output = protocol(
                            task=query,
                            doc_metadata="""Folder containing Zoom Meeting Summaries produced by Zoom's AI Companion.
Each file starts with the meeting title and date.
A short **Attendees:** line immediately follows the title/date, listing participant names (and sometimes roles).
Section headers marked with markdown H2 (##) appear in this order: Meeting Highlights, Detailed Summary, Action Items / Next Steps, and Full Transcript.
Highlights and action items are written as bullet points, while the transcript is time-stamped and lists the speaker name before each utterance.
When answering, you can reference information by citing these section headers or quoting specific bullet points/transcript lines.
Assume the requester is Sarah Chen, the Vice President of Marketing, who wants concise, actionable insights she can forward to her team.
""",
                            context=document_list,
                            max_rounds=5,
                            # use_retrieval="bm25"
                        )
                        end_time = time.time()
                        
                        # Display final answer
                        st.markdown("---")
                        st.markdown("## 🎯 Final Answer")
                        if isinstance(output, dict) and "final_answer" in output:
                            st.info(output["final_answer"])
                        else:
                            st.info(output)
                        
                        st.caption(f"⏱️ Completed in {end_time - start_time:.2f} seconds • Minions Protocol • {model_name} + gpt-4o-mini")
                        
                        # Store results in session state
                        st.session_state.last_query = query
                        st.session_state.last_minions_output = output
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    else:
        # Show instructions when no documents loaded
        st.info("👈 Please load documents from the sidebar to get started")
        
        st.markdown("""
        ### How to use:
        1. **Choose Protocol**: Select either "Local RAG" or "Minions" in the sidebar
        2. **Load Documents**: Enter a path to your markdown files in the sidebar
        3. **Ask Questions**: Type your question in the text box
        4. **Get Answers**: The AI will process your documents and provide answers
        
        ### Protocol Differences:
        - **Local RAG**: Fast, local-only processing with document retrieval
        - **Minions**: Hybrid local + cloud processing with intermediate step visualization
        
        ### Example questions:
        - "How many FTEs were approved?"
        - "What was discussed about the marketing budget?"
        - "Which vendor was selected for the website redesign?"
        """)

if __name__ == "__main__":
    main()