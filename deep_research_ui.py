import streamlit as st
import os
import time
import json
from typing import Optional, Dict, Any


def render_deep_research_ui(minions_instance=None):
    """
    Render the Deep Research UI component.
    
    Args:
        minions_instance: An initialized DeepResearchMinions instance
    """
    st.title("Deep Research")
    
    # API Keys section - directly in main area, not sidebar
    st.subheader("API Keys")
    
    api_keys_col1, api_keys_col2 = st.columns(2)
    
    with api_keys_col1:
        # Firecrawl API key
        firecrawl_env_key = os.getenv("FIRECRAWL_API_KEY", "")
        firecrawl_user_key = st.text_input(
            "Firecrawl API Key",
            type="password",
            value=firecrawl_env_key,
            key="firecrawl_key_dr_ui",
            help="For extracting content from web pages"
        )
        firecrawl_api_key = firecrawl_user_key if firecrawl_user_key else firecrawl_env_key
        
        if firecrawl_api_key:
            st.success("✓ Firecrawl API key set")
        else:
            st.error("✗ Firecrawl API key required")
    
    with api_keys_col2:
        # SERP API key
        serpapi_env_key = os.getenv("SERPAPI_API_KEY", "")
        serpapi_user_key = st.text_input(
            "SERP API Key",
            type="password",
            value=serpapi_env_key,
            key="serpapi_key_dr_ui",
            help="For performing web searches"
        )
        serpapi_key = serpapi_user_key if serpapi_user_key else serpapi_env_key
        
        if serpapi_key:
            st.success("✓ SERP API key set")
        else:
            st.error("✗ SERP API key required")
    
    # Research Settings
    with st.expander("Research Settings", expanded=False):
        max_rounds = st.slider(
            "Maximum Search Attempts", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="Maximum number of search attempts to perform"
        )
        
        max_sources = st.slider(
            "Sources per Search", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="Maximum number of web sources to process per search"
        )
    
    # User query input
    user_query = st.text_area(
        "Enter your research question",
        placeholder="Explain how Anthropic's MCP works?",
        height=70
    )
    
    # Submit button
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_button = st.button("Research", type="primary", use_container_width=True)
    
    if submit_button and user_query:
        # Validate API keys
        if not firecrawl_api_key or not serpapi_key:
            st.error("Both Firecrawl and SERP API keys are required to continue.")
            return
        
        # Set API keys in session state for access by the main app
        st.session_state.firecrawl_api_key = firecrawl_api_key
        st.session_state.serpapi_api_key = serpapi_key
        
        # Research progress
        with st.status("Researching your query...", expanded=True) as status:
            start_time = time.time()
            
            # Track search queries and found information
            search_queries = []
            information_sections = []
            
            # Create containers for updating search progress
            search_container = st.empty()
            info_container = st.empty()
            
            try:
                # If minions_instance is provided, use it directly
                if minions_instance:
                    # Configure the instance with our settings
                    minions_instance.max_rounds = max_rounds
                    minions_instance.max_sources_per_round = max_sources
                    
                    # Execute the research
                    result, visited_urls = minions_instance(
                        query=user_query,
                        firecrawl_api_key=firecrawl_api_key,
                        serpapi_key=serpapi_key
                    )
                else:
                    # Placeholder when running UI independently for testing
                    st.warning("DeepResearchMinions instance not provided. This is a UI preview only.")
                    result = "This is a placeholder answer. In actual usage, the research results would appear here."
                    visited_urls = ["https://example.com/1", "https://example.com/2"]
            except Exception as e:
                st.error(f"Research failed: {str(e)}")
                status.update(label="Research failed", state="error")
                return
            
            elapsed_time = time.time() - start_time
            status.update(label=f"Research completed in {elapsed_time:.2f} seconds!", state="complete")
        
        # Display results
        st.subheader("Research Results")
        st.markdown(result)  # Simply display the result string directly
        st.markdown("### Sources Referenced")
        for i, url in enumerate(visited_urls, 1):
            st.markdown(f"{i}. [{url}]({url})")