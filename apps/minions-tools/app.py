import streamlit as st
import json
import time
from typing import Dict, Any, List, Tuple
import sys
import os

# Add the parent directory to the path to import minions modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import utilities
from utils import (
    format_execution_time, 
    format_response_for_display,
    display_prerequisite_status,
    display_sample_tasks,
    save_comparison_results,
    create_method_comparison_chart
)

from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.utils.minion_mcp import _make_mcp_minion

# Configure Streamlit page
st.set_page_config(
    page_title="Minions Tools Comparison",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .comparison-container {
        border: 2px solid #e1e5e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .method-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .metrics-container {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .response-text {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Source Code Pro', monospace;
    }
    
    /* Enhanced styling for response containers */
    div[data-testid="stContainer"] > div {
        background: #f8f9fb;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Specific styling for bordered containers */
    div[data-testid="stContainer"][style*="border"] {
        background: linear-gradient(135deg, #f8f9fb 0%, #ffffff 100%);
        border: 2px solid #e1e5e9 !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def mcp_callback(role, message, is_final=True):
    """Callback function to display intermediate MCP outputs"""
    try:
        # Skip None messages and empty messages
        if message is None or (isinstance(message, str) and not message.strip()):
            return
            
        # Initialize session state for MCP logs if not exists
        if 'mcp_logs' not in st.session_state:
            st.session_state.mcp_logs = []
        if 'mcp_turn_counter' not in st.session_state:
            st.session_state.mcp_turn_counter = 0
        
        # Create a formatted log entry
        timestamp = time.strftime("%H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "role": role,
            "message": message,
            "is_final": is_final
        }
        
        # Add to session state logs
        st.session_state.mcp_logs.append(log_entry)


        
        # Display messages in the stream container
        if hasattr(st, '_mcp_stream_container'):
            with st._mcp_stream_container:
                # Display messages with cleaner formatting based on callback data
                if role.lower() == 'supervisor':
                    if isinstance(message, dict):
                        # if 'role' in message and 'content' in message:
                        #     # Standard message format
                        #     content = message['content']
                        #     if content.strip():
                        #         st.markdown("---")
                        #         st.markdown(f"### 🎯 **SUPERVISOR** `{timestamp}`")
                        #         st.info(content[:500] + "..." if len(content) > 500 else content)
                        # else:
                            # Parse supervisor JSON response structure
                        try:    
                            message = json.loads(message["content"])
                        except:
                            message = message["content"]
                        st.markdown("---")
                        st.markdown(f"### 🎯 **SUPERVISOR** `{timestamp}`")
                        
                        # Extract and display message
                        if 'message' in message:
                            st.markdown("**📝 Message to Worker:**")
                            st.info(message['message'])
                        
                        # Extract and display decision
                        if 'decision' in message:
                            decision = message['decision']
                            decision_emoji = {
                                'continue': '⏭️',
                                'final_answer': '✅', 
                                'request_additional_info': '❓',
                                'terminate': '🛑'
                            }.get(decision, '🔄')
                            st.markdown(f"**{decision_emoji} Decision:** `{decision}`")
                        
                        # Extract and display MCP tool calls
                        if 'mcp_tool_calls' in message and message['mcp_tool_calls']:
                            st.markdown("**🔧 MCP Tool Calls:**")
                            for i, tool_call in enumerate(message['mcp_tool_calls'], 1):
                                with st.expander(f"Tool Call {i}: {tool_call.get('tool_name', 'Unknown')}", expanded=False):
                                    st.code(json.dumps(tool_call, indent=2), language="json")
                        
                        # Display any other fields
                        other_fields = {k: v for k, v in message.items() 
                                        if k not in ['message', 'decision', 'mcp_tool_calls']}
                        if other_fields:
                            st.markdown("**📋 Additional Data:**")
                            st.json(other_fields)
                    elif isinstance(message, str):
                        if message.strip():
                            st.markdown(f"### 🎯 **SUPERVISOR** `{timestamp}`")
                            st.info(message[:500] + "..." if len(message) > 500 else message)
                
                elif role.lower() == 'worker':
                    if isinstance(message, dict):
                        if 'role' in message and 'content' in message:
                            content = message['content']
                            if content.strip():
                                st.markdown(f"### 🔨 **WORKER** `{timestamp}`")
                                st.success(content[:500] + "..." if len(content) > 500 else content)
                        else:
                            # Handle other dict structures
                            st.markdown(f"### 🔨 **WORKER** `{timestamp}`")
                            st.json(message)
                    elif isinstance(message, str):
                        if message.strip():
                            st.markdown(f"### 🔨 **WORKER** `{timestamp}`")
                            st.success(message[:500] + "..." if len(message) > 500 else message)
                
                # Update counter for activity indication
                if hasattr(st, '_mcp_scroll_counter'):
                    st._mcp_scroll_counter += 1
                else:
                    st._mcp_scroll_counter = 1
                
                # Force a small update to trigger refresh
                if hasattr(st, '_mcp_status_placeholder'):
                    st._mcp_status_placeholder.caption(f"Live updates: {st._mcp_scroll_counter}")
                    
    except Exception as e:
        # Silently ignore callback errors to avoid disrupting the flow
        pass

def initialize_clients():
    """Initialize the clients for both methods"""
    try:
        from pydantic import BaseModel

        # Initialize clients
        class StructuredLocalOutput(BaseModel):
            explanation: str
            citation: str | None
            answer: str | None

        # Method 1: Direct Ollama (no MCP)
        ollama_direct = OllamaClient(
            model_name="llama3.2:3b",
            temperature=0.0,
            structured_output_schema=StructuredLocalOutput,
        )
        
        # Method 2: MCP Minion (using the utility function)
        minion_mcp = _make_mcp_minion("filesystem", callback=mcp_callback)
        
        return {
            "ollama_direct": ollama_direct,
            "minion_mcp": minion_mcp,
        }
    except Exception as e:
        st.error(f"Failed to initialize clients: {str(e)}")
        return None

def run_ollama_direct_method(client, task: str) -> Tuple[str, Dict[str, Any]]:
    """Run the direct Ollama method (no MCP)"""
    start_time = time.time()
    
    messages = [
        {
            "role": "user",
            "content": task
        }
    ]
    
    try:
        response, usage, done_reason = client.chat(messages)
        end_time = time.time()
        
        metrics = {
            "execution_time": end_time - start_time,
            "tokens_used": usage.total_tokens if hasattr(usage, 'total_tokens') else 0,
            "prompt_tokens": usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0,
            "completion_tokens": usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0,
            "done_reason": done_reason[0] if done_reason else "unknown"
        }
        
        return response[0] if response else "No response", metrics
        
    except Exception as e:
        end_time = time.time()
        return f"Error: {str(e)}", {
            "execution_time": end_time - start_time,
            "error": True
        }

def run_minion_mcp_method(minion_client, task: str) -> Tuple[str, Dict[str, Any]]:
    """Run the MCP Minion method using only callbacks"""
    start_time = time.time()
    
    try:
        # Clear previous MCP logs
        if 'mcp_logs' in st.session_state:
            st.session_state.mcp_logs = []
        
        # Create containers for streaming output
        st.markdown("### 🔄 MCP Execution Stream")
        st._mcp_stream_container = st.container()
        st._mcp_status_placeholder = st.empty()
        
        with st._mcp_stream_container:
            st.markdown("**🚀 Starting MCP Minion execution...**")
        
        # Execute the minion task - the callback will handle all streaming
        result = minion_client(
            task=task,
            context=[],
            max_rounds=5
        )
        
        end_time = time.time()
        
        # Clear the status placeholder
        if hasattr(st, '_mcp_status_placeholder'):
            st._mcp_status_placeholder.empty()
        
        # Display completion message
        with st._mcp_stream_container:
            st.success("✅ MCP Minion execution completed!")
        
        metrics = {
            "execution_time": end_time - start_time,
            "method": "minion_mcp",
            "log_entries": len(st.session_state.get('mcp_logs', []))
        }
        
        # Extract final answer based on result structure
        if isinstance(result, dict):
            final_answer = result.get("final_answer", str(result))
        else:
            final_answer = str(result)
            
        return final_answer, metrics
        
    except Exception as e:
        end_time = time.time()
        # Clear the status placeholder on error
        if hasattr(st, '_mcp_status_placeholder'):
            st._mcp_status_placeholder.empty()
        
        # Display error in stream
        if hasattr(st, '_mcp_stream_container'):
            with st._mcp_stream_container:
                st.error(f"❌ Error: {str(e)}")
        
        return f"Error: {str(e)}", {
            "execution_time": end_time - start_time,
            "error": True
        }

def display_metrics(metrics: Dict[str, Any], method_name: str):
    """Display metrics in a formatted way"""
    with st.container():
        st.markdown(f"### 📊 {method_name} Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Execution Time", 
                format_execution_time(metrics.get('execution_time', 0))
            )
        
        with col2:
            if 'tokens_used' in metrics:
                st.metric("Total Tokens", metrics['tokens_used'])
            elif 'rounds_used' in metrics:
                st.metric("Rounds Used", metrics['rounds_used'])
            else:
                st.metric("Method Type", method_name.split()[0])
        
        with col3:
            if 'done_reason' in metrics:
                st.metric("Done Reason", metrics['done_reason'])
            elif 'method' in metrics:
                st.metric("Method", metrics['method'])
            else:
                st.metric("Status", "Success" if not metrics.get('error') else "Error")
      
        
        if metrics.get('error'):
            st.error("⚠️ Execution resulted in an error")

def main():
    # Main header
    st.markdown('<h1 class="main-header">Minions Tools Comparison</h1>', unsafe_allow_html=True)
    
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Display system status
        display_prerequisite_status()
        
        # Task input
        default_task = st.session_state.get('selected_task', 
            "Read the mcp config file in this directory and explain what MCP servers are configured.")
        
        task_input = st.text_area(
            "Enter your task:",
            value=default_task,
            height=100
        )
        
        # Sample tasks
        display_sample_tasks()
        
        # Method selection
        st.subheader("Select Methods to Compare")
        compare_ollama_direct = st.checkbox("Direct Ollama (w/MCP Tools)", value=True)
        compare_minion_mcp = st.checkbox("Minion MCP (with MCP Tools)", value=True)
        
        # Execution button
        run_comparison = st.button("🚀 Run Comparison", type="primary")
        
        # Results export
        if 'comparison_results' in st.session_state:
            st.subheader("📊 Export Results")
            if st.button("💾 Save Results (JSON)"):
                filename = save_comparison_results(st.session_state.comparison_results)
                if filename:
                    st.success(f"Results saved to {filename}")
        
        # Model info
        st.subheader("ℹ️ Model Information")
        st.info("""
        **Direct Method**: llama3.2:3b (Ollama)
        **MCP Method**: llama3.2:3b (Ollama) + gpt-4o (OpenAI) 
        **MCP Server**: filesystem
        """)
    
    # Initialize clients
    if 'clients' not in st.session_state:
        with st.spinner("Initializing clients..."):
            st.session_state.clients = initialize_clients()
    
    if st.session_state.clients is None:
        st.error("Failed to initialize clients. Please check your configuration.")
        return
    
    # Main content area
    if run_comparison and task_input.strip():
        st.markdown("## 🔄 Running Comparison...")
        
        results = {}
        
        # Create columns for side-by-side comparison
        methods_to_run = []
        if compare_ollama_direct:
            methods_to_run.append("ollama_direct")
        if compare_minion_mcp:
            methods_to_run.append("minion_mcp")
        
        if not methods_to_run:
            st.warning("Please select at least one method to compare.")
            return
        
        # Run comparisons
        for method in methods_to_run:
            with st.spinner(f"Running {method.replace('_', ' ').title()}..."):
                if method == "ollama_direct":
                    response, metrics = run_ollama_direct_method(
                        st.session_state.clients["ollama_direct"], 
                        task_input
                    )
                    results[method] = {"response": response, "metrics": metrics}
                
                elif method == "minion_mcp":
                    response, metrics = run_minion_mcp_method(
                        st.session_state.clients["minion_mcp"], 
                        task_input
                    )
                    results[method] = {"response": response, "metrics": metrics}
        
        # Save results to session state
        st.session_state.comparison_results = results
        
        # Display results
        st.markdown("## 📋 Comparison Results")
        
        # Create tabs for each method
        if len(results) > 0:
            tabs = st.tabs([method.replace('_', ' ').title() for method in results.keys()])
            
            for i, (method, result) in enumerate(results.items()):
                with tabs[i]:
                    # Display metrics
                    display_metrics(result["metrics"], method.replace('_', ' ').title())
                    
                    # Display response  
                    st.markdown("### 💬 Response")
                    
                    response_text = result["response"]
                    
                    # Add display options
                    display_option = st.radio(
                        "Display format:",
                        ["Formatted Text", "Raw Text", "Code Block"],
                        horizontal=True,
                        key=f"display_{method}_{i}"
                    )
                    
                    # Create a bordered container for the response
                    response_container = st.container(border=True)
                    with response_container:
                        if display_option == "Formatted Text":
                            # Use markdown for formatted display
                            if len(response_text) > 2000:
                                with st.expander("Click to view full response", expanded=True):
                                    st.markdown(response_text)
                            else:
                                st.markdown(response_text)
                        
                        elif display_option == "Raw Text":
                            # Use text area for raw display
                            height = 300 if len(response_text) > 2000 else 200
                            st.text_area(
                                "Raw Response",
                                value=response_text,
                                height=height,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                        
                        else:  # Code Block
                            # Use code block for structured display
                            if len(response_text) > 2000:
                                with st.expander("Click to view full response", expanded=True):
                                    st.code(response_text, language="text")
                            else:
                                st.code(response_text, language="text")
                    
                    # Display MCP execution logs for minion_mcp method
                    if method == "minion_mcp" and 'mcp_logs' in st.session_state and st.session_state.mcp_logs:
                        st.markdown("### 📋 MCP Execution Logs")
                        
                        with st.expander("View detailed MCP conversation log", expanded=False):
                            current_turn = 0
                            for idx, log_entry in enumerate(st.session_state.mcp_logs):
                                timestamp = log_entry.get('timestamp', '')
                                role = log_entry.get('role', '')
                                message = log_entry.get('message', '')
                                is_final = log_entry.get('is_final', True)
                                
                                # Skip None or empty messages
                                if message is None or (isinstance(message, str) and not message.strip()):
                                    continue
                                
                                # Display with clean turn-based formatting
                                if role.lower() == 'supervisor':
                                    if isinstance(message, dict):
                                        if 'content' in message:
                                            # Standard message format
                                            content = message['content']
                                            if content.strip():
                                                st.markdown(f"#### 🎯 Supervisor Turn {current_turn + 1} - {timestamp}")
                                                st.info(content[:300] + "..." if len(content) > 300 else content)
                                                current_turn += 1
                                        else:
                                            # Parse supervisor JSON response structure
                                            st.markdown(f"#### 🎯 Supervisor Turn {current_turn + 1} - {timestamp}")
                                            current_turn += 1
                                            
                                            # Extract and display message
                                            if 'message' in message:
                                                st.markdown("**📝 Message to Worker:**")
                                                msg_text = message['message']
                                                st.info(msg_text[:200] + "..." if len(msg_text) > 200 else msg_text)
                                            
                                            # Extract and display decision
                                            if 'decision' in message:
                                                decision = message['decision']
                                                decision_emoji = {
                                                    'continue': '⏭️',
                                                    'final_answer': '✅', 
                                                    'request_additional_info': '❓',
                                                    'terminate': '🛑'
                                                }.get(decision, '🔄')
                                                st.markdown(f"**{decision_emoji} Decision:** `{decision}`")
                                            
                                            # Extract and display MCP tool calls
                                            if 'mcp_tool_calls' in message and message['mcp_tool_calls']:
                                                st.markdown("**🔧 MCP Tool Calls:**")
                                                for i, tool_call in enumerate(message['mcp_tool_calls'], 1):
                                                    tool_name = tool_call.get('tool_name', 'Unknown')
                                                    st.markdown(f"- **{tool_name}**")
                                                    if 'parameters' in tool_call:
                                                        st.code(json.dumps(tool_call['parameters'], indent=2), language="json")
                                    elif isinstance(message, str) and message.strip():
                                        st.markdown(f"#### 🎯 Supervisor Turn {current_turn + 1} - {timestamp}")
                                        st.info(message[:300] + "..." if len(message) > 300 else message)
                                        current_turn += 1
                                
                                elif role.lower() == 'worker':
                                    if isinstance(message, dict):
                                        if 'content' in message:
                                            content = message['content']
                                            if content.strip():
                                                st.markdown(f"#### 🔨 Worker Response - {timestamp}")
                                                st.success(content[:300] + "..." if len(content) > 300 else content)
                                        else:
                                            st.markdown(f"#### 🔨 Worker - {timestamp}")
                                            st.json(message)
                                    elif isinstance(message, str) and message.strip():
                                        st.markdown(f"#### 🔨 Worker Response - {timestamp}")
                                        st.success(message[:300] + "..." if len(message) > 300 else message)
                                
                                # Add separator between major turns
                                if idx < len(st.session_state.mcp_logs) - 1:
                                    next_entry = st.session_state.mcp_logs[idx + 1]
                                    if (next_entry.get('role', '').lower() == 'supervisor' and 
                                        role.lower() == 'worker'):
                                        st.markdown("---")
        
        # # Enhanced comparison summary
        # if len(results) > 1:
        #     st.markdown("## 📊 Performance Analysis")
            
        #     # Create comparison charts
        #     try:
        #         create_method_comparison_chart(results)
        #     except Exception as e:
        #         st.warning(f"Could not create charts: {str(e)}")
            
        #     # Summary table
        #     st.markdown("### Summary Table")
        #     summary_data = []
        #     for method, result in results.items():
        #         from utils import calculate_performance_score
        #         summary_data.append({
        #             "Method": method.replace('_', ' ').title(),
        #             "Execution Time": format_execution_time(result['metrics'].get('execution_time', 0)),
        #             "Status": "Error" if result['metrics'].get('error') else "Success",
        #             "Response Length": len(result["response"])
        #         })
            
        #     st.table(summary_data)
    
    elif run_comparison and not task_input.strip():
        st.warning("Please enter a task to compare.")
    
   

if __name__ == "__main__":
    main() 