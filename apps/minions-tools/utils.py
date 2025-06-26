"""
Utility functions for the Minions Tools Comparison Streamlit app.
"""

import json
import os
import time
from typing import Dict, Any, List, Optional
import streamlit as st


def format_execution_time(seconds: float) -> str:
    """Format execution time in a human-readable way."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def format_response_for_display(response: str, max_length: int = 5000) -> str:
    """Format response text for display in Streamlit."""
    if len(response) > max_length:
        return response[:max_length] + "\n\n... (truncated)"
    return response


def save_comparison_results(results: Dict[str, Any], filename: str = None) -> str:
    """Save comparison results to a JSON file."""
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_results_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        return filename
    except Exception as e:
        st.error(f"Failed to save results: {str(e)}")
        return None


def load_comparison_results(filename: str) -> Optional[Dict[str, Any]]:
    """Load comparison results from a JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load results: {str(e)}")
        return None


def check_prerequisites() -> Dict[str, bool]:
    """Check if all prerequisites are met for running the app."""
    checks = {
        "mcp_config": False,
        "ollama_available": False,
        "openai_key": False
    }
    
    # Check MCP config
    mcp_config_paths = ["mcp.json", "../mcp.json", "../../mcp.json"]
    for path in mcp_config_paths:
        if os.path.exists(path):
            checks["mcp_config"] = True
            break
    
    # Check Ollama (simplified check)
    try:
        import ollama
        models = ollama.list()
        if any("llama3.2" in model.model for model in models.get("models", [])):
            checks["ollama_available"] = True
    except:
        pass
    
    # Check OpenAI key
    checks["openai_key"] = "OPENAI_API_KEY" in os.environ
    
    return checks


def display_prerequisite_status():
    """Display the status of prerequisites in the sidebar."""
    st.sidebar.markdown("### 🔍 System Status")
    
    checks = check_prerequisites()
    
    for check_name, status in checks.items():
        if check_name == "mcp_config":
            label = "MCP Config"
        elif check_name == "ollama_available":
            label = "Ollama + Model"
        elif check_name == "openai_key":
            label = "OpenAI API Key"
        
        if status:
            st.sidebar.success(f"✅ {label}")
        else:
            st.sidebar.error(f"❌ {label}")
    
    if not all(checks.values()):
        st.sidebar.warning("⚠️ Some prerequisites are missing. The app may not function correctly.")


def create_sample_tasks() -> List[Dict[str, str]]:
    """Create a list of sample tasks for testing."""
    return [
        {
            "name": "Configuration File Analysis",
            "description": "Read and analyze configuration files",
            "task": "Read the mcp config file in this directory and explain what MCP servers are configured."
        },
        {
            "name": "Directory Structure Analysis",
            "description": "Analyze and summarize a directory structure",
            "task": "Can you show me the directory structure of the examples folder in the current directory and then summarize what you find?"
        },
        {
            "name": "File Search",
            "description": "Find specific types of files",
            "task": "Find all Python files in the current directory and list their names with brief descriptions."
        },
    ]


def display_sample_tasks():
    """Display sample tasks in the sidebar."""
    st.sidebar.markdown("### 📝 Sample Tasks")
    
    sample_tasks = create_sample_tasks()
    
    for task in sample_tasks:
        with st.sidebar.expander(task["name"]):
            st.write(f"**Description**: {task['description']}")
            st.write(f"**Task**: {task['task']}")
            if st.button(f"Use this task", key=f"task_{task['name']}"):
                st.session_state.selected_task = task["task"]


def calculate_performance_score(metrics: Dict[str, Any]) -> float:
    """Calculate a simple performance score based on metrics."""
    base_score = 100.0
    
    # Deduct points for execution time (more time = lower score)
    execution_time = metrics.get("execution_time", 0)
    time_penalty = min(execution_time * 10, 50)  # Max 50 point penalty
    
    # Deduct points for errors
    error_penalty = 50 if metrics.get("error", False) else 0
    
    # Add points for successful tool usage
    if metrics.get("done_reason") == "stop" or metrics.get("method"):
        success_bonus = 10
    else:
        success_bonus = 0
    
    score = base_score - time_penalty - error_penalty + success_bonus
    return max(0, min(100, score))  # Clamp between 0 and 100


def export_results_to_csv(results: Dict[str, Any]) -> str:
    """Export comparison results to a CSV file."""
    import pandas as pd
    
    # Prepare data for CSV
    csv_data = []
    for method, result in results.items():
        row = {
            "Method": method.replace('_', ' ').title(),
            "Execution_Time_s": result["metrics"].get("execution_time", 0),
            "Success": not result["metrics"].get("error", False),
            "Response_Length": len(result["response"]),
            "Performance_Score": calculate_performance_score(result["metrics"])
        }
        
        # Add method-specific metrics
        if "tokens_used" in result["metrics"]:
            row["Tokens_Used"] = result["metrics"]["tokens_used"]
        if "rounds_used" in result["metrics"]:
            row["Rounds_Used"] = result["metrics"]["rounds_used"]
        
        csv_data.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_results_{timestamp}.csv"
    
    try:
        df.to_csv(filename, index=False)
        return filename
    except Exception as e:
        st.error(f"Failed to export CSV: {str(e)}")
        return None


def create_method_comparison_chart(results: Dict[str, Any]):
    """Create a comparison chart for the methods."""
    import pandas as pd
    import plotly.express as px
    
    # Prepare data for visualization
    chart_data = []
    for method, result in results.items():
        chart_data.append({
            "Method": method.replace('_', ' ').title(),
            "Execution Time (s)": result["metrics"].get("execution_time", 0),
            "Performance Score": calculate_performance_score(result["metrics"]),
            "Status": "Success" if not result["metrics"].get("error", False) else "Error"
        })
    
    df = pd.DataFrame(chart_data)
    
    # Create execution time comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Execution Time Comparison")
        fig_time = px.bar(
            df, 
            x="Method", 
            y="Execution Time (s)",
            color="Status",
            title="Execution Time by Method"
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        st.subheader("Performance Score Comparison")
        fig_score = px.bar(
            df,
            x="Method",
            y="Performance Score", 
            color="Status",
            title="Performance Score by Method"
        )
        st.plotly_chart(fig_score, use_container_width=True) 