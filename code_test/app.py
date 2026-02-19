import streamlit as st
import pandas as pd
import os
import json
import tempfile
import time
import sys
from io import StringIO
import contextlib
import traceback

# Import the BuggyCodeProcessor from the improved module
# Make sure to have the improved processor file in the same directory
from processor import BuggyCodeProcessor

# Set page configuration
st.set_page_config(
    page_title="Buggy Code Processor",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 5px;
        padding-bottom: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
        border-bottom: 2px solid #4b91ff;
    }
    .output-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e9ecef;
        height: 300px;
        overflow-y: auto;
        font-family: monospace;
    }
    .function-card {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e9ecef;
        margin-bottom: 10px;
    }
    .success {
        background-color: #d4edda;
        border-color: #c3e6cb;
    }
    .fail {
        background-color: #f8d7da;
        border-color: #f5c6cb;
    }
    .info {
        background-color: #e6f2ff;
        border-color: #b8daff;
    }
    .iteration-card {
        padding: 5px;
        border-radius: 3px;
        margin-top: 5px;
        font-size: 0.9em;
    }
    .progress-container {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .title-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# Custom stdout capture class to display output in real-time
class StreamlitIOCapture:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.captured_output = StringIO()
        self.current_text = ""
        
    def write(self, text):
        self.captured_output.write(text)
        self.current_text += text
        self.placeholder.markdown(f"```\n{self.current_text}\n```")
        
    def flush(self):
        pass

# Context manager for capturing output
@contextlib.contextmanager
def capture_output(placeholder):
    io_capture = StreamlitIOCapture(placeholder)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io_capture
    sys.stderr = io_capture
    try:
        yield io_capture
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def main():
    st.title("🐛 Buggy Code Processor")
    st.markdown("""
    Upload a CSV file containing buggy code and apply the iterative fixing process.
    The app will extract functions, generate test cases, and attempt to fix any bugs.
    """)
    
    # Create sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        # Parameters
        pass_threshold = st.slider(
            "Pass Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.7, 
            step=0.1,
            help="Minimum pass rate required to consider a function fixed"
        )
        
        max_iterations = st.slider(
            "Max Iterations", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="Maximum number of improvement attempts"
        )
        
        # API key options
        st.subheader("OpenAI API Key")
        
        # Check if environment variable exists
        env_api_key_exists = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]
        
        # Option to use environment variable or input directly
        api_key_option = st.radio(
            "API Key Source:",
            ["Use environment variable", "Enter manually"],
            index=0 if env_api_key_exists else 1,
            help="Choose whether to use the API key from environment variables or enter it manually"
        )
        
        # Show appropriate UI based on selection
        if api_key_option == "Enter manually":
            openai_key = st.text_input(
                "Enter your OpenAI API key:", 
                type="password",
                help="Required for GPT-4o fixes"
            )
            has_valid_key = bool(openai_key)
            if not has_valid_key:
                st.warning("⚠️ OpenAI API key is required for running fixes")
        else:
            # Display status of environment variable
            if env_api_key_exists:
                st.success("✅ Using OpenAI API key from environment variables")
                openai_key = os.environ["OPENAI_API_KEY"]
                has_valid_key = True
            else:
                st.error("❌ No OpenAI API key found in environment variables")
                st.info("Set the OPENAI_API_KEY environment variable or choose 'Enter manually'")
                openai_key = ""
                has_valid_key = False
        
        # Process button
        process_btn = st.button("Process Code", type="primary", disabled=not uploaded_file or not has_valid_key)
    
    # Main content area
    if uploaded_file and process_btn:
        # Save API key to environment variable if entered manually
        if api_key_option == "Enter manually":
            os.environ["OPENAI_API_KEY"] = openai_key
        
        # Create a temporary directory to store the CSV file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process the uploaded file
            csv_path = os.path.join(temp_dir, uploaded_file.name)
            output_dir = os.path.join(temp_dir, "output")
            
            # Save the uploaded file to the temporary directory
            with open(csv_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display processing status
            status = st.empty()
            status.info("Initializing processor...")
            
            # Create a placeholder for console output
            output_placeholder = st.empty()
            
            # Initialize the processor
            try:
                with capture_output(output_placeholder):
                    processor = BuggyCodeProcessor(
                        csv_path=csv_path,
                        output_dir=output_dir,
                        pass_threshold=pass_threshold,
                        max_iterations=max_iterations
                    )
                    
                    # Process the code
                    status.info("Processing code... This may take a while")
                    results = processor.process_all()
                    
                    # Update status
                    status.success("Processing complete!")
                
                # Display results
                st.header("Results")
                
                if not results:
                    st.warning("No functions were processed.")
                else:
                    # Read the summary.json file
                    try:
                        with open(os.path.join(output_dir, "summary.json"), "r") as f:
                            summary = json.load(f)
                    except Exception:
                        summary = {}
                
                    # Create tabs for each function
                    tabs = st.tabs(list(results.keys()))
                    
                    for i, (func_name, data) in enumerate(results.items()):
                        with tabs[i]:
                            # Function header with stats
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.subheader(f"Function: {func_name}")
                            with col2:
                                pass_rate = data.get('pass_rate', 0)
                                st.metric(
                                    "Pass Rate", 
                                    f"{pass_rate:.1%}",
                                    f"{data.get('tests_passed', 0)}/{data.get('tests_total', 0)} tests"
                                )
                            with col3:
                                st.metric(
                                    "Fixed By", 
                                    data.get('fixed_by', 'N/A'),
                                    f"{data.get('iterations_used', 0)} iterations" if data.get('fixed_by') == 'gpt4o' else None
                                )
                            
                            # Progress of iterations
                            if 'iterations_results' in data and data['iterations_results']:
                                st.subheader("Iteration Progress")
                                
                                # Create progress visualization
                                iterations = data['iterations_results']
                                max_iter = len(iterations) - 1  # Subtract 1 because iteration 0 is the original
                                
                                # Get columns for each iteration
                                cols = st.columns(len(iterations))
                                
                                for idx, iter_data in enumerate(iterations):
                                    with cols[idx]:
                                        iter_num = iter_data['iteration']
                                        pass_rate = iter_data['pass_rate']
                                        
                                        # Display iteration number
                                        if iter_num == 0:
                                            st.markdown("**Original**")
                                        else:
                                            st.markdown(f"**Iteration {iter_num}**")
                                        
                                        # Display pass rate with color
                                        if pass_rate >= pass_threshold:
                                            st.markdown(
                                                f"<div class='iteration-card success'>"
                                                f"Pass: {pass_rate:.1%}<br>"
                                                f"{iter_data['passed']}/{iter_data['total']}"
                                                f"</div>", 
                                                unsafe_allow_html=True
                                            )
                                        else:
                                            st.markdown(
                                                f"<div class='iteration-card {'fail' if idx == max_iter else 'info'}'>"
                                                f"Pass: {pass_rate:.1%}<br>"
                                                f"{iter_data['passed']}/{iter_data['total']}"
                                                f"</div>", 
                                                unsafe_allow_html=True
                                            )
                            
                            # Code tabs
                            code_tabs = st.tabs(["Original Code", "Final Code", "Test Cases"])
                            
                            with code_tabs[0]:
                                st.code(data.get('original_code', 'No code available'), language='python')
                            
                            with code_tabs[1]:
                                st.code(data.get('final_code', 'No code available'), language='python')
                                
                                # Show failure messages if any
                                failure_messages = data.get('failure_messages', [])
                                if failure_messages:
                                    with st.expander(f"Remaining Failures ({len(failure_messages)})", expanded=False):
                                        for i, failure in enumerate(failure_messages):
                                            st.text(f"Failure {i+1}:\n{failure}")
                            
                            with code_tabs[2]:
                                # Get test results
                                test_results = data.get('test_results', [])
                                
                                if test_results:
                                    pass_count = sum(1 for t in test_results if t.get('passed', False))
                                    fail_count = len(test_results) - pass_count
                                    
                                    # Create columns for pass/fail counts
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Passed Tests", pass_count, f"{pass_count/len(test_results):.1%}")
                                    with col2:
                                        st.metric("Failed Tests", fail_count, f"{fail_count/len(test_results):.1%}")
                                    
                                    # Display test cases in a table
                                    test_df = pd.DataFrame([
                                        {
                                            "Test Name": t.get('name', 'Unknown'),
                                            "Arguments": str(t.get('arguments', [])),
                                            "Expected": str(t.get('correct', 'N/A')),
                                            "Status": "✅ Passed" if t.get('passed', False) else "❌ Failed"
                                        }
                                        for t in test_results
                                    ])
                                    
                                    st.dataframe(
                                        test_df, 
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            "Status": st.column_config.TextColumn(
                                                "Status",
                                                width="small"
                                            )
                                        }
                                    )
                                else:
                                    st.info("No test results available")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error(traceback.format_exc())
    
    # Display instructions if no file is uploaded
    elif not uploaded_file:
        st.info("Please upload a CSV file containing buggy code to get started.")
        
        # Sample CSV format
        st.subheader("Expected CSV Format")
        sample_df = pd.DataFrame([
            {"Buggy Code": "import pandas as pd\n\nclass DataProcessor:\n    def __init__(self):\n        pass\n        \n    def process_data(self, data):\n        # Buggy code here\n        return data"}
        ])
        
        st.dataframe(sample_df, use_container_width=True)

if __name__ == "__main__":
    main()