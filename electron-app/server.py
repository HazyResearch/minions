# server.py
from flask import Flask, request, jsonify
from app import initialize_clients, run_protocol
import os

app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query():
    data = request.json

    # 1. Extract parameters from frontend
    task = data.get("query")
    context = data.get("context", "")
    doc_metadata = data.get("metadata", "")
    protocol = data.get("protocol", "Minion")
    local_provider = data.get("local_provider", "Ollama")
    remote_provider = data.get("provider", "OpenAI")
    local_model = data.get("local_model", "llama3.2")
    remote_model = data.get("remote_model", "gpt-4o")
    local_temp = float(data.get("local_temperature", 0.0))
    local_max_tokens = int(data.get("local_max_tokens", 4096))
    remote_temp = float(data.get("remote_temperature", 0.0))
    remote_max_tokens = int(data.get("remote_max_tokens", 4096))
    reasoning_effort = data.get("reasoning_effort", "medium")
    api_key = data.get("api_key") or os.environ.get(f"{remote_provider.upper()}_API_KEY")
    num_ctx = int(data.get("num_ctx", 4096))
    mcp_server_name = data.get("mcp_server_name", None)
    images = data.get("images", None)

    try:
        # 2. Initialize LLM clients
        local_client, remote_client, method = initialize_clients(
            local_model,
            remote_model,
            remote_provider,
            local_provider,
            protocol,
            local_temp,
            local_max_tokens,
            remote_temp,
            remote_max_tokens,
            api_key,
            num_ctx,
            mcp_server_name=mcp_server_name,
            reasoning_effort=reasoning_effort,
        )

        # 3. Run the query protocol
        output, setup_time, exec_time = run_protocol(
            task=task,
            context=context,
            doc_metadata=doc_metadata,
            status=None,  # No Streamlit status needed
            protocol=protocol,
            local_provider=local_provider,
            images=images,
        )

        return jsonify({
            "final_answer": output.get("final_answer", ""),
            "setup_time": setup_time,
            "exec_time": exec_time
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000)
