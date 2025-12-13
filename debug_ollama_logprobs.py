import ollama
import sys

# Get available model
models = ollama.list()
if hasattr(models, 'models'):
    model_list = [m.model for m in models.models]
elif isinstance(models, dict) and 'models' in models:
    model_list = [m['model'] for m in models['models']]
else:
    model_list = ['llama-3.2']
target_model = model_list[0] if model_list else 'llama-3.2'
print(f"Using model: {target_model}")

try:
    print("\n--- Trying top-level logprobs=True ---")
    # Some libraries use this kwarg
    response = ollama.chat(
        model=target_model,
        messages=[{'role': 'user', 'content': 'Hi'}],
        stream=True,
        logprobs=True  # Trying this
    )
    for chunk in response:
        print(f"Resulting logprobs: {chunk.logprobs if hasattr(chunk, 'logprobs') else 'N/A'}")
        if hasattr(chunk, 'logprobs') and chunk.logprobs:
           print("SUCCESS: Logprobs found!")
        break
except Exception as e:
    print(f"Error: {e}")
    # One more try: maybe kwargs is eaten?
    
