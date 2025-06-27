from local_rag_document_search import load_markdown_files

DOC_PATH = "data/meeting_summaries"

# Load the documents
file_contents, file_paths = load_markdown_files(DOC_PATH)


from minions.minions import Minions
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from pydantic import BaseModel

class StructuredLocalOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None

LOCAL_MODEL_NAME = "qwen2.5:3b"
REMOTE_MODEL_NAME = "gpt-4o-mini"

local_client = OllamaClient(
                        model_name=LOCAL_MODEL_NAME,
                        temperature=0.0,
                        max_tokens=4096,
                        num_ctx=4096,
                        use_async=True, # TODO: consider changing to True
                        structured_output_schema=StructuredLocalOutput
                    )
                    

remote_client = OpenAIClient(
    model_name=REMOTE_MODEL_NAME,
    temperature=0.0,
    max_tokens=4096
)

protocol = Minions(local_client, remote_client)

from minions.minions import Document
document_list = [Document(content=content, filename=filename) for filename, content in zip(file_paths, file_contents)]
print(document_list[0].content[:300])

TASK = "what are the key results and projections for the marketing team?"  # "What are the key results and projections for the marketing team?" # "how many languages are supported by the new website?" "how many one-on-one meetings did I have in total?"


import time

start_time = time.time()
output = protocol(
        task= TASK,
        doc_metadata= "a list of short meeting summaries for Sarah Chen (obtained from zoom AI companion)", #"a list of short zoom AI companion meeting summaries with the following filenames: " + ", ".join([doc.filename for doc in document_list]),
        context=document_list,
        max_rounds=5,  # you can adjust rounds as needed for testing
        use_retrieval="bm25",  # Enable BM25 retrieval
    )
end_time = time.time()
print(f"Protocol execution took {end_time - start_time:.2f} seconds")