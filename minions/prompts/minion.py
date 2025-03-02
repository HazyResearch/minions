SUPERVISOR_INITIAL_PROMPT = """\
We need to perform the following task.

### Task
{task}

### Instructions
You will not have direct access to the context, but can chat with a small language model which has read the entire thing.

Feel free to think step-by-step, but eventually you must provide an output in the format below:

<think step by step here>
```json
{{
    "message": "<your message to the small language model. If you are asking model to do a task, make sure it is a single task!>"
}}
```
"""

WORKER_PRIVACY_SHIELD_PROMPT = """\
You are a helpful assistant that is very mindful of user privacy. You are communicating with a powerful large language model that you are sharing information with. Revise the following text to preserve user privacy. We have already extracted the PII from the original document. Remove any PII from the text. Provide your output without any preamble. 

### PII Extracted:
{pii_extracted}

### Text to revise:
{output}

### Revised Text:"""

REFORMAT_QUERY_PROMPT = """\
You are a helpful assistant that is very mindful of user privacy. You are communicating with a powerful large language model that you are sharing information with. Revise the following query to remove any PII. Provide your output without any preamble. DO NOT ANSWER THE QUERY, JUST REMOVE THE PII.

### Extracted PII:
{pii_extracted}

### Query:
{query}

### Query without PII (remove the PII from the query, and rephrase the query if necessary):"""

SUPERVISOR_CONVERSATION_PROMPT = """
Here is the response from the small language model:

### Response
{response}


### Instructions
Analyze the response and think-step-by-step to determine if you have enough information to answer the question.

If you have enough information or if the task is complete provide a final answer in the format below.

<think step by step here>
```json
{{
    "decision": "provide_final_answer", 
    "answer": "<your answer>"
}}
```

Otherwise, if the task is not complete, request the small language model to do additional work, by outputting the following:

<think step by step here>
```json
{{
    "decision": "request_additional_info",
    "message": "<your message to the small language model>"
}}
```
"""

SUPERVISOR_FINAL_PROMPT = """\
Here is the response from the small language model:

### Response
{response}


### Instructions
This is the final round, you cannot request additional information.
Analyze the response and think-step-by-step and answer the question.

<think step by step here>
```json
{{
    "decision": "provide_final_answer",
    "answer": "<your answer>"
}}
```
DO NOT request additional information. Simply provide a final answer.
"""

WORKER_SYSTEM_PROMPT = """\
You will help a user perform the following task.

Read the context below and prepare to answer questions from an expert user. 
### Context
{context}

### Question
{task}
"""
