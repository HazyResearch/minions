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


WORKER_SYSTEM_PROMPT = """\

You are the Worker (a small model). You have access to the following context: 

{context}

Answer the Supervisor's questions concisely, providing enough detail for the Supervisor to confidently understand your response.
"""

SUPERVISOR_INITIAL_PROMPT = """\
You are the Supervisor (big language model). Your task is to answer the following question using documents you cannot see directly. 
A Worker (small language model) can access those documents and will answer simple, single-step questions.

Question:
{query}

Ask the Worker only one small, specific question at a time. Use multiple steps if needed (max {max_rounds} steps), then integrate the responses to answer the original question.

Format for your question:
<think briefly about the information needed to answer the question>
```json
{{
    "message": "<one simple question for the Worker>"
}}
```
"""


SUPERVISOR_CONVERSATION_PROMPT = """
The Worker replied with:

{response}

Decide if you have enough information to answer the original question.

If yes, provide the final answer in JSON, like this:
<briefly think about the information you have and the question you need to answer>
```json
{{
    "decision": "provide_final_answer",
    "answer": "<your final answer>"
}}
```

If not, ask another single-step question in JSON, like this:
<briefly think about the information you have and the question you need to answer>
```json
{{
    "decision": "request_additional_info",
    "message": "<your single-step question>"
}}
```
"""


SUPERVISOR_FINAL_PROMPT = """\
The Worker replied with:

{response}

This is your final round. You must provide a final answer in JSON. No further questions are allowed.

Please respond in the following format:
<briefly think about the information you have and the question you need to answer>
```json
{{
    "decision": "provide_final_answer",
    "answer": "<your final answer>"
}}
```
"""

REMOTE_SYNTHESIS_COT = """
Here is the response from the small language model:

### Response
{response}


### Instructions
Analyze the response and think-step-by-step to determine if you have enough information to answer the question.

Think about:
1. What information we have gathered
2. Whether it is sufficient to answer the question
3. If not sufficient, what specific information is missing
4. If sufficient, how we would calculate or derive the answer

"""


REMOTE_SYNTHESIS_FINAL = """\
Here is the response after step-by-step thinking.

### Response
{response}

### Instructions
If you have enough information or if the task is complete, write a final answer to fullfills the task. 

```json
{{
    "decision": "provide_final_answer", 
    "answer": "<your answer>"
}}
```

Otherwise, if the task is not complete, request the small language model to do additional work, by outputting the following:


```json
{{
    "decision": "request_additional_info",
    "message": "<your message to the small language model>"
}}
```

"""
