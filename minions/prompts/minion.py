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



# Override the supervisor initial prompt to encourage task decomposition.
SUPERVISOR_INITIAL_PROMPT = """\
We need to perform the following task.

### Task
{task}

### Instructions
You will not have direct access to the context, but you can chat with a small language model that has read the entire content.

Let's use an incremental, step-by-step approach to ensure we fully decompose the task before proceeding. Please follow these steps:

1. Decompose the Task:
   Break down the overall task into its key components or sub-tasks. Identify what needs to be done and list these sub-tasks.

2. Explain Each Component:
   For each sub-task, briefly explain why it is important and what you expect it to achieve. This helps clarify the reasoning behind your breakdown.

3. Formulate a Focused Message:
   Based on your breakdown, craft a single, clear message to send to the small language model. This message should represent one focused sub-task derived from your decomposition.

4. Conclude with a Final Answer:  
   After your reasoning, please provide a **concise final answer** that directly and conclusively addresses the original task. Make sure this final answer includes all the specific details requested in the task.

Your output should be in the following JSON format:

```json
{{
    "reasoning": "<your detailed, step-by-step breakdown here>",
    "message": "<your final, focused message to the small language model>"
}}
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



# Override the final response prompt to encourage a more informative final answer
REMOTE_SYNTHESIS_FINAL = """\
Here is the detailed response from the step-by-step reasoning phase.

### Detailed Response
{response}

### Instructions
Based on the detailed reasoning above, synthesize a clear and informative final answer that directly addresses the task with all the specific details required. In your final answer, please:

1. Summarize the key findings and reasoning steps.
2. Clearly state the conclusive answer, incorporating the important details.
3. Ensure the final answer is self-contained and actionable.

If you determine that you have gathered enough information to fully answer the task, output the following JSON with your final answer:

```json
{{
    "decision": "provide_final_answer", 
    "answer": "<your detailed, conclusive final answer here>"
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
