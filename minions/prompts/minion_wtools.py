WORKER_SYSTEM_PROMPT_TOOLCALLING = """\
You will help a user perform the following task. 

Read the context below and prepare to answer questions from an expert user. 
### Context
{context}

### Task
{task}

You have access to the following tools:
{TOOL_DESCRIPTION}
"""


SUPERVISOR_INITIAL_PROMPT_TOOLCALLING = """\
We need to perform the following task.

### Task
{task}

### Instructions
You will not have direct access to the context, but can chat with a small language model has access to local context via the following tools:
{TOOL_DESCRIPTION}

Feel free to think step-by-step, but eventually you must provide an output in the format below:

```json
{{
    "message": "<your message to the small language model. If you are asking model to do a task or use a tool, make sure it is a single task or tool call!>"
}}
```
"""

SUPERVISOR_CONVERSATION_PROMPT_TOOLCALLING = """
Here is the response from the small language model:

### Response
{response}


### Instructions
Analyze the response and think-step-by-step to determine if you have enough information to answer the question.

If you have enough information or if the task is complete provide a final answer in the format below.

```json
{{
    "decision": "provide_final_answer", 
    "answer": "<your answer>"
}}
```

Otherwise, if the task is not complete, request the small language model to do additional work, by outputting the following. Recall that the small language model has access to the following tools:
{TOOL_DESCRIPTION}

```json
{{
    "decision": "request_additional_info",
    "message": "<your message to the small language model>"
}}
```
"""

REMOTE_SYNTHESIS_COT_TOOLCALLING = """
Here is the response from the small language model:

### Response
{response}


### Instructions
Analyze the response and think-step-by-step to determine if you have enough information to answer the question. 

The small language model has access to the following tools:
{TOOL_DESCRIPTION}

Think about:
1. What information we have gathered
2. Whether it is sufficient to answer the question
3. If not sufficient, what specific information is missing and what tool call is needed to get the missing information
4. If sufficient, how we would calculate or derive the answer
"""

REMOTE_SYNTHESIS_FINAL_TOOLCALLING = """\
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

Recall that the small language model has access to the following tools:
{TOOL_DESCRIPTION}


```json
{{
    "decision": "request_additional_info",
    "message": "<your message to the small language model>" # consider adding details about the tool call you want to make -- i.e., Please read the file (read_file where path="/Users/avanikanarayan/Downloads/summary.pdf') and extract the information about the company 
```

"""
