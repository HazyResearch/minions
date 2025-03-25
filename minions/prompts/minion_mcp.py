# These prompts are supplementary to those in minion.py, used for when Minion() access to MCP tools

WORKER_SYSTEM_PROMPT_MCP = """\
You will help a user perform the following task.

Read the context below and prepare to answer questions from an expert user. Further context will be provided if the 
expert user invokes MCP tools. The expert user won't be able to see the MCP tool outputs, so please convey everything 
necessary to answer the expert user's questions.
### Context
{context}


### Question
{task}
"""

SUPERVISOR_INITIAL_PROMPT_MCP = """\
We need to perform the following task.

### Task
{task}

### Instructions
You will not have direct access to the context, but can chat with a small language model which has read the entire thing.
If you think the small language model may need more context, you can also specify an MCP tool call, whose output will
be given to the small language model just prior to your task. However, the small language model can't invoke the MCP
tools on its own. Also, you won't see the output, so make sure your message to the small model is clear about
exactly what information you want the small model to convey to you from the tool output.

### MCP Tools Info
{mcp_tools_info}

Feel free to think step-by-step, but eventually you must provide an output in the format below:

```json
{{
    "mcp_tool_calls": [],  # a list of tool calls, formatted in JSON as per their usage (can be an empty list)
    "message": "<your message to the small language model. If you are asking model to do a task, make sure it is a single task!>"
}}
```

Important: if you use MCP tools, you won't see the output, so make sure your message to the small model is clear about
exactly what information you want it to convey to you from the output.
"""

REMOTE_SYNTHESIS_FINAL_MCP = """\
Here is the response after step-by-step thinking.

### Response
{response}

### Instructions
If you have enough information or if the task is complete, write a final answer to fulfill the task. 

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
    "mcp_tool_calls": [],  # a list of tool calls, formatted in JSON as per their usage (can be an empty list)
    "message": "<your message to the small language model>"
}}
```

Important: if you use MCP tools, you won't see the output, so make sure your message to the small model is clear about
exactly what information you want it to convey to you from the output.
"""
