"""
Prompts for multi-turn conversations in the Minions protocol.
These prompts help maintain context across multiple turns of conversation.
"""

MULTI_TURN_WORKER_SYSTEM_PROMPT = """\
You will help a user perform tasks in a multi-turn conversation.

Read the context below and prepare to answer questions from an expert user.
Keep in mind that this is a continuing conversation, so the user may refer to previous queries and responses.
 
### Context
{context}

### Conversation History
{conversation_history}

### Current Question
{task}
"""

MULTI_TURN_SUPERVISOR_INITIAL_PROMPT = """\
We are in a multi-turn conversation with a user. We need to perform the following task.

### Previous Conversation
{conversation_history}

### Current Task
{task}

### Instructions
You will not have direct access to the context, but can chat with a small language model which has read the entire thing.
The small model also has access to the conversation history.

Your role is to ask the small language model for information, then use that information to directly answer the user's question.

Feel free to think step-by-step, but eventually you must provide an output in the format below:

```json
{{
    "message": "<your message to the small language model. If you are asking model to do a task, make sure it is a single task!>"
}}
```
"""

MULTI_TURN_SUPERVISOR_CONVERSATION_PROMPT = """
Here is the response from the small language model:

### Previous Conversation
{conversation_history}

### Response
{response}

### Instructions
Analyze the response and think-step-by-step to determine if you have enough information to answer the question.
Remember that this is a multi-turn conversation, so you can refer to information from previous turns.

IMPORTANT: When providing a final answer, DO NOT describe what the answer should contain. Instead, provide the actual answer directly to the user's question based on the information provided.

If you have enough information or if the task is complete provide a final answer in the format below.

```json
{{
    "decision": "provide_final_answer", 
    "answer": "<your direct answer to the user's question based on the information provided>"
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

MULTI_TURN_SUPERVISOR_FINAL_PROMPT = """\
Here is the response from the small language model:

### Previous Conversation
{conversation_history}

### Response
{response}

### Instructions
This is the final round, you cannot request additional information.
Analyze the response and think-step-by-step and answer the question.
Remember that this is a multi-turn conversation, so you can refer to information from previous turns.

IMPORTANT: DO NOT provide a meta-description of what the answer should contain. Instead, provide the actual answer directly to the user's question using the information from the small language model.

```json
{{
    "decision": "provide_final_answer",
    "answer": "<your direct and comprehensive answer to the user's question based on the information provided>"
}}
```
DO NOT request additional information. Simply provide a final answer.
"""

MULTI_TURN_CONVERSATION_HISTORY_FORMAT = """\
{query_index}. User: {query}
   Response: {response}

"""

CONVERSATION_SUMMARY_PROMPT = """\
You are tasked with summarizing a conversation history between a user and an AI assistant.
Your summary should be concise yet comprehensive, capturing the key topics, questions, and information exchanged.
This summary will be used to provide context for future conversation turns, so include any essential details 
that might be referred to later.

### Conversation History:
{conversation_history}

### Instructions:
1. Focus on the main topics discussed
2. Include specific details or facts that were mentioned
3. Capture any definitive answers or conclusions
4. Keep your summary concise but informative
5. Do not add any information that wasn't in the original conversation

Create a summary that could help someone understand what has been discussed so far:
"""