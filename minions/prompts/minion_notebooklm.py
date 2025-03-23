# Modified prompt templates for section-by-section summary and podcast generation

SUPERVISOR_INITIAL_PROMPT = """\
We need to perform the following task.

### Task
Generate a comprehensive podcast script (at least 2000 words) covering all sections of the paper in detail.

### Instructions
You will not have direct access to the context, but can chat with a small language model which has read the entire thing.

I want you to first get a section-by-section summary of the document, then use that information to create a podcast conversation between two hosts discussing this content.

Feel free to think step-by-step, but eventually you must provide an output in the format below:

```json
{{
    "message": "Please provide a section-by-section summary of the document. For each section, include the section title/heading and a concise summary of its key points."
}}
```
"""

SUPERVISOR_CONVERSATION_PROMPT = """
Here is the response from the small language model:

### Response
{response}


### Instructions
Analyze the response and think step-by-step to determine if you have enough information about the document structure and content to create a compelling podcast conversation.

If you have enough information to create a podcast script, provide a final answer in the format below:

```json
{{
    "decision": "provide_final_answer", 
    "answer": "<PODCAST TRANSCRIPT>\\nHOST 1: Welcome to Research Roundup! I'm [Host 1 Name].\\nHOST 2: And I'm [Host 2 Name]. Today we're discussing a fascinating paper about [topic].\\n[Continue natural conversation discussing the key sections, findings, and implications]\\nHOST 1: That's all for today's episode!\\nHOST 2: Thanks for joining us on Research Roundup. Until next time!"
}}
```

Otherwise, if you need more specific information about certain sections or aspects of the document, request the small language model to provide additional details:

```json
{{
    "decision": "request_additional_info",
    "message": "Thank you for the section summaries. Could you please provide more details about [specific section or aspect] including [specific questions or points of interest]?"
}}
```
"""

SUPERVISOR_FINAL_PROMPT = """\
Here is the response from the small language model:

### Response
{response}

### Instructions
This is the final round, you cannot request additional information.
Based on all the section summaries and details provided so far, generate a COMPREHENSIVE and DETAILED podcast conversation between two hosts discussing this document.

Your podcast script MUST be at least 1500-2000 words long and cover EACH section of the paper in detail.
Each host should ask 2-3 questions about EVERY major section, and the other host should provide detailed explanations.

DO NOT provide planning notes or outlines. YOU MUST GENERATE THE ACTUAL WORD-FOR-WORD PODCAST SCRIPT with dialogue between Host 1 and Host 2.

The podcast script MUST follow this structure:
1. Introduction (brief welcome and paper overview)
2. Background section discussion (at least 250 words)
3. Model Architecture discussion (at least 300 words)
4. Detailed exploration of self-attention mechanism (at least 300 words)
5. Results and performance discussion (at least 250 words)
6. Implications and conclusion (at least 200 words)

Each section should feature multiple exchanges between hosts with specific technical details from the paper.

```json
{{
    "decision": "provide_final_answer",
    "answer": "HOST 1: Welcome to Research Roundup! I'm Alex.\\nHOST 2: And I'm Jordan. Today we're diving into a revolutionary paper called 'Attention Is All You Need'.\\n\\n[EXTENSIVE DIALOGUE COVERING EACH SECTION IN DETAIL...]\\n\\nHOST 1: That wraps up our discussion of this groundbreaking paper.\\nHOST 2: Thanks for joining us on Research Roundup. Until next time!"
}}
```
"""

REMOTE_SYNTHESIS_COT = """
Here is the response from the small language model:

### Response
{response}


### Instructions
Analyze the section summaries and information provided. Think step-by-step about how to use this information to create an engaging podcast.

Think about:
1. Do I have a good understanding of the document's overall structure and key sections?
2. Do I have sufficient details about the content, methodology, findings, or arguments in each section?
3. Are there any important sections or aspects that need more clarification?
4. How could the information from each section be presented in a conversational format?
5. What different perspectives might the two podcast hosts have on this content?
6. What aspects would be most interesting or relevant to highlight in a podcast discussion?

"""

REMOTE_SYNTHESIS_FINAL = """\
Here is the response after step-by-step thinking.

### Response
{response}

### Instructions
If you have enough information about the document and its sections, create a podcast script where two hosts discuss the content section by section:

```json
{{
    "decision": "provide_final_answer", 
    "answer": "<PODCAST TRANSCRIPT>\\nHOST 1: Welcome to Research Roundup! I'm [Host 1 Name].\\nHOST 2: And I'm [Host 2 Name]. Today we're discussing [document title/topic].\\n\\n[Conversational discussion of each section, including key points, questions, and insights]\\n\\nHOST 1: That's all for today's episode.\\nHOST 2: Thanks for joining us on Research Roundup. Until next time!"
}}
```

Otherwise, if you need more specific information about certain sections or aspects of the document, request the small language model to provide it:

```json
{{
    "decision": "request_additional_info",
    "message": "Could you please provide more detailed information about [specific section or aspect] including [specific details needed]?"
}}
```
"""

WORKER_SYSTEM_PROMPT = """\
You will help a user perform the following task.

Read the context below and prepare to answer questions from an expert user. 
The context contains a document or paper that you need to analyze carefully.

### Context
{context}

### Question
Generate a comprehensive podcast script (at least 2000 words) covering all sections of the paper in detail.

When asked to provide section-by-section summaries, carefully identify all major sections, subsections, and important parts of the document. For each section, provide:
1. The section title/heading
2. A concise but comprehensive summary of the key points, methodologies, findings, or arguments contained in that section
3. Any particularly notable or important information that stands out

Be thorough and make sure to cover all significant parts of the document.

Make sure to provide detailed content from each section that would be sufficient to create a podcast discussion. Include specific technical details, methodologies, and results that could be discussed in a conversational format.
"""