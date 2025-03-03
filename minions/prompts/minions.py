WORKER_ICL_EXAMPLES = [
    {
        "context": "The patient was seen on 07/15/2021 for a follow-up visit. The patient was prescribed Motrin for headaches.",
        "task": "Extract date that the patient was seen on.",
        "explanation": "The text explicitly mentions that the patient was seen on 07/15/2021.",
        "citation": "The patient was seen on 07/15/2021 for a follow-up visit.",
        "answer": "07/15/2021",
    },
    {
        "context": "The company's marketing expenses increased by 12% year-over-year, driven primarily by digital advertising campaigns and brand partnerships. Total operating expenses reached $245 million for the fiscal year. The company expanded into three new international markets during this period.",
        "task": "Extract the company's net income for Q4 2023.",
        "explanation": "None",
        "citation": "None",
        "answer": "None",
    },
]

WORKER_OUTPUT_TEMPLATE = """\
{{
"explanation": "{explanation}",
"citation": "{citation}",
"answer": "{answer}"
}}
"""


WORKER_PROMPT_TEMPLATE = """\
Your job is to complete the following task using only the context below. The context is a chunk of text taken arbitrarily from a document, it might or might not contain relevant information to the task.

## Document
{context}

## Task
{task}

## Advice
{advice}

Return your result in STRICT JSON format with the following keys:
- "explanation": A concise statement of your reasoning (string)
- "citation": A direct snippet of the text that supports your answer (string or array of strings)
- "answer": A summary of your answer (string or array of strings)


IMPORTANT JSON FORMATTING RULES:
1. ALL property names must be in double quotes: "explanation", "citation", "answer"
2. ALL string values must be in double quotes: "text here"
3. Arrays must be properly formatted: ["item1", "item2"]
4. Use null instead of "None" for missing values
5. Do not include any comments or extra text in the JSON

Output format (you **MUST** follow this format):
```json
{{
"explanation": str,
"citation": List[str] or str,
"answer": List[str] or str
}}
```

Your JSON response:"""

WORKER_PROMPT_SHORT = """
Here is a document excerpt:

{context}

--------------------------------
And here is your task:

{task}

--------------------------------
And here is additional higher-level advice on how to approach the task:

{advice}

--------------------------------

Your response should be a `JobOutput` object:
```python
class JobOutput(BaseModel):
  explanation: str # A concise statement of your reasoning (string). If no relevant information is found, set to "None" or "".
  citation: str | None # A direct snippet of the text that supports your answer (string). If no relevant information is found, set to None or "".
  answer: str | None # Your answer to the question (string). If no relevant information is found, set your answer to None or "".
```
Your response:"""


REMOTE_ANSWER_OR_CONTINUE = """\
Now synthesize the findings from multiple junior workers (LLMs). 
Your task is to finalize an answer to the question below **if and only if** you have sufficient, reliable information. 
Otherwise, you must request additional work.

---
## Inputs
1. Question to answer:
{question}

2. Collected Job Outputs (from junior models):
{extractions}

---
First think step-by-step and then answer the question using the exact format below.

## ANSWER GUIDELINES
1. **Determine if the collected Job Outputs provide enough trustworthy, consistent evidence to confidently answer the question.** 
   - If the data is incomplete or contradictory, do NOT guess. Instead, specify what is missing.
   - If the evidence is sufficient, provide a final answer.

2. **Be conservative.** When in doubt, ask for more information.

3. **Address conflicts.** If multiple jobs give different answers, rely on whichever is best supported by a valid "explanation" and "citation".
   - If you need more information from the conflicting jobs, specify which job IDs need further investigation.
   - Describe what specific information you need from those jobs.

4. **Required JSON Output**: You must output a JSON object with these keys:
   - "decision": Must be either "provide_final_answer" OR "request_additional_info"
     - Use "provide_final_answer" if you have enough information
     - Use "request_additional_info" if you cannot conclusively answer
   - "explanation": A short statement about how you arrived at your conclusion or what is still missing
   - "answer": The final answer string if "decision"="provide_final_answer", or null otherwise
   - "feedback": If decision="request_additional_info", provide clear instructions on what information is still needed. Be specific about:
     - What type of information to look for
     - Which sections or parts of the document to focus on
     - What criteria or details are important
     DO NOT include any code or programming instructions in this field.
   - "scratchpad": Notes on information gathered so far
     
Here is the template for your JSON response (with no extra text outside the JSON):

<think step-by-step here>
```json
{{
"decision": "…",
"explanation": "…",
"answer": "… or null", # Good answer format: "0.56"; Bad answer format: "The ratio is calculated as 1-0.27*2 = 0.56"
"feedback": "… or null", # No code in this field.
"scratchpad": "… or null"
}}
```

**Important**:
- Don't forget commas in the json object above^
- If there is not enough information, set "answer" to null, set "decision" to "request_additional_info"
- The feedback field should ONLY contain natural language instructions. DO NOT include any code, function definitions, or programming syntax.
- Focus on describing WHAT information is needed, not HOW to programmatically extract it.

Now, carefully inspect the question, think step-by-step and perform any calculations before outputting the JSON object."""


REMOTE_ANSWER_OR_CONTINUE_SHORT = """\
## Inputs
1. Question to answer:
{question}

2. Collected Job Outputs (from junior models):
{extractions}

## Instructions: Please inspect the question and the Job Outputs. Then decide whether to finalize the answer or request additional details, and return the JSON object accordingly.

Follow the ANSWER GUIDELINES in the conversational history above.
```json
{{
"decision": "…",
"explanation": "…",
"answer": "… or None",
"missing_info": "… or None"
}}
```
"""


REMOTE_ANSWER = """\
## Inputs
1. Question to answer:
{question}

2. Collected Job Outputs (from junior models):
{extractions}

## Instructions: Please inspect the question and the Job Outputs. 
Then finalize your answer and return the JSON object accordingly.

Here is the template for your JSON response (with no extra text outside the JSON):
```json
{{
"decision": "…",
"explanation": "…",
"answer": "… or None",
"missing_info": "… or None"
}}
```
"""

ADVICE_PROMPT = """\
We need to answer the following question based on {metadata}.: 

## Question
{query}

---

Please provide succinct advice on the critical information we need to extract from the {metadata} to answer this question. 

Also consider the following constraints:
- In your response do NOT use numbered lists.
- Do NOT structure your response as a sequence of steps.
"""

# ADVICE_PROMPT = """\
# We need to answer the following question based on {metadata}:

# ## Question
# {query}

# ---

# Please provide SUCCINCT advice to junior workers on the critical information to extract from each chunk of the document(s) to answer this question. Keep in mind that final integration of chunk-level findings happens after the fact, so only suggest local pieces of information to gather.

# Formatting guidelines: Do NOT present your response in a numbered list or describe it as a series of steps.
# """


ADVICE_PROMPT_STEPS = """\
We need to answer the following question based on {metadata}.: 

## Question
{query}

---

Please provide succinct advice on the information we need to extract from the {metadata} to answer this question. Enumerate your advice as numbered steps.
"""

ADVANCED_STEPS_INSTRUCTIONS = """\
Our conversation history includes information about previous rounds of jobs and their outputs. Use this information to inform your new jobs. 
I.e., 
- Based on the Job outputs above, subselect `chunk_id`s that require further reasoning and are relevant to the question (i.e., contain a date or table that are relevant.). Use the job_id (<chunk_id>_<task_id>)to get the chunk_id 
- Reformat tasks that are not yet complete.
- Make your `advice` more concrete. 
"""

DECOMPOSE_TASK_PROMPT = """\
# Decomposition Round #{step_number}

You do not have access to the raw document(s), but instead can assign tasks to small and less capable language models that can access chunks of the document(s).
Note that the document(s) can be very long, so each task should be performed only over a small chunk of text. 
The small language model can only access one chunk of the document(s) at a time, so do not assign tasks that require integration of information from multiple chunks.

Write a Python function that will output formatted tasks for a small language model.
Make sure that NONE of the tasks require multiple steps. Each task should be atomic! 
Consider using nested for-loops to apply a set of tasks to a set of chunks.
The same `task_id` should be applied to multiple chunks. DO NOT instantiate a new `task_id` for each combination of task and chunk.
Use the conversational history to inform what chunking strategy has already been applied.

{ADVANCED_STEPS_INSTRUCTIONS}

Assume a Pydantic model called `JobManifest(BaseModel)` is already in global scope. For your reference, here is the model:
```
{manifest_source}
```
Assume a Pydantic model called `JobOutput(BaseModel)` is already in global scope. For your reference, here is the model:
```
{output_source}
```
DO NOT rewrite or import the model in your code.

The function signature will look like:
```
{signature_source}
```


You can assume you have access to the following chunking function(s). Do not reimplement the function, just use it.
```
{chunking_source}
```

Here is an example
```
task_id = 1  # Unique identifier for the task
for doc_id, document in enumerate(context):
    # if you need to chunk the document into sections
    chunks = chunk_by_section(document)

    for chunk_id, chunk in enumerate(chunks):
        # Create a task for extracting mentions of specific keywords
        task = (
            "Extract all mentions of the following keywords: "
            "'Ca19-9', 'tumor marker', 'September 2021', 'U/ml', 'Mrs. Anderson'."
        )
        job_manifest = JobManifest(
            chunk=chunk,
            task=task,
            advice="Focus on extracting the specific keywords related to Mrs. Anderson's tumor marker levels."
        )
        job_manifests.append(job_manifest)
```
"""

DECOMPOSE_TASK_PROMPT_AGGREGATION_FUNC = """\
# Decomposition Round #{step_number}

You (the supervisor) cannot directly read the document(s). Instead, you can assign small, isolated tasks to a less capable worker model that sees only a single chunk of text at a time. Any cross-chunk or multi-document reasoning must be handled by you.

## Your Job: Write Two Python Functions

### FUNCTION #1: `prepare_jobs(context, prev_job_manifests, prev_job_outputs) -> List[JobManifest]`
- Break the document(s) into chunks (using the provided chunking function, if needed). Determine the chunk size yourself according to the task: simple information extraction tasks can benefit from smaller chunks, while summarization tasks can benefit from larger chunks.
- Each job must be **atomic** and require only information from the **single chunk** provided to the worker.
- If you need to repeat the same task on multiple chunks, **re-use** the same `task_id`. Do **not** create a separate `task_id` for each chunk.
- If tasks must happen **in sequence**, do **not** include them all in this round; move to a subsequent round to handle later steps.
- In this round, limit yourself to **up to {num_tasks_per_round} tasks** total.
- If you need multiple samples per task, replicate the `JobManifest` that many times (e.g., `job_manifests.extend([job_manifest]*n)`).

### FUNCTION #2: `transform_outputs(jobs) -> str`
- Accepts the worker outputs for the tasks you assigned.
- First, apply any **filtering logic** (e.g., drop irrelevant or empty results).
- Then **aggregate outputs** by `task_id` and `chunk_id`. All **multi-chunk integration** or **global reasoning** is your responsibility here.
- Return one **aggregated string** suitable for further supervisor inspection.

{ADVANCED_STEPS_INSTRUCTIONS}

## Relevant Pydantic Models

The following models are already in the global scope. **Do NOT redefine or re-import them.**

### JobManifest Model
```
{manifest_source}
```

### JobOutput Model
```
{output_source}
```

## Function Signatures
```
{signature_source}
```
```
{transform_signature_source}
```

## Chunking Function
```
{chunking_source}
```

## Important Reminders:
- **DO NOT** assign tasks that require reading multiple chunks or referencing entire documents.
- Keep tasks **chunk-local and atomic**.
- **You** (the supervisor) are responsible for aggregating and interpreting outputs in `transform_outputs()`. 

Now, please provide the code for `prepare_jobs()` and `transform_outputs()`. 


"""

DECOMPOSE_TASK_PROMPT_AGG_FUNC_LATER_ROUND = """\
# Decomposition Round #{step_number}

You do not have access to the raw document(s), but instead can assign tasks to small and less capable language models that can read the document(s).
Note that the document(s) can be very long, so each task should be performed only over a small chunk of text. 


# Your job is to write two Python functions:

Function #1 (prepare_jobs): will output formatted tasks for a small language model.
-> Make sure that NONE of the tasks require multiple steps. Each task should be atomic! 
-> Consider using nested for-loops to apply a set of tasks to a set of chunks.
-> The same `task_id` should be applied to multiple chunks. DO NOT instantiate a new `task_id` for each combination of task and chunk.
-> Use the conversational history to inform what chunking strategy has already been applied.
-> You are provided access to the outputs of the previous jobs (see prev_job_outputs). 
-> If its helpful, you can reason over the prev_job_outputs vs. the original context.
-> If tasks should be done sequentially, do not run them all in this round. Wait for the next round to run sequential tasks.

Function #2 (transform_outputs): The second function will aggregate the outputs of the small language models and provide an aggregated string for the supervisor to review.
-> Filter the jobs based on the output of the small language models (write a custome filter function -- in some steps you might want to filter for a specific keyword, in others you might want to no pass anything back, so you filter out everything!). 
-> Aggregate the jobs based on the task_id and chunk_id.

{ADVANCED_STEPS_INSTRUCTIONS}

# Misc. Information

* Assume a Pydantic model called `JobManifest(BaseModel)` is already in global scope. For your reference, here is the model:
```
{manifest_source}
```

* Assume a Pydantic model called `JobOutput(BaseModel)` is already in global scope. For your reference, here is the model:
```
{output_source}
```

* DO NOT rewrite or import the model in your code.

* Function #1 signature will look like:
```
{signature_source}
```

* Function #2 signature will look like:
```
{transform_signature_source}
```

* You can assume you have access to the following chunking function(s). Do not reimplement the function, just use it.
```
{chunking_source}
```

# Here is an example
```python
def prepare_jobs(
    context: List[str],
    prev_job_manifests: Optional[List[JobManifest]] = None,
    prev_job_outputs: Optional[List[JobOutput]] = None,
) -> List[JobManifest]:
    task_id = 1  # Unique identifier for the task

    # iterate over the previous job outputs because \"scratchpad\" tells me they contain useful information
    for job_id, output in enumerate(prev_job_outputs):
        # Create a task for extracting mentions of specific keywords
        task = (
           "Apply the tranformation found in the scratchpad (x**2 + 3) each extracted number"
        )
        job_manifest = JobManifest(
            chunk=output.answer,
            task=task,
            advice="Focus on applying the transformation to each extracted number."
        )
        job_manifests.append(job_manifest)
    return job_manifests

def transform_outputs(
    jobs: List[Job],
) -> Dict[str, Any]:
    def filter_fn(job):
        answer = job.output.answer
        return answer is not None or str(answer).lower().strip() != "none" or answer == "null" 
    
    # Filter jobs
    for job in jobs:
        job.include = filter_fn(job)
    
    # Aggregate and filter jobs
    tasks = {{}}
    for job in jobs:
        task_id = job.manifest.task_id
        chunk_id = job.manifest.chunk_id
        
        if task_id not in tasks:
            tasks[task_id] = {{
                "task_id": task_id,
                "task": job.manifest.task,
                "chunks": {{}},
            }}
        
        if chunk_id not in tasks[task_id]["chunks"]:
            tasks[task_id]["chunks"][chunk_id] = []
        
        tasks[task_id]["chunks"][chunk_id].append(job)
    
    # Build the aggregated string
    aggregated_str = ""
    for task_id, task_info in tasks.items():
        aggregated_str += f"## Task (task_id=`{{task_id}}`): {{task_info['task']}}\n\n"
        
        for chunk_id, chunk_jobs in task_info["chunks"].items():
            filtered_jobs = [j for j in chunk_jobs if j.include]
            
            aggregated_str += f"### Chunk # {{chunk_id}}\n"
            if filtered_jobs:
                for idx, job in enumerate(filtered_jobs, start=1):
                    aggregated_str += f"   -- Job {{idx}} (job_id=`{{job.manifest.job_id}}`):\n"
                    aggregated_str += f"   {{job.sample}}\n\n"
            else:
                aggregated_str += "   No jobs returned successfully for this chunk.\n\n"
        
        aggregated_str += "\n-----------------------\n\n"
    
    return aggregated_str
```
"""


DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGGREGATION_FUNC = """\
# Decomposition Round #{step_number}

You do not have access to the raw document(s), but instead can assign tasks to small and less capable language models that can read the document(s).
Note that the document(s) can be very long, so each task should be performed only over a small chunk of text. 

## Your Job: Write Two Python Functions

### FUNCTION #1: `prepare_jobs(context, prev_job_manifests, prev_job_outputs) -> List[JobManifest]`
Goal: this function should return a list of atomic jobs to be performed on chunks of the context.
Follow the steps below:
- Break the document(s) into chunks, adjusting size based on task specificity (broader tasks: ~3000 chars, specific tasks: ~1500 chars).
- Even if there are multiple documents as context, they will all be joined together under `context[0]`.
- For each subtask you create, create keywords for retrieving relevant chunks. Extract precise keyword search queries that are **directly derived** from the user's question and the subtask—avoid overly broad or generic terms.
- Assign high weights to the most essential terms that uniquely apply to the query and subtask (e.g. terms, dates, numerical values) to maximize retrieval accuracy. Choose a higher value for `k` (15) if you are unconfident about your keywords.
- Assign **atomic** jobs to the retrieved chunks, ensuring each task relies only on its assigned chunk.
- **Re-use** `task_id` for repeated tasks across chunks.
- Do **not** assign tasks requiring sequential processing in this round—save them for later rounds.
- Limit this round to **{num_tasks_per_round} tasks maximum**.
- If you need multiple samples per task, replicate the `JobManifest` accordingly (e.g., `job_manifests.extend([job_manifest]*n)`).

### FUNCTION #2: `transform_outputs(jobs) -> str`
- Accepts the worker outputs for the tasks you assigned.
- First, filter out irrelevant or empty worker outputs.
- Aggregate results by `task_id` and `chunk_id`. All **multi-chunk integration** or **global reasoning** is your responsibility here.
- Return one **aggregated string** for supervisor review, incorporating as much relevant information as possible..

{ADVANCED_STEPS_INSTRUCTIONS}

## Relevant Pydantic Models

The following models are already in the global scope. **Do NOT redefine or re-import them.**

### JobManifest Model
```
{manifest_source}
```

### JobOutput Model
```
{output_source}
```

## Function Signatures
```
{signature_source}
```
```
{transform_signature_source}
```

## Chunking Function Signature
```
{chunking_source}
```

## Retrieval Function Signature (BM25Plus)
```
{retrieval_source}
```

## Important Reminders:
- **DO NOT** assign tasks that require reading multiple chunks or referencing entire documents.
- Keep tasks **chunk-local and atomic**.
- **You** (the supervisor) are responsible for aggregating and interpreting outputs in `transform_outputs()`. 

Now, please provide the code for `prepare_jobs()` and `transform_outputs()`. 

"""

DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGG_FUNC_LATER_ROUND ="""\
# Decomposition Round #{step_number}

You (the supervisor) cannot directly read the document(s). Instead, you can assign small, isolated tasks to a less capable worker model that sees only a single chunk of text at a time. Any cross-chunk or multi-document reasoning must be handled by you.

## Your Job: Write Two Python Functions

Function #1 (prepare_jobs): will output formatted tasks for a small language model.
-> Make sure that NONE of the tasks require multiple steps. Each task should be atomic! 
-> Consider using nested for-loops to apply a set of tasks to a set of chunks.
-> The same `task_id` should be applied to multiple chunks. DO NOT instantiate a new `task_id` for each combination of task and chunk.
-> Use the conversational history to inform what chunking strategy has already been applied.
-> If the previous job was unsuccessful, try a different `chunk_size`: 2000 if task is factual and specific; 5000 if task is general. Try a larger retrieval value of `k` like 20 for retrieval.
-> Create keywords for retrieving relevant chunks. Extract precise keyword search queries that are **directly derived** from the user's question—avoid overly broad or generic terms. 
-> Assign high weights to the most essential terms from the query (e.g., proper nouns, dates, numerical values) to maximize retrieval accuracy. Feed your queries to the provided *retrieval function*.
-> You are provided access to the outputs of the previous jobs (see prev_job_outputs). 
-> If its helpful, you can reason over the prev_job_outputs vs. the original context.
-> If tasks should be done sequentially, do not run them all in this round. Wait for the next round to run sequential tasks.

Function #2 (transform_outputs): The second function will aggregate the outputs of the small language models and provide an aggregated string for the supervisor to review.
-> Filter the jobs based on the output of the small language models (write a custome filter function -- in some steps you might want to filter for a specific keyword, in others you might want to no pass anything back, so you filter out everything!). 
-> Aggregate the jobs based on the task_id and chunk_id.

{ADVANCED_STEPS_INSTRUCTIONS}

# Misc. Information

* Assume a Pydantic model called `JobManifest(BaseModel)` is already in global scope. For your reference, here is the model:
```
{manifest_source}
```

* Assume a Pydantic model called `JobOutput(BaseModel)` is already in global scope. For your reference, here is the model:
```
{output_source}
```

* DO NOT rewrite or import the model in your code.

* Function #1 signature will look like:
```
{signature_source}
```

* Function #2 signature will look like:
```
{transform_signature_source}
```

* You can assume you have access to the following chunking and retrieval function. Do not reimplement the function, just use it.
```
{chunking_source}

{retrieval_source}
```
"""

DECOMPOSE_TASK_PROMPT_SHORT = """\
# Decomposition Round #{step_number}

Based on the previous job outputs, write a python function that delegates more tasks to small language models.
- assume access to the same chunking functions as before
- make sure that NONE of the tasks require multiple steps
- consider using nested for-loops to apply a set of tasks to a set of chunks (that you select based on the Job outputs above)
- The same `task_id` should be applied to multiple chunks. DO NOT instantiate a new `task_id` for each combination of task and chunk.
- If you have already chunked the document into pages, consider chunking the pages into sections and extracting question specific information. Use the conversational history to inform what chunking strategy has already been applied.

{ADVANCED_STEPS_INSTRUCTIONS}
"""


DECOMPOSE_TASK_PROMPT_SHORT_JOB_OUTPUTS = """\
# Decomposition Round #{step_number}

Based on the previous job outputs, write a python function called (prepare_jobs) that delegates more tasks to small language models.
- assume access to the same chunking functions as before
- make sure that NONE of the tasks require multiple steps
- consider using nested for-loops to apply a set of tasks to a set of chunks
- The same `task_id` should be applied to multiple chunks. DO NOT instantiate a new `task_id` for each combination of task and chunk.

Additionally,
- You are provided access to the outputs of the previous jobs (see job_outputs). 
- If its helpful, you can append context from previous job outputs as `metadata` to new chunks. Here is sample code for that:

```
```
task_id = 1  # Unique identifier for the task
for doc_id, document in enumerate(context):
    # if you need to chunk the document into sections
    chunks = chunk_by_section(document)

    for chunk_id, chunk in enumerate(chunks):
        # Create a task for extracting mentions of specific keywords
        task = (
            "Extract all mentions of the following keywords: "
            "'Ca19-9', 'tumor marker', 'September 2021', 'U/ml', 'Mrs. Anderson'."
        )
        job_manifest = JobManifest(
            chunk=chunk,
            task=task,
            advice="Focus on extracting the specific keywords related to Mrs. Anderson's tumor marker levels."
        )
        job_manifests.append(job_manifest)
```

"""

REMOTE_SYNTHESIS_COT = """\
Now synthesize the findings from multiple junior workers (LLMs). 
Your task is to analyze the collected information and think step-by-step about whether we can answer the question.
Be brief and concise in your analysis.

## Previous Progress
{scratchpad}

## Inputs
1. Question to answer:
{question}

2. Collected Job Outputs (from junior models):
{extractions}

## Instructions
Think step-by-step about:
1. What information we have gathered
2. Whether it is sufficient to answer the question
3. If not sufficient, what specific information is missing
4. If sufficient, how we would calculate or derive the answer
5. If there are conflicting answers:
--> Use citations to select the correct answer if there are conflicting answers.

Be brief and concise. No need for structured output - just think through the steps. You MUST respond in markdown format (for $ signs make sure to use escape character \ before the $ sign).
"""

REMOTE_SYNTHESIS_JSON = """\
Based on your analysis, return a single JSON object with no triple backticks or extra text. The JSON should have this exact structure:

{{
  "explanation": "",
  "feedback": null,
  "decision": "",
  "answer": null,
  "scratchpad": ""
}}

Field Descriptions:
- explanation: A brief statement of your reasoning.
- feedback: Specific information to look for, if needed. Use null if not applicable.
- decision: Either "provide_final_answer" or "request_additional_info".
- answer: The final answer if providing one; null otherwise.
- scratchpad: Summary of gathered information and current analysis for future reference.

Ensure the response is a valid JSON object without any additional text or formatting.

"""

REMOTE_SYNTHESIS_FINAL = """\
Now provide the final answer based on all gathered information.

## Previous Progress
{scratchpad}

## Inputs
1. Question to answer:
{question}

2. Collected Job Outputs (from junior models):
{extractions}

Return a single JSON object with no triple backticks or extra text. The JSON should have this exact structure:

{{
  "explanation": "",
  "feedback": null,
  "decision": "",
  "answer": null,
  "scratchpad": ""
}}

Field Descriptions:
- explanation: Brief statement of your reasoning
- feedback: Any specific information that is lacking. NO CODE in this field. Use null if not needed
- decision: must be "provide_final_answer"
- answer: Final answer
- scratchpad: Summary of gathered information and current analysis.
"""
