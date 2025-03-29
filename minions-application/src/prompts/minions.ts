export const WORKER_PROMPT_TEMPLATE = `Your job is to complete the following task using only the context below. The context is a chunk of text taken arbitrarily from a document, it might or might not contain relevant information to the task.

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
\`\`\`json
{
"explanation": str,
"citation": List[str] or str,
"answer": List[str] or str
}
\`\`\`

Your JSON response:`;

export const WORKER_PROMPT_SHORT = `Here is a document excerpt:

{context}

--------------------------------
And here is your task:

{task}

--------------------------------
And here is additional higher-level advice on how to approach the task:

{advice}

--------------------------------

Your response should be a \`JobOutput\` object:
\`\`\`python
class JobOutput(BaseModel):
  explanation: str # A concise statement of your reasoning (string). If no relevant information is found, set to "None" or "".
  citation: str | None # A direct snippet of the text that supports your answer (string). If no relevant information is found, set to None or "".
  answer: str | None # Your answer to the question (string). If no relevant information is found, set your answer to None or "".
\`\`\`
Your response:`;

export const ADVANCED_STEPS_INSTRUCTIONS = `Our conversation history includes information about previous rounds of jobs and their outputs. Use this information to inform your new jobs. 
I.e., 
- Based on the Job outputs above, subselect \`chunk_id\`s that require further reasoning and are relevant to the question (i.e., contain a date or table that are relevant.). Use the job_id (<chunk_id>_<task_id>)to get the chunk_id 
- Reformat tasks that are not yet complete.
- Make your \`advice\` more concrete.`;

export const DECOMPOSE_TASK_PROMPT = `# Decomposition Round #{step_number}

You do not have access to the raw document(s), but instead can assign tasks to small and less capable language models that can access chunks of the document(s).
Note that the document(s) can be very long, so each task should be performed only over a small chunk of text. 
The small language model can only access one chunk of the document(s) at a time, so do not assign tasks that require integration of information from multiple chunks.

Write a Python function that will output formatted tasks for a small language model.
Make sure that NONE of the tasks require multiple steps. Each task should be atomic! 
Consider using nested for-loops to apply a set of tasks to a set of chunks.
The same \`task_id\` should be applied to multiple chunks. DO NOT instantiate a new \`task_id\` for each combination of task and chunk.
Use the conversational history to inform what chunking strategy has already been applied.

{ADVANCED_STEPS_INSTRUCTIONS}

Assume a Pydantic model called \`JobManifest(BaseModel)\` is already in global scope. For your reference, here is the model:
\`\`\`
{manifest_source}
\`\`\`
Assume a Pydantic model called \`JobOutput(BaseModel)\` is already in global scope. For your reference, here is the model:
\`\`\`
{output_source}
\`\`\`
DO NOT rewrite or import the model in your code.

The function signature will look like:
\`\`\`
{signature_source}
\`\`\`

You can assume you have access to the following chunking function(s). Do not reimplement the function, just use it.
\`\`\`
{chunking_source}
\`\`\`

Here is an example
\`\`\`python
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
\`\`\``;

export const DECOMPOSE_TASK_PROMPT_AGGREGATION_FUNC = `# Decomposition Round #{step_number}

You (the supervisor) cannot directly read the document(s). Instead, you can assign small, isolated tasks to a less capable worker model that sees only a single chunk of text at a time. Any cross-chunk or multi-document reasoning must be handled by you.

## Your Job: Write Two Python Functions

### FUNCTION #1: \`prepare_jobs(context, prev_job_manifests, prev_job_outputs) -> List[JobManifest]\`
- Break the document(s) into chunks (using the provided chunking function, if needed). Determine the chunk size yourself according to the task: simple information extraction tasks can benefit from smaller chunks, while summarization tasks can benefit from larger chunks.
- Each job must be **atomic** and require only information from the **single chunk** provided to the worker.
- If you need to repeat the same task on multiple chunks, **re-use** the same \`task_id\`. Do **not** create a separate \`task_id\` for each chunk.
- If tasks must happen **in sequence**, do **not** include them all in this round; move to a subsequent round to handle later steps.
- In this round, limit yourself to **up to {num_tasks_per_round} tasks** total.
- If you need multiple samples per task, replicate the \`JobManifest\` that many times (e.g., \`job_manifests.extend([job_manifest]*n)\`).

### FUNCTION #2: \`transform_outputs(jobs) -> str\`
- Accepts the worker outputs for the tasks you assigned.
- First, apply any **filtering logic** (e.g., drop irrelevant or empty results).
- Then **aggregate outputs** by \`task_id\` and \`chunk_id\`. All **multi-chunk integration** or **global reasoning** is your responsibility here.
- Return one **aggregated string** suitable for further supervisor inspection.

{ADVANCED_STEPS_INSTRUCTIONS}

## Relevant Pydantic Models

The following models are already in the global scope. **Do NOT redefine or re-import them.**

### JobManifest Model
\`\`\`
{manifest_source}
\`\`\`

### JobOutput Model
\`\`\`
{output_source}
\`\`\`

## Function Signatures
\`\`\`
{signature_source}
\`\`\`
\`\`\`
{transform_signature_source}
\`\`\`

## Chunking Function
\`\`\`
{chunking_source}
\`\`\`

## Important Reminders:
- **DO NOT** assign tasks that require reading multiple chunks or referencing entire documents.
- Keep tasks **chunk-local and atomic**.
- **You** (the supervisor) are responsible for aggregating and interpreting outputs in \`transform_outputs()\`. 

Now, please provide the code for \`prepare_jobs()\` and \`transform_outputs()\`.`;

export const DECOMPOSE_TASK_PROMPT_AGG_FUNC_LATER_ROUND = `# Decomposition Round #{step_number}

You do not have access to the raw document(s), but instead can assign tasks to small and less capable language models that can read the document(s).
Note that the document(s) can be very long, so each task should be performed only over a small chunk of text. 


# Your job is to write two Python functions:

Function #1 (prepare_jobs): will output formatted tasks for a small language model.
-> Make sure that NONE of the tasks require multiple steps. Each task should be atomic! 
-> Consider using nested for-loops to apply a set of tasks to a set of chunks.
-> The same \`task_id\` should be applied to multiple chunks. DO NOT instantiate a new \`task_id\` for each combination of task and chunk.
-> Use the conversational history to inform what chunking strategy has already been applied.
-> You are provided access to the outputs of the previous jobs (see prev_job_outputs). 
-> If its helpful, you can reason over the prev_job_outputs vs. the original context.
-> If tasks should be done sequentially, do not run them all in this round. Wait for the next round to run sequential tasks.

Function #2 (transform_outputs): The second function will aggregate the outputs of the small language models and provide an aggregated string for the supervisor to review.
-> Filter the jobs based on the output of the small language models (write a custome filter function -- in some steps you might want to filter for a specific keyword, in others you might want to no pass anything back, so you filter out everything!). 
-> Aggregate the jobs based on the task_id and chunk_id.

{ADVANCED_STEPS_INSTRUCTIONS}

# Misc. Information

* Assume a Pydantic model called \`JobManifest(BaseModel)\` is already in global scope. For your reference, here is the model:
\`\`\`
{manifest_source}
\`\`\`

* Assume a Pydantic model called \`JobOutput(BaseModel)\` is already in global scope. For your reference, here is the model:
\`\`\`
{output_source}
\`\`\`

* DO NOT rewrite or import the model in your code.

* Function #1 signature will look like:
\`\`\`
{signature_source}
\`\`\`

* Function #2 signature will look like:
\`\`\`
{transform_signature_source}
\`\`\`

* You can assume you have access to the following chunking function(s). Do not reimplement the function, just use it.
\`\`\`
{chunking_source}
\`\`\`

# Here is an example
\`\`\`python
def prepare_jobs(
    context: List[str],
    prev_job_manifests: Optional[List[JobManifest]] = None,
    prev_job_outputs: Optional[List[JobOutput]] = None,
) -> List[JobManifest]:
    task_id = 1  # Unique identifier for the task

    # iterate over the previous job outputs because "scratchpad" tells me they contain useful information
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
    tasks = {}
    for job in jobs:
        task_id = job.manifest.task_id
        chunk_id = job.manifest.chunk_id
        
        if task_id not in tasks:
            tasks[task_id] = {
                "task_id": task_id,
                "task": job.manifest.task,
                "chunks": {},
            }
        
        if chunk_id not in tasks[task_id]["chunks"]:
            tasks[task_id]["chunks"][chunk_id] = []
        
        tasks[task_id]["chunks"][chunk_id].append(job)
    
    # Build the aggregated string
    aggregated_str = ""
    for task_id, task_info in tasks.items():
        aggregated_str += f"## Task (task_id=\`{task_id}\`): {task_info['task']}\n\n"
        
        for chunk_id, chunk_jobs in task_info["chunks"].items():
            filtered_jobs = [j for j in chunk_jobs if j.include]
            
            aggregated_str += f"### Chunk # {chunk_id}\n"
            if filtered_jobs:
                for idx, job in enumerate(filtered_jobs, start=1):
                    aggregated_str += f"   -- Job {idx} (job_id=\`{job.manifest.job_id}\`):\n"
                    aggregated_str += f"   {job.sample}\n\n"
            else:
                aggregated_str += "   No jobs returned successfully for this chunk.\n\n"
        
        aggregated_str += "\n-----------------------\n\n"
    
    return aggregated_str
\`\`\``; 