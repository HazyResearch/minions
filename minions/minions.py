from typing import List, Dict, Any, Optional, Union, Tuple
import json
import re
import os
import json
import os
import time
from datetime import datetime
from pydantic import BaseModel, field_validator, Field
from inspect import getsource

from minions.utils.multimodal_retrievers import (
    retrieve_chunks_from_chroma,
)

from minions.usage import Usage

from minions.utils.chunking import (
    chunk_by_section,
    chunk_by_page,
    chunk_sentences,
    chunk_by_paragraph,
    extract_imports,
    extract_function_header,
    extract_function,
    chunk_by_code,
    chunk_by_function_and_class,
)


from minions.prompts.minions import (
    WORKER_PROMPT_TEMPLATE,
    WORKER_OUTPUT_TEMPLATE,
    WORKER_ICL_EXAMPLES,
    WORKER_PROMPT_SHORT,
    ADVICE_PROMPT,
    ADVICE_PROMPT_STEPS,
    DECOMPOSE_TASK_PROMPT,
    DECOMPOSE_TASK_PROMPT_SHORT_JOB_OUTPUTS,
    REMOTE_ANSWER_OR_CONTINUE,
    REMOTE_ANSWER_OR_CONTINUE_SHORT,
    REMOTE_ANSWER,
    DECOMPOSE_TASK_PROMPT_AGGREGATION_FUNC,
    DECOMPOSE_TASK_PROMPT_AGG_FUNC_LATER_ROUND,
    DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGGREGATION_FUNC,
    DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGG_FUNC_LATER_ROUND,
    REMOTE_SYNTHESIS_COT,
    REMOTE_SYNTHESIS_JSON,
    REMOTE_SYNTHESIS_FINAL,
    BM25_INSTRUCTIONS,
    EMBEDDING_INSTRUCTIONS,
)

from minions.utils.retrievers import *


def chunk_by_section(
    doc: str, max_chunk_size: int = 3000, overlap: int = 20
) -> List[str]:
    sections = []
    start = 0
    while start < len(doc):
        end = start + max_chunk_size
        sections.append(doc[start:end])
        start += max_chunk_size - overlap
    return sections


class JobManifest(BaseModel):
    chunk: str  # the actual text for the chunk of the document
    task: str  # the actual task instruction for the small model
    advice: str  # optional, any additional advice on how to perform the task

    chunk_id: Optional[int] = (
        None  # you do NOT need to set this, it will be handled automatically
    )
    task_id: Optional[int] = (
        None  # you do NOT need to set this, it will be handled automatically
    )
    job_id: Optional[int] = (
        None  # you do NOT need to set this, it will be handled automatically
    )


class JobOutput(BaseModel):
    explanation: str
    citation: Optional[str]
    answer: Optional[str]


def prepare_jobs(
    context: List[str],
    prev_job_manifests: Optional[List[JobManifest]] = None,
    prev_job_outputs: Optional[List[JobOutput]] = None,
) -> List[JobManifest]:
    """
    Args:
        context (List[str]): A list of documents. Assume each document is greater >100k tokens.
            Each document can be further chunked using `chunk_pages`.
            If context is empty, use the MCP functions to get information that you need to complete your task: i.e., ```context = mcp_tools.execute_tool(....)```
        prev_job_manifests (Optional[List[JobManifest]]): A list of job manifests from the previous round.
            None if we are on the first round.
        prev_job_outputs (Optional[List[JobOutput]]): A list of job outputs from the previous round.
            None if we are on the first round.
    Returns:
        List[JobManifest]: A list of job manifests for the current round.
    """
    ...


class Job(BaseModel):
    """
    An object for us to filter job manifests. not seen by the worker or used in the code block.
    """

    manifest: JobManifest
    output: JobOutput
    sample: str  # this is the raw client sample
    include: Optional[bool] = None


def transform_outputs(
    jobs: List[Job],
) -> str:
    """
    Args:
        jobs (List[Job]): A list of jobs from the workers.
    Returns:
        str: A transformed view of all the job outputs (including answer, citation + explanation) that will be analyzed to make a final decision. Make sure to use **as much** information from the outputs as possible in final aggregated str (output.answer, output.sample, output.explanation, output.citation)

        Note: Job has following attributes:
        - manifest: JobManifest(chunk, task, advice, chunk_id, task_id, job_id)
        - sample: entire response from the worker
        - output: JobOutput(answer="". explanation="", citation="", raw="")
    """
    ...


# these objects are passed to the exec_globals so the code block can use them without
# having to import them itself
USEFUL_IMPORTS = {
    "List": List,
    "Optional": Optional,
    "Dict": Dict,
    "Any": Any,
    "Union": Union,
    "Tuple": Tuple,
    "BaseModel": BaseModel,
    "field_validator": field_validator,
}


class Minions:
    def __init__(
        self,
        local_client=None,
        remote_client=None,
        max_rounds=5,
        callback=None,
        log_dir="minions_logs",
        **kwargs,
    ):
        """Initialize the Minion with local and remote LLM clients.

        Args:
            local_client: Client for the local model (e.g. OllamaClient)
            remote_client: Client for the remote model (e.g. OpenAIClient)
            max_rounds: Maximum number of conversation rounds
            callback: Optional callback function to receive message updates
            log_dir: Directory for logging conversation history
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.max_rounds = max_rounds
        self.max_jobs_per_round = 2048
        self.callback = callback
        self.log_dir = log_dir
        self.num_samples = 1 or kwargs.get("num_samples", None)
        self.worker_batch_size = 1 or kwargs.get("worker_batch_size", None)
        self.max_code_attempts = kwargs.get("max_code_attempts", 10)
        # TODO: removed worker_prompt
        self.worker_prompt_template = WORKER_PROMPT_SHORT or kwargs.get(
            "worker_prompt_template", None
        )
        self.worker_icl_examples = WORKER_ICL_EXAMPLES or kwargs.get(
            "worker_icl_examples", None
        )
        self.worker_icl_messages = []
        self.advice_prompt = ADVICE_PROMPT or kwargs.get("advice_prompt", None)

        self.decompose_task_prompt = (
            kwargs.get("decompose_task_prompt", None)
            or DECOMPOSE_TASK_PROMPT_AGGREGATION_FUNC
        )
        self.decompose_task_prompt_abbreviated = (
            kwargs.get("decompose_task_prompt_abbreviated", None)
            or DECOMPOSE_TASK_PROMPT_AGG_FUNC_LATER_ROUND
        )
        self.decompose_retrieval_task_prompt = (
            kwargs.get("decompose_retrieval_task_prompt", None)
            or DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGGREGATION_FUNC
        )
        self.decompose_retrieval_task_prompt_abbreviated = (
            kwargs.get("decompose_retrieval_task_prompt_abbreviated", None)
            or DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGG_FUNC_LATER_ROUND
        )
        self.synthesis_cot_prompt = REMOTE_SYNTHESIS_COT or kwargs.get(
            "synthesis_cot_prompt", None
        )
        self.synthesis_json_prompt = REMOTE_SYNTHESIS_JSON or kwargs.get(
            "synthesis_json_prompt", None
        )
        self.synthesis_final_prompt = REMOTE_SYNTHESIS_FINAL or kwargs.get(
            "synthesis_final_prompt", None
        )
        self.chunking_fns = {
            "chunk_by_section": chunk_by_section,
            "chunk_by_page": chunk_by_page,
            "chunk_by_paragraph": chunk_by_paragraph,
            "extract_imports": extract_imports,
            "extract_function_header": extract_function_header,
            "extract_function": extract_function,
            "chunk_by_code": chunk_by_code,
            "chunk_by_function_and_class": chunk_by_function_and_class,
        }
        self.chunking_fn = chunk_by_section
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

    def _execute_code(
        self,
        code: str,
        starting_globals: Dict[str, Any] = {},
        fn_name: str = "prepare_jobs",
        **kwargs,
    ) -> Tuple[Any, str]:
        exec_globals = {
            **starting_globals
        }  # dictionary to store variables in the code block
        exec(code, exec_globals)  # first execution, with example usage
        if fn_name not in exec_globals:
            raise ValueError(f"Function {fn_name} not found in the code block.")
        output = exec_globals[fn_name](
            **kwargs
        )  # by default, grab the prepare_jobs function, execute it with the kwargs, i.e., context

        # call exec_globsl (filter_fnf)
        return output, code

    def __call__(
        self,
        task: str,
        doc_metadata: str,
        context: List[str],
        max_rounds=None,
        max_jobs_per_round=None,
        num_tasks_per_round=3,
        num_samples_per_task=1,
        mcp_tools_info=None,
        use_retrieval=None,
        log_path=None,
        logging_id=None,
        retrieval_model=None,
        chunk_fn="chunk_by_section",
    ):
        """Run the minions protocol to answer a task using local and remote models.

        Args:
            task: The task/question to answer
            doc_metadata: Type of document being analyzed
            context: List of context strings
            max_rounds: Override default max_rounds if provided
            max_jobs_per_round: Override default max_jobs_per_round if provided
            retrieval: Retrieval strategy to use. Options:
                - None: Don't use retrieval
                - "bm25": Use BM25 keyword-based retrieval
                - "embedding": Use embedding-based retrieval
            log_path: Optional path to save conversation logs

        Returns:
            Dict containing final_answer and conversation histories
        """

        self.chunking_fn = self.chunking_fns[chunk_fn]

        # Initialize timing metrics
        start_time = time.time()
        timing = {
            "local_call_time": 0.0,
            "remote_call_time": 0.0,
            "total_time": 0.0,
        }

        print("\n========== MINIONS TASK STARTED ==========")
        print(f"Task: {task}")
        print(f"Max rounds: {max_rounds or self.max_rounds}")
        print(f"Retrieval: {use_retrieval}")

        self.max_rounds = max_rounds or self.max_rounds
        self.max_jobs_per_round = max_jobs_per_round or self.max_jobs_per_round

        # Initialize the log structure
        conversation_log = {
            "task": task,
            "doc_metadata": doc_metadata,
            "conversation": [],
            "generated_final_answer": "",
            "usage": {
                "remote": {},
                "local": {},
            },
            "timing": timing,
        }

        # Initialize usage tracking
        remote_usage = Usage()
        local_usage = Usage()

        retriever = None

        if use_retrieval:
            if use_retrieval == "bm25":
                retriever = bm25_retrieve_top_k_chunks
            elif use_retrieval == "embedding":
                retriever = embedding_retrieve_top_k_chunks
            elif use_retrieval == "multimodal-embedding":
                retriever = retrieve_chunks_from_chroma

        # 1. [REMOTE] ADVICE --- Read the query with big model and provide advice
        # ---------- START ----------
        supervisor_messages = [
            {
                "role": "user",
                "content": self.advice_prompt.format(query=task, metadata=doc_metadata),
            },
        ]

        # Add initial supervisor prompt to conversation log
        conversation_log["conversation"].append(
            {
                "user": "supervisor",
                "prompt": self.advice_prompt.format(query=task, metadata=doc_metadata),
                "output": None,
            }
        )

        if self.callback:
            self.callback("supervisor", None, is_final=False)

        remote_start_time = time.time()
        advice_response, usage = self.remote_client.chat(
            supervisor_messages,
        )
        current_time = time.time()
        timing["remote_call_time"] += current_time - remote_start_time

        remote_usage += usage

        supervisor_messages.append(
            {"role": "assistant", "content": advice_response[0]},
        )

        # Update conversation log with response
        conversation_log["conversation"][-1]["output"] = advice_response[0]

        if self.callback:
            self.callback("supervisor", supervisor_messages[-1], is_final=True)
        # ---------- END ----------

        last_jobs: Optional[List[Job]] = None
        feedback: Optional[str] = None
        scratchpad: str = ""
        meta: List[Dict[str, any]] = []
        final_answer: Optional[str] = None

        for round_idx in range(self.max_rounds):
            print(f"Round {round_idx + 1}/{self.max_rounds}")
            try:
                total_chars = int(
                    doc_metadata.split("Total extracted text length: ")[1].split(
                        " characters"
                    )[0]
                )
            except:
                # compute characters in context
                total_chars = sum(len(doc) for doc in context)

            retrieval_source = ""
            retrieval_instructions = ""

            if use_retrieval:
                retrieval_source = getsource(retriever).split("    weights = ")[0]

                retrieval_instructions = (
                    BM25_INSTRUCTIONS
                    if use_retrieval == "bm25"
                    else EMBEDDING_INSTRUCTIONS if use_retrieval == "embedding" else ""
                )

            print(getsource(self.chunking_fn))
            decompose_message_kwargs = dict(
                num_samples=self.num_samples,
                ADVANCED_STEPS_INSTRUCTIONS="",
                manifest_source=getsource(JobManifest),
                output_source=getsource(JobOutput),
                signature_source=getsource(prepare_jobs),
                transform_signature_source=getsource(transform_outputs),
                # read_file_source=getsource(read_folder),
                chunking_source="\n\n".join(
                    [getsource(self.chunking_fn).split("    sections = ")[0]]
                ),
                retrieval_source=retrieval_source,
                retrieval_instructions=retrieval_instructions,
                num_tasks_per_round=num_tasks_per_round,
                num_samples_per_task=num_samples_per_task,
                total_chars=total_chars,
            )

            decompose_prompt = (
                self.decompose_task_prompt
                if not use_retrieval
                else self.decompose_retrieval_task_prompt
            )
            # create the decompose prompt -- if in later rounds, use a shorter version
            decompose_message = {
                "role": "user",
                "content": decompose_prompt.format(
                    step_number=1,
                    mcp_tools_info=mcp_tools_info,
                    **decompose_message_kwargs,
                ),
            }

            if round_idx == 0:
                supervisor_messages.append(decompose_message)
            else:
                decompose_prompt_abbrev = (
                    self.decompose_task_prompt_abbreviated
                    if not use_retrieval
                    else self.decompose_retrieval_task_prompt_abbreviated
                )
                if feedback is not None:
                    decompose_message = {
                        "role": "user",
                        "content": decompose_prompt_abbrev.format(
                            step_number=round_idx + 1,
                            feedback=feedback,
                            scratchpad=scratchpad,
                            mcp_tools_info=mcp_tools_info,
                            **decompose_message_kwargs,
                        ),
                    }
                supervisor_messages = supervisor_messages[:2] + [decompose_message]

            # Add decompose prompt to conversation log
            conversation_log["conversation"].append(
                {
                    "user": "supervisor",
                    "prompt": decompose_message["content"],
                    "output": None,
                }
            )

            # 2. [REMOTE] PREPARE TASKS --- Prompt the supervisor to write code
            # ---------- START ----------
            for attempt_idx in range(self.max_code_attempts):
                print(f"Attempt {attempt_idx + 1}/{self.max_code_attempts}")

                if self.callback:
                    self.callback("supervisor", None, is_final=False)

                remote_start_time = time.time()
                task_response, usage = self.remote_client.chat(
                    messages=supervisor_messages,
                )
                current_time = time.time()
                timing["remote_call_time"] += current_time - remote_start_time

                remote_usage += usage

                task_response = task_response[0]
                print(task_response)
                supervisor_messages.append(
                    {"role": "assistant", "content": task_response},
                )

                # Update conversation log with response
                conversation_log["conversation"][-1]["output"] = task_response

                if self.callback:
                    self.callback("supervisor", supervisor_messages[-1], is_final=True)

                code_block_match = re.search(
                    r"```(?:python)?\s*(.*?)```",
                    task_response,
                    re.DOTALL,
                )

                if code_block_match:
                    code_block = code_block_match.group(1).strip()
                else:
                    print(f"No code block found in the supervisor response.")
                    supervisor_messages.append(
                        {
                            "role": "user",
                            "content": f"Please try again. No code block found in the supervisor response.",
                        }
                    )
                    continue

                # prepare the inputs for the code execution
                starting_globals = {
                    **USEFUL_IMPORTS,
                    "chunk_by_section": chunk_by_section,
                    f"{chunk_fn}": self.chunking_fn,
                    "JobManifest": JobManifest,
                    "JobOutput": JobOutput,
                    "Job": Job,
                }

                if use_retrieval:
                    starting_globals[f"{use_retrieval}_retrieve_top_k_chunks"] = (
                        retriever
                    )

                fn_kwargs = {
                    "context": context,
                    "prev_job_manifests": (
                        [job.manifest for job in last_jobs]
                        if last_jobs is not None
                        else None
                    ),
                    "prev_job_outputs": (
                        [job.output for job in last_jobs]
                        if last_jobs is not None
                        else None
                    ),
                }
                try:
                    job_manifests, compiled_code_block = self._execute_code(
                        code_block,
                        starting_globals=starting_globals,
                        fn_name="prepare_jobs",  # the global variable to extract from the code block
                        **fn_kwargs,
                    )

                    # We need to coerce the type below to ensure that the type is
                    # not a different `JobManifest` object the model defined in it's
                    # own code. We also need to set the chunk_id and task_id.
                    chunk_ids, task_ids = {}, {}
                    job_manifests = [
                        JobManifest(
                            chunk=job_manifest.chunk,
                            task=job_manifest.task,
                            advice=job_manifest.advice,
                            chunk_id=chunk_ids.setdefault(
                                job_manifest.chunk, len(chunk_ids)
                            ),
                            task_id=task_ids.setdefault(
                                job_manifest.task, len(task_ids)
                            ),
                            job_id=job_id,
                        )
                        for job_id, job_manifest in enumerate(job_manifests)
                    ]

                    if len(job_manifests) > self.max_jobs_per_round:
                        print(
                            f"Exceeded max jobs per round: {len(job_manifests)} > {self.max_jobs_per_round}. Trying again."
                        )
                        supervisor_messages.append(
                            {
                                "role": "user",
                                "content": f"Your code is output {len(job_manifests)} jobs which exceeds the max jobs per round ({self.max_jobs_per_round}). Please try again.",
                            }
                        )

                        # Log this error to conversation log
                        conversation_log["conversation"].append(
                            {
                                "user": "supervisor",
                                "prompt": f"Your code is output {len(job_manifests)} jobs which exceeds the max jobs per round ({self.max_jobs_per_round}). Please try again.",
                                "output": None,
                            }
                        )

                        continue
                    print(
                        f"Created {len(job_manifests)} job manifests ({len(chunk_ids)} chunks, apriori requested {self.num_samples} samples per chunk, {len(task_ids)} tasks)"
                    )
                    break
                except Exception as e:
                    print(
                        f"Error executing code (attempt {attempt_idx} of {self.max_code_attempts} max attempts): {type(e).__name__}: {e}"
                    )

                    error_message = f"Please try again. I got this error when executing the code: \n\n```{type(e).__name__}: {e}```"
                    supervisor_messages.append(
                        {
                            "role": "user",
                            "content": error_message,
                        }
                    )

                    # Log this error to conversation log
                    conversation_log["conversation"].append(
                        {
                            "user": "supervisor",
                            "prompt": error_message,
                            "output": None,
                        }
                    )
            else:
                # if we have exhausted all attempts, break
                print(
                    f"Exhausted all attempts to execute code. Breaking out of round loop."
                )
                break
            # --------- END ---------

            # 3. [REMOTE] LOCAL WORKERS EXECUTE TASKS
            # ---------- START ----------
            worker_chats = []
            # output is a list of task_dicts
            # print total number of job_manfiests
            print(f"Total number of job_manifests: {len(job_manifests)}")
            for job_manifest in job_manifests:
                # Each worker is going to see a unique task+chunk combo
                # removed the external list
                worker_messages = {
                    "role": "user",
                    "content": self.worker_prompt_template.format(
                        context=job_manifest.chunk,
                        task=job_manifest.task,
                        advice=job_manifest.advice,
                    ),
                }
                worker_chats.append(worker_messages)

            if self.callback:
                self.callback("worker", None, is_final=False)

            print(f"Sending {len(worker_chats)} worker chats to the worker client")

            # Add worker tasks to conversation log
            conversation_log["conversation"].append(
                {
                    "user": "worker",
                    "prompt": f"Sending {len(worker_chats)} worker chats",
                    "output": None,
                    "job_manifests": [job.model_dump() for job in job_manifests],
                }
            )

            local_start_time = time.time()
            worker_response, usage, done_reasons = self.local_client.chat(
                worker_chats,
            )
            current_time = time.time()
            timing["local_call_time"] += current_time - local_start_time

            local_usage += usage

            def extract_job_output(response: str) -> JobOutput:
                output = JobOutput.model_validate_json(response)
                return output

            jobs: List[Job] = []
            for worker_messages, sample, job_manifest, done_reason in zip(
                worker_chats, worker_response, job_manifests, done_reasons
            ):
                if done_reason == "length":
                    job_output = JobOutput(
                        answer=None,
                        explanation="The model returned a truncated response. Please try again.",
                        citation=None,
                    )
                    continue
                elif done_reason == "stop":
                    job_output = extract_job_output(response=sample)
                else:
                    raise ValueError(f"Unknown done reason: {done_reason}")
                jobs.append(
                    Job(
                        manifest=job_manifest,
                        sample=sample,
                        output=job_output,
                    )
                )

            fn_kwargs = {
                "jobs": jobs,
            }
            if self.callback:
                self.callback("worker", jobs, is_final=True)

            # Update conversation log with worker responses
            conversation_log["conversation"][-1]["output"] = [
                job.sample for job in jobs
            ]
            conversation_log["conversation"][-1]["job_outputs"] = [
                {
                    "job_id": job.manifest.job_id,
                    "chunk_id": job.manifest.chunk_id,
                    "task_id": job.manifest.task_id,
                    "output": job.output.model_dump(),
                }
                for job in jobs
            ]

            try:
                # Model generated Filter + Aggregation code
                for job in jobs:
                    print(job.output.answer)

                aggregated_str, code_block = self._execute_code(
                    code_block,
                    starting_globals=starting_globals,
                    fn_name="transform_outputs",  # the global variable to extract from the code block
                    **fn_kwargs,
                )

            except Exception as e:
                # Log exception
                print(f"Error executing transformation code: {type(e).__name__}: {e}")
                conversation_log["conversation"].append(
                    {
                        "user": "supervisor",
                        "prompt": f"Error executing transformation code: {type(e).__name__}: {e}",
                        "output": None,
                    }
                )

                # 4. [EDGE] FILTER
                # ---------- START ----------
                def filter_fn(job: Job) -> bool:
                    answer = job.output.answer
                    if answer is None or str(answer).lower().strip() == "none":
                        return False
                    return True

                for job in jobs:
                    job.include = filter_fn(job)

                print(
                    f"After filtering, {sum(job.include for job in jobs)}/{len(jobs)} jobs were included"
                )
                # ---------- END ----------

                # 5. [REMOTE] AGGREGATE AND FILTER --- Synthesize the results from the worker models
                # ---------- START ----------
                tasks = {}
                for job in jobs:
                    # 1. Create a container for each task_id if it doesn't exist yet.
                    if job.manifest.task_id not in tasks:
                        tasks[job.manifest.task_id] = {
                            "task_id": job.manifest.task_id,
                            "task": job.manifest.task,  # <-- Store the actual task string here
                            "chunks": {},  # <-- We'll group by chunk_id next
                        }

                    # 2. For the given task_id, group by chunk_id
                    c_id = job.manifest.chunk_id
                    if c_id not in tasks[job.manifest.task_id]["chunks"]:
                        tasks[job.manifest.task_id]["chunks"][c_id] = []

                    tasks[job.manifest.task_id]["chunks"][c_id].append(job)

                # Step 2: Build the string to pass to the big model,
                # grouping by task first and then by chunk.
                aggregated_str = ""
                for task_id, task_info in tasks.items():
                    aggregated_str += (
                        f"## Task (task_id=`{task_id}`): {task_info['task']}\n\n"
                    )
                    # task_info['task'] is the string you saved above.

                    # Inside each task, go chunk by chunk.
                    for chunk_id, chunk_jobs in task_info["chunks"].items():
                        # Filter out any jobs that failed or are flagged "include=False".
                        filtered_jobs = [j for j in chunk_jobs if j.include]

                        if filtered_jobs:
                            aggregated_str += f"### Chunk # {chunk_id}\n"
                            for idx, job in enumerate(filtered_jobs, start=1):
                                aggregated_str += f"   -- Job {idx} (job_id=`{job.manifest.job_id}`):\n"
                                aggregated_str += f"   {job.sample}\n\n"
                        else:
                            aggregated_str += f"### Chunk # {chunk_id}\n"
                            aggregated_str += (
                                "   No jobs returned successfully for this chunk.\n\n"
                            )
                    # Separate tasks with a short delimiter
                    aggregated_str += "\n-----------------------\n\n"

            if round_idx == self.max_rounds - 1:
                # Final round - use the final prompt directly
                final_prompt = self.synthesis_final_prompt.format(
                    extractions=aggregated_str,
                    question=task,
                    scratchpad=(scratchpad if scratchpad else "No previous progress."),
                )
                supervisor_messages.append(
                    {
                        "role": "user",
                        "content": final_prompt,
                    }
                )

                # Add synthesis prompt to conversation log
                conversation_log["conversation"].append(
                    {
                        "user": "supervisor",
                        "prompt": final_prompt,
                        "output": None,
                    }
                )
            else:
                # First step: Think through the synthesis
                cot_prompt = self.synthesis_cot_prompt.format(
                    extractions=aggregated_str,
                    question=task,
                    scratchpad=(scratchpad if scratchpad else "No previous progress."),
                )
                supervisor_messages.append(
                    {
                        "role": "user",
                        "content": cot_prompt,
                    }
                )

                # Add COT prompt to conversation log
                conversation_log["conversation"].append(
                    {
                        "user": "supervisor",
                        "prompt": cot_prompt,
                        "output": None,
                    }
                )

                remote_start_time = time.time()
                step_by_step_response, usage = self.remote_client.chat(
                    supervisor_messages,
                )
                current_time = time.time()
                timing["remote_call_time"] += current_time - remote_start_time

                remote_usage += usage
                if self.callback:
                    self.callback("supervisor", step_by_step_response[0])

                supervisor_messages.append(
                    {"role": "assistant", "content": step_by_step_response[0]}
                )

                # Update conversation log with COT response
                conversation_log["conversation"][-1]["output"] = step_by_step_response[
                    0
                ]

                # Second step: Get structured output
                json_prompt = self.synthesis_json_prompt
                supervisor_messages.append(
                    {
                        "role": "user",
                        "content": json_prompt,
                    }
                )

                # Add JSON prompt to conversation log
                conversation_log["conversation"].append(
                    {
                        "user": "supervisor",
                        "prompt": json_prompt,
                        "output": None,
                    }
                )

            # Get the structured output and validate JSON response
            max_attempts = 5
            for attempt_idx in range(max_attempts):
                try:
                    if self.callback:
                        self.callback("supervisor", None, is_final=False)
                    # Request JSON response from remote client
                    remote_start_time = time.time()
                    synthesized_response, usage = self.remote_client.chat(
                        supervisor_messages, response_format={"type": "json_object"}
                    )
                    current_time = time.time()
                    timing["remote_call_time"] += current_time - remote_start_time

                    # Parse and validate JSON response
                    response_text = synthesized_response[0]
                    print(
                        f"Attempt {attempt_idx + 1}/{max_attempts} response: {response_text}"
                    )

                    obj = json.loads(response_text)
                    if not isinstance(obj, dict) or "decision" not in obj:
                        raise ValueError("Response missing required 'decision' field")

                    # Valid JSON with decision field found
                    break

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Attempt {attempt_idx + 1}/{max_attempts} failed: {str(e)}")
                    if attempt_idx == max_attempts - 1:
                        raise ValueError(
                            f"Failed to get valid JSON response after {max_attempts} attempts"
                        )

            supervisor_messages.append(
                {"role": "assistant", "content": synthesized_response[0]}
            )

            # Update conversation log with synthesis response
            conversation_log["conversation"][-1]["output"] = synthesized_response[0]

            if self.callback:
                self.callback("supervisor", supervisor_messages[-1], is_final=True)
            # ---------- END ----------

            last_jobs = jobs

            meta.append(
                {
                    "local": {
                        "jobs": [
                            {k: v for k, v in job.model_dump().items() if k != "sample"}
                            for job in jobs
                        ]
                    },
                    "remote": {"messages": supervisor_messages},
                }
            )

            if obj["decision"] != "request_additional_info":
                final_answer = obj.get("answer", None)
                conversation_log["generated_final_answer"] = final_answer
                break  # answer was found, so we are done!
            else:
                feedback = obj.get("explanation", None)
                scratchpad = obj.get("scratchpad", None)

        if final_answer == None:
            print(
                f"Exhausted all rounds without finding a final answer. Returning the last synthesized response."
            )
            final_answer = "No answer found."
            conversation_log["generated_final_answer"] = final_answer

        # Calculate total time at the end
        end_time = time.time()
        timing["total_time"] = end_time - start_time
        timing["overhead_time"] = timing["total_time"] - (
            timing["local_call_time"] + timing["remote_call_time"]
        )

        # Add usage statistics to the conversation log
        conversation_log["usage"]["remote"] = remote_usage.to_dict()
        conversation_log["usage"]["local"] = local_usage.to_dict()

        # Save the conversation log to a file
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(conversation_log, f, indent=2, ensure_ascii=False)
        else:
            # Create a log filename based on timestamp and task or provided logging_id
            if logging_id:
                log_filename = f"{logging_id}_minions.json"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_task = re.sub(r"[^a-zA-Z0-9]", "_", task[:15])
                log_filename = f"{timestamp}_{safe_task}_minions.json"

            log_path = os.path.join(self.log_dir, log_filename)

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            # Save the log file
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(conversation_log, f, indent=2, ensure_ascii=False)

            print(f"\n=== SAVED LOG TO {log_path} ===")

        print("\n=== MINIONS TASK COMPLETED ===")

        result = {
            "final_answer": final_answer,
            "meta": meta,
            "log_file": log_path,
            "conversation_log": conversation_log,
            "timing": timing,
            "local_usage": local_usage.to_dict(),
            "remote_usage": remote_usage.to_dict(),
        }

        return result
