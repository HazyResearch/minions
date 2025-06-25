from typing import List, Any, Optional, Dict, Union, Callable
import concurrent.futures
import json
from pydantic import BaseModel
import os

from minions.minions import Minions, chunk_by_section
from minions.utils.firecrawl_util import scrape_url
from minions.utils.serpapi_util import get_web_urls
from minions.prompts.minions_deep_research import WORKER_SUMMARIZE_PROMPT, INITIAL_QUERY_PROMPT, ASSESSMENT_PROMPT, FINAL_SYNTHESIS_PROMPT
from minions.clients import LemonadeClient
class JobManifest(BaseModel):
    """
    Represents a job manifest for a Minions Deep Researchtask
    """
    chunk: str
    chunk_id: int
    task_id: int
    job_id: int

class JobOutput(BaseModel):
    """
    Represents the output of a Minions Deep Research task
    """
    explanation: str
    answer: str

class AssessmentOutput(BaseModel):
    """
    Represents the output of a Minions Deep Research assessment task
    """
    more_info_required: bool
    search_query: Optional[str] = None



class DeepResearchMinions:
    """
    Deep Research version of Minions that uses web search to gather context
    """
    def __init__(self, local_client=None, remote_client=None, max_rounds=3, callback=None, max_sources_per_round=5):
        """
        Initialize the DeepResearchMinions class
        Args:
            local_client: Local client for the Minions protocol
            remote_client: Remote client for the Minions protocol
            max_rounds: Maximum number of rounds
            callback: Optional callback function to receive message updates (not used in this implementation)
            max_sources_per_round: Maximum number of sources per round
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.max_rounds = max_rounds
        self.callback = callback
        self.max_sources_per_round = max_sources_per_round
        self.worker_batch_size = 10

    def extract_metadata(self, query: str) -> List[str]:
        """
        Extracts the metadata for a given query using the firecrawl API and SERPAPI utility functions
        Returns:
            tuple: (metadata_list, execution_time_in_seconds)
        """
        
        urls = get_web_urls(query, num_urls=self.max_sources_per_round)
        if not urls:
            return []

        metadata = []

        # Parallelize the scraping of the urls
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(urls))) as executor:
            futures_to_url = {executor.submit(scrape_url, url): url for url in urls}

            for future in concurrent.futures.as_completed(futures_to_url):
                url = futures_to_url[future]
                try:
                    result = future.result()
                    if result and result.get("markdown"):
                        metadata.append(result.get("markdown"))
                except Exception as e:
                    metadata.append(f"Contents not found for {url}")

        
        return urls,metadata

    def summarize_metadata(self, query: str, metadata: List[str]) -> List[Dict[str, Any]]:
        """
        Summarizes the metadata for a given query using the local model
        Args:
            query: The query to summarize the metadata for
            metadata: The metadata to summarize
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the summary of the metadata
        """
        if not metadata:
            return []
        
        job_manifests = []
        for source_id, content in enumerate(metadata):
            if not content:
                continue
            
            chunks = chunk_by_section(content, max_chunk_size=3000, overlap=50)
            for chunk_id, chunk in enumerate(chunks):
                if len(chunk) < 100:
                    continue
                
                job_manifest = JobManifest(
                    chunk=chunk,
                    chunk_id=chunk_id,
                    task_id=source_id,
                    job_id=len(job_manifests)
                )
                job_manifests.append(job_manifest)

        if not job_manifests:
            return []

        worker_messages = []
        for job_manifest in job_manifests:
            prompt = WORKER_SUMMARIZE_PROMPT.format(query=query, content=job_manifest.chunk)
            worker_messages.append({"role": "user", "content": prompt})
        
        summaries = []
        batch_size = min(self.worker_batch_size, len(job_manifests))

        for i in range(0, len(job_manifests), batch_size):
            batch = worker_messages[i:i+batch_size]
            batch_manifests = job_manifests[i:i+batch_size]

            worker_responses, usage, done_reasons = self.local_client.chat(batch)

            for worker_response, done_reason, job_manifest in zip(worker_responses, done_reasons, batch_manifests):
                try:
                    if done_reason == "stop":
                        job_output = JobOutput.model_validate_json(worker_response)
                        is_relevant = job_output.answer == "relevant"
                        summary = {
                            "source_id": job_manifest.task_id,
                            "chunk_id": job_manifest.chunk_id,
                            "summary": job_output.explanation,
                            "raw_chunk": job_manifest.chunk,
                            "relevant": is_relevant,
                        }
                        summaries.append(summary)
                except Exception as e:
                    print(f"[Deep Research Minions] [Summarize Metadata] Error processing worker response: {e}")
                    summary = {
                        "source_id": job_manifest.task_id,
                        "chunk_id": job_manifest.chunk_id,
                        "summary": f"Error processing response: {str(e)}",
                        "raw_chunk": job_manifest.chunk,
                        "relevant": False,
                    }
                    summaries.append(summary)
            
            print(f"[Deep Research Minions] [Summarize Metadata] Processing batch {i//batch_size + 1}/{len(job_manifests)//batch_size + 1}")
        
        relevant_summaries = [s for s in summaries if s.get('relevant')]
        return relevant_summaries
    
    def format_summaries(self, query_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Formats the summaries into a string for the remote client, organized by search query
        
        Args:
            query_results: Dictionary mapping search queries to their summary results
            
        Returns:
            Formatted string with search queries and their results
        """
        formatted_sections = []
        
        for query, summaries in query_results.items():
            # Add the search query as a section header
            formatted_sections.append(f"SEARCH QUERY: {query}")
            
            if summaries:
                # Simply list all summaries
                summary_lines = []
                for summary in summaries:
                    content = summary.get("summary", "").strip()
                    if content:
                        summary_lines.append(f"Source {summary.get('source_id', 0) + 1}: {content}")
                
                formatted_sections.append("\n".join(summary_lines))
            else:
                formatted_sections.append("No information found for this query.")
            
            formatted_sections.append("----------")
        
        return "\n\n".join(formatted_sections)
     
    def assess_and_generate_query(self, query: str, query_results: Dict[str, List[Dict[str, Any]]]=None) -> AssessmentOutput:
        """
        Assesses the need for more information and generates a new query if needed
        Args:
            query: The original user query
            summaries: List of summaries from previous rounds
        Returns:
            AssessmentOutput: Object containing should_continue and search_query
        """
        try:
            if not query_results:
                messages = [{
                    "role": "user",
                    "content": INITIAL_QUERY_PROMPT.format(query=query)
                }]
                response, _ = self.remote_client.chat(
                    messages=messages,
                )
                response_object = {
                    "more_info_required": "True",
                    "search_query": response[0]
                }
                return AssessmentOutput.model_validate_json(json.dumps(response_object))
            else:
                formatted_summaries = self.format_summaries(query_results)
                messages = [{
                    "role": "user",
                    "content": ASSESSMENT_PROMPT.format(
                        query=query,
                        information=formatted_summaries
                    )
                }]
                response, _ = self.remote_client.chat(
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                return AssessmentOutput.model_validate_json(response[0])
                        

        except Exception as e:
            print(f"[Deep Research Minions] [Assess and Generate Query] Error in assess_and_generate_query: {e}")
            # Return a default AssessmentOutput indicating we should stop
            return AssessmentOutput(
                more_info_required=True,
                search_query=None,
            )

    def __call__(
        self,
        query: str,
        firecrawl_api_key: Optional[str] = None,
        serpapi_key: Optional[str] = None,
        max_rounds: Optional[int] = None,
        max_sources_per_round: Optional[int] = None,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Main method to execute the Deep Research Minions protocol
        Args:
            query: The original user query
            firecrawl_api_key: Optional API key for Firecrawl
            serpapi_key: Optional API key for SERPAPI
            max_rounds: Optional maximum number of research rounds
            max_sources_per_round: Optional maximum sources per round
            callback: Optional callback function for progress updates
        Returns:
            str: The final research response
        """
        # Update instance variables if provided
        if max_rounds is not None:
            self.max_rounds = max_rounds
        if max_sources_per_round is not None:
            self.max_sources_per_round = max_sources_per_round
        if callback is not None:
            self.callback = callback

        # Set API keys if provided
        if firecrawl_api_key:
            os.environ['FIRECRAWL_API_KEY'] = firecrawl_api_key
        if serpapi_key:
            os.environ['SERPAPI_API_KEY'] = serpapi_key

        current_round = 0
        query_results = {}
        visited_urls = set()

        assessment = self.assess_and_generate_query(query)
        while assessment.more_info_required and current_round < self.max_rounds:
            if self.callback:
                if assessment.search_query:
                    self.callback("supervisor", f"🔍 {assessment.search_query}")
            
            read_urls, metadata = self.extract_metadata(assessment.search_query)
            visited_urls.update(read_urls)
            summaries = self.summarize_metadata(assessment.search_query, metadata)
            if self.callback:
                if summaries:
                    # Create a summary of all sources
                    preview_text = f"📚 Found {len(summaries)} relevant sources:\n\n"
                    
                    # Add preview for each source (limited to first 5 to avoid too much text)
                    for i, summary in enumerate(summaries[:5], 1):
                        preview = summary['summary'][:100] + "..." if summary['summary'] else "No summary available"
                        preview_text += f"Source {i}:\n```\n{preview}\n```\n\n"
                    
                    # If there are more sources, add a note
                    if len(summaries) > 5:
                        preview_text += f"... and {len(summaries) - 5} more sources"
                    
                    self.callback("Worker", preview_text)
                else:
                    self.callback("Worker", "⚠️ No relevant sources found in this search.")
            query_results[assessment.search_query] = summaries

            assessment = self.assess_and_generate_query(query, query_results)
            current_round += 1

        if self.callback:
            self.callback("Worker", " 📃 Combining sources...", is_final=True)

        formatted_summaries = self.format_summaries(query_results)
        prompt = FINAL_SYNTHESIS_PROMPT.format(query=query, information=formatted_summaries)
        try: 
            responses, usage, done_reasons = self.local_client.chat([{"role": "user", "content": prompt}])
            final_response = json.loads(responses[0])["explanation"] if isinstance(self.local_client, LemonadeClient) else responses[0]
            return final_response, visited_urls
        except Exception as e:
            print(f"[Deep Research Minions] [Synthesize Final Response] Error in synthesize_final_response: {e}")
            return "An error occurred while synthesizing the final response.", []