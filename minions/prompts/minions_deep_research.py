SEARCH_QUERY_PROMPT = """
Based on the user's original question and any information gathered so far, generate an optimal search query
that can be passed to a search engine to find the most relevant information to answer the question.

Original query: {query}

{previous_info_text}

Your task is to create a search query that will find the most relevant information to answer the question.
- Focus on technical terms and specific concepts
- Target authoritative sources
- Be specific enough to find relevant information but not so narrow that important context is missed
- If we have partial information already, focus the query on filling the gaps

Return only the search query, without any explanation or additional text.
"""

WORKER_SUMMARIZE_PROMPT = """
Your job is to summarize the following content related to this query: {query}

## Content
{content}

Provide a concise summary of the key information related to the query.
Focus on extracting the most relevant facts, concepts, and explanations.
If the content is not relevant to the query, state why it's not relevant.

Return your summary in JSON format with these fields:
- "explanation": Your concise summary of the relevant information. The summary should cover all the important aspects of the content and be at least 2 sentences.
- "answer": "relevant" if this content is relevant to the query or covers an aspect related to the query, "not relevant" otherwise

Remember to only RETURN the JSON object, nothing else.
"""

INITIAL_QUERY_PROMPT = """
You are a search engine optimization expert. Your task is to transform the user's question into ONE optimal search query.

USER QUESTION: "{query}"

Create a single search query that will:
1. Find the most relevant information to answer this question
2. Be formatted optimally for a search engine (Google)
3. Include key terms and proper context
4. Be concise but comprehensive

IMPORTANT: Provide only ONE search query, not multiple alternatives.
"""

ASSESSMENT_PROMPT = """
You are an expert researcher tasked with determining if we have enough information to provide a comprehensive answer to a research query.

ORIGINAL QUERY: "{query}"

INFORMATION GATHERED SO FAR:
{information}

First, assess if the information we've gathered is sufficient to provide a comprehensive answer to the original query.
If there are important aspects of the query that are not covered by the information gathered, generate ONE optimized search query that will best fill in the missing information. This query should:
1. Be specific and focused on the most important missing information
2. Be formatted to work well with a search engine (Google)
3. Use proper terminology and relevant keywords
4. NOT repeat any of the search queries shown in the "INFORMATION GATHERED SO FAR" section
5. NOT simply restate the original query unless truly optimal

Return your assessment as a JSON object with the following structure:
{{
    "more_info_required": "True" or "False",
    "search_query": "A single optimized search query" (only if more information needed/more_info_required is True)
}}

IMPORTANT GUIDELINES:
- Generate only ONE search query, make it count
- Ensure the new query explores aspects not covered by previous queries
- If we already have sufficient information, set more_info_required to "False" and don't include a search_query
"""

FINAL_SYNTHESIS_PROMPT = """
Based on all the information gathered, provide a comprehensive, research-quality response to the following query:

QUERY: "{query}"

INFORMATION GATHERED:
{information}

Guidelines for your response:
- Synthesize information from all sources into a clear, flowing narrative
- Support key points with relevant facts, figures, and examples
- Address complexities and nuances where they exist
- Note any significant limitations or gaps in the available information
- Conclude with the most important insights

Your response should be thorough yet engaging, written in a clear, professional style. Structure the response naturally based on the topic and findings, rather than following a rigid format.

Return your response as a string
"""