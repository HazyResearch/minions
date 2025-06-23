# Development-focused prompts for DevMinion

RUNBOOK_GENERATION_PROMPT = """You are an expert software architect tasked with breaking down a development task into manageable steps.

**Development Task:** {task}

**Requirements/Context:** {requirements}

Create a detailed runbook with numbered steps to complete this task. Each step should be specific and actionable.

Output your response as JSON with the following structure:
{{
    "project_overview": "Brief description of what will be built",
    "technology_stack": ["list", "of", "technologies"],
    "steps": [
        {{
            "step_number": 1,
            "title": "Brief step title",
            "description": ["Subtask 1", "Subtask 2", "Subtask 3", ...],
            "files_to_create": ["list", "of", "files", "to", "create"],
            "files_to_modify": ["list", "of", "existing", "files", "to", "modify"],
            "tests_needed": ["list", "of", "tests", "to", "write"],
            "acceptance_criteria": "What constitutes completion of this step"
        }}
    ],
    "final_testing": "Description of final integration testing needed"
}}

Make sure each step is:
1. Atomic - can be completed independently
2. Testable - has clear success criteria
3. Incremental - builds on previous steps
4. Specific - includes exact file names and functionality
"""

SUBTASK_EXECUTION_PROMPT = """You are a skilled developer working on a specific subtask. 

**Current Step:** {step_number} - {step_title}
**Step Description:** {step_description}
**Files to Create:** {files_to_create}
**Files to Modify:** {files_to_modify}
**Tests Needed:** {tests_needed}
**Acceptance Criteria:** {acceptance_criteria}

**Current Workspace State:**
{current_workspace}

**Previous Steps Completed:** {completed_steps}

{previous_feedback_section}

Your task is to:
1. Implement the required functionality for this step
2. Create/modify the specified files
3. Document your changes in a markdown file

For each file you create or modify, provide:
- The complete file content
- Brief explanation of changes made
- Any dependencies or setup instructions

Create a detailed documentation file describing:
- What you implemented
- How it works
- Any issues encountered
- Next steps or recommendations

NOTE: You do NOT need to write tests - the testing will be handled separately by another process.

Output your response as JSON:
{{
    "files": {{
        "filename.py": "complete file content here"
    }},
    "documentation": "# Step {step_number} Implementation\\n\\nDetailed markdown documentation here",
    "setup_instructions": ["any", "setup", "commands", "needed"],
    "completion_notes": "Brief summary of what was accomplished"
}}
"""

TEST_GENERATION_PROMPT = """You are a senior test engineer tasked with writing comprehensive tests for a development step.

**Step Being Tested:** {step_number} - {step_title}
**Step Description:** {step_description}
**Tests Needed:** {tests_needed}
**Acceptance Criteria:** {acceptance_criteria}

**Code Implementation:**
{code_changes}

**Documentation:**
{documentation}

**Current Workspace State:**
{workspace_state}

Your task is to create comprehensive test scripts that verify the implementation meets the acceptance criteria and tests all the specified functionality.

Write tests that:
1. Test all the functionality described in the acceptance criteria
2. Cover the specific tests mentioned in "tests_needed"
3. Include both positive and negative test cases
4. Test edge cases and error conditions
5. Verify integration with existing code
6. Are executable as simple Python scripts

Output your response as JSON:
{{
    "test_files": {{
        "test_filename.py": "complete test file content here"
    }},
    "test_commands": ["list", "of", "commands", "to", "run", "tests", "(e.g., python test_filename.py)"],
    "test_documentation": "# Test Documentation\\n\\nDescription of what tests cover and how to run them",
    "expected_outcomes": "What should happen when tests pass"
}}
"""

CODE_REVIEW_PROMPT = """You are a senior developer conducting a code review for a development subtask.

**Step Being Reviewed:** {step_number} - {step_title}
**Expected Deliverables:** {acceptance_criteria}

**Code Changes:**
{code_changes}

**Documentation:**
{documentation}

**Test Results:**
{test_results}

**Workspace State:**
{workspace_state}

Review the implementation against the following criteria:
1. **Functionality**: Does it meet the acceptance criteria?
2. **Code Quality**: Is the code well-structured and readable?
3. **Documentation**: Is the implementation properly documented?
4. **Integration**: Does it work well with existing code?
5. **Best Practices**: Does it follow coding standards?

Provide feedback and make a decision. Output as JSON:
{{
    "overall_score": "score out of 10",
    "feedback": {{
        "strengths": ["list", "of", "positive", "aspects"],
        "issues": ["list", "of", "problems", "found"],
        "suggestions": ["list", "of", "improvement", "suggestions"]
    }},
    "specific_changes_needed": ["if request_edits, list specific changes needed"],
    "decision": "merge_changes" OR "request_edits" (if score > 7, then merge_changes, otherwise request_edits),
    "approval_notes": "Comments for approval or reasons for rejection",
}}

If requesting edits, be specific about what needs to be changed and why. Don't request changes unless the code is not working as expected or is not meeting the acceptance criteria.
"""

EDIT_REQUEST_PROMPT = """You are a developer addressing code review feedback on your previous implementation.

**Original Step:** {step_number} - {step_title}
**Your Previous Implementation:**
{previous_implementation}

**Code Review Feedback:**
{review_feedback}

**Specific Changes Requested:**
{changes_requested}

**Current Workspace State:**
{current_workspace}

Address the feedback by:
1. Making the requested changes
2. Fixing any identified issues
3. Improving code quality based on suggestions
4. Updating tests if needed
5. Updating documentation to reflect changes

Output your revised implementation as JSON:
{{
    "files": {{
        "filename.py": "updated complete file content",
        "test_filename.py": "updated test file content"
    }},
    "documentation": "# Step {step_number} Implementation (Revised)\\n\\nUpdated documentation",
    "test_commands": ["updated", "test", "commands"],
    "changes_made": "Summary of changes made to address feedback",
    "completion_notes": "Brief summary of revisions"
}}
"""

FINAL_INTEGRATION_PROMPT = """You are conducting a final integration review of the completed development project.

**Original Task:** {original_task}
**Project Overview:** {project_overview}
**Completed Steps:** {completed_steps}

**Final Workspace State:**
{final_workspace}

**All Documentation:**
{all_documentation}

**Final Test Results:**
{final_test_results}

Conduct a comprehensive review to determine if the project successfully meets the original requirements:

1. **Completeness**: Are all required features implemented?
2. **Functionality**: Does everything work as expected?
3. **Quality**: Is the code maintainable and well-structured?
4. **Testing**: Is there adequate test coverage?
5. **Documentation**: Is the project well-documented?
6. **Deployment Ready**: Can this be deployed/used?

Output your final assessment as JSON:
{{
    "project_status": "complete" OR "needs_work",
    "completion_percentage": "percentage of requirements met",
    "final_assessment": {{
        "strengths": ["major", "accomplishments"],
        "weaknesses": ["areas", "needing", "improvement"],
        "missing_features": ["if", "any"],
        "quality_score": "score out of 10"
    }},
    "deployment_readiness": {{
        "ready_to_deploy": true/false,
        "setup_instructions": "How to run/deploy the project",
        "known_issues": ["any", "remaining", "issues"]
    }},
    "recommendations": ["suggestions", "for", "improvement", "or", "next", "steps"]
}}
""" 