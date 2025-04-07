import time
from minions.utils.minion_mcp import _make_mcp_minion


def example_local_codebase():
    minion = _make_mcp_minion("filesystem")
    context = ""
    task = "Sumamrize the contents of the minions directory"

    output = minion(
        task=task,
        context=[context],
        max_rounds=10,
        logging_id=int(time.time()),
    )

    print(f"Final answer: {output['final_answer']}")


def example_github():
    minion = _make_mcp_minion("github")
    context = "The MCP tools give access to GitHub."
    task = "Summarize the coding profile of the user, jacksonk, based on their repos."

    output = minion(
        task=task,
        context=[context],
        max_rounds=10,
        logging_id=int(time.time()),
    )

    print(f"Final answer: {output['final_answer']}")


def example_github_read_files():
    minion = _make_mcp_minion("github")
    context = "The MCP tools give access to the repo."
    task = "Identify how many .py files are in my anon/minions repo, including .py files in subdirectories."

    output = minion(
        task=task,
        context=[context],
        max_rounds=10,
        logging_id=int(time.time()),
    )

    print(f"Final answer: {output['final_answer']}")


def example_local():
    minion = _make_mcp_minion("filesystem")
    context = ""
    task = "Summarize the contents of my mcp_test directory"

    output = minion(
        task=task,
        context=[context],
        max_rounds=5,
        logging_id=int(time.time()),
    )

    print(f"Final answer: {output['final_answer']}")


def example_mcp_unnecessary():
    minion = _make_mcp_minion("filesystem")

    context = """
    Patient John Doe is a 60-year-old male with a history of hypertension. In his latest checkup, his blood pressure was recorded at 160/100 mmHg, and he reported occasional chest discomfort during physical activity.
    Recent laboratory results show that his LDL cholesterol level is elevated at 170 mg/dL, while his HDL remains within the normal range at 45 mg/dL. Other metabolic indicators, including fasting glucose and renal function, are unremarkable.
    """

    task = "Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications."

    output = minion(
        task=task,
        context=[context],
        max_rounds=5,
        logging_id=int(time.time()),
    )

    print(f"Final answer: {output['final_answer']}")


if __name__ == "__main__":
    ### Make sure to set you mcp.json file correctly
    example_local_codebase()
