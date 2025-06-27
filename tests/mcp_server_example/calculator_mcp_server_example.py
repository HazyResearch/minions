import sys
import traceback

print(">>> [DEBUG] mcp_calculator.py starting", file=sys.stderr, flush=True)

try:
    # Use FastMCP as the server interface
    from mcp.server.fastmcp import FastMCP

    # Create the MCP server instance
    mcp = FastMCP("Calculator Server")

    # --- Define your calculator tool functions here ---

    @mcp.tool(description="Add two numbers")
    def add(a: float, b: float) -> float:
        """Add two numbers and return the result."""
        return a + b

    @mcp.tool(description="Subtract b from a")
    def subtract(a: float, b: float) -> float:
        """Subtract b from a and return the result."""
        return a - b

    @mcp.tool(description="Multiply two numbers")
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers and return the result."""
        return a * b

    @mcp.tool(description="Divide a by b")
    def divide(a: float, b: float) -> float:
        """Divide a by b and return the result. Returns error if b is zero."""
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return a / b

    print(">>> [DEBUG] Calculator tools registered", file=sys.stderr, flush=True)

    # --- Start the server ---
    if __name__ == "__main__":
        print(">>> [DEBUG] Running FastMCP server", file=sys.stderr, flush=True)
        mcp.run()  # FastMCP auto-detects stdio when run as a subprocess

except Exception as e:
    print(f">>> [ERROR] Exception in mcp_calculator.py: {e}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
