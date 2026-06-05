from strands.types.tools import ToolUse

TOOL_SPEC = {
    "name": "think",
    "description": (
        "Structure your reasoning before committing to a multi-step action. "
        "This tool does not read files, run code, or change the repository — "
        "it forces a deliberation step so you act on a considered plan rather than the first idea.\n\n"
        "Call it in these situations:\n"
        "1. After localizing a bug, before writing the patch — state the root cause in one sentence, "
        "then list 2-4 candidate fixes. For each, note whether it addresses the root cause or just the "
        "observed symptom, and pick the minimal change that fixes the cause.\n"
        "2. After a failed test or reproduction — list 2-4 hypotheses for the failure ranked by likelihood, "
        "and identify the cheapest next action that would discriminate between them.\n"
        "3. Before a refactor or multi-file change — outline the approach, the files touched, and one "
        "concrete risk (e.g., a caller you might break).\n\n"
        "Skip for trivial actions: one-line fixes, obvious next commands, simple file lookups, or steps "
        "where the next action is already determined. Do not use it to narrate what you just did.\n\n"
        "Format: 3-5 concise bullets. No prose paragraphs. No restating the problem."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your thoughts."
                },
            "required": ["thought"],
            }   
        },
    }
}

def think(tool: ToolUse, **kwargs):
    tool_use_id = tool.get("toolUseId", "default-id")
    tool_input = tool.get("input", {})
    try:
        if not tool_input.get("thought"):
            raise ValueError("thought is required")
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text":"thought logged; continue"}], 
        }
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": error_msg}],
        }