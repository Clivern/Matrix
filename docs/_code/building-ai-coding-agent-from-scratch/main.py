import os
import sys
import json
import time
import subprocess
import requests
from dotenv import load_dotenv

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

load_dotenv()


# --- HTTP Helpers ---

def request_with_retry(url, headers, payload, max_retries=10):
    """POST with retry on 429, 5xx, and network failures."""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
        except requests.exceptions.RequestException as e:
            wait_time = 2 ** attempt
            print(f"Network error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue

        if response.status_code == 429 or response.status_code >= 500:
            retry_after = response.headers.get("retry-after")
            try:
                wait_time = int(retry_after) if retry_after else 2 ** attempt
            except (ValueError, TypeError):
                wait_time = 2 ** attempt
            print(f"Error {response.status_code}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue

        if response.status_code >= 400:
            try:
                error_msg = response.json()["error"]["message"]
            except (KeyError, ValueError, requests.exceptions.JSONDecodeError):
                error_msg = response.text
            raise Exception(f"API error ({response.status_code}): {error_msg}")

        return response

    raise Exception(f"Request failed after {max_retries} retries")


# --- Exceptions ---

class AgentStop(Exception):
    """Raised when the agent should stop processing."""
    pass


# --- Brain Response Types ---

class ToolCall:
    """A tool invocation request from the brain."""

    def __init__(self, id, name, args):
        self.id = id
        self.name = name
        self.args = args  # dict


class Thought:
    """Standardized response from any Brain."""

    def __init__(self, text=None, tool_calls=None, raw_message=None, thinking=None):
        self.text = text
        self.tool_calls = tool_calls or []
        self.raw_message = raw_message
        self.thinking = thinking


# --- Memory ---

class Memory:
    """Persistent scratchpad for the agent."""

    def __init__(self, path=".leif/memory.md"):
        self.path = path
        self._ensure_exists()
        self.content = self._load()

    def _ensure_exists(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                f.write("I am Leif, a helpful coding assistant.\n")

    def _load(self):
        with open(self.path, "r") as f:
            return f.read()

    def save(self, content):
        self.content = content
        with open(self.path, "w") as f:
            f.write(content)


class ToolContext:
    """What tools need to know about the agent's state."""

    def __init__(self, memory=None):
        self.memory = memory


# --- Models ---

MODELS = {
    "claude": "anthropic/claude-sonnet-4.6",
    "deepseek": "deepseek/deepseek-chat",
    "gpt": "openai/gpt-4o",
}

CONTEXT_LIMITS = {
    "claude": 200_000,
    "deepseek": 128_000,
    "gpt": 128_000,
}


# --- Brain (one class) ---

class Brain:
    """OpenRouter chat completions — one door, many models."""

    def __init__(self, name="claude", memory=None, tools=None):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env")
        if name not in MODELS:
            raise ValueError(f"Unknown brain: {name}")
        self.name = name
        self.model = MODELS[name]
        self.memory = memory
        self.system = None
        self.tools = tools or []
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.context_limit = CONTEXT_LIMITS.get(name, 200_000)
        self.last_input_tokens = 0

    def think(self, conversation):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Clivern/Leif",
            "X-OpenRouter-Title": "Leif",
        }
        messages = list(conversation)
        if self.system:
            messages = [{"role": "system", "content": self.system}] + messages

        payload = {
            "model": self.model,
            "max_tokens": 16000,
            "reasoning": {
                "effort": "low",  # or "medium" / "high"
            },
            "messages": messages,
        }
        if self.tools:
            payload["tools"] = self.tools

        response = request_with_retry(self.url, headers, payload)
        data = response.json()
        usage = data.get("usage") or {}
        self.last_input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        return self._parse_response(data)

    def _parse_response(self, data):
        message = data["choices"][0]["message"]
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args) if args else {}
            tool_calls.append(ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                args=args,
            ))
        return Thought(
            text=message.get("content") or None,
            tool_calls=tool_calls,
            raw_message=message,
            thinking=message.get("reasoning") or None,
        )


# --- Tools ---

class ReadFile:
    name = "read_file"
    plan_safe = True
    description = "Reads a file from the filesystem. Use this to examine code."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The path to the file"}
        },
        "required": ["path"]
    }

    def execute(self, context, path):
        print(f"  → Reading {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            numbered_lines = [f"{i+1} | {line}" for i, line in enumerate(lines)]
            return "".join(numbered_lines)
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {e}"


class WriteFile:
    name = "write_file"
    plan_safe = False
    description = "Writes content to a file. OVERWRITES existing content."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The path to the file"},
            "content": {"type": "string", "description": "The full content to write"}
        },
        "required": ["path", "content"]
    }

    def execute(self, context, path, content):
        print(f"  → Writing {path}")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing file: {e}"


class WritePlan:
    name = "write_plan"
    plan_safe = True
    description = "Saves a plan to PLAN.md. Use this to outline your approach before making changes."
    input_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The plan content in markdown"}
        },
        "required": ["content"]
    }

    def execute(self, context, content):
        print("  → Writing PLAN.md")
        try:
            with open("PLAN.md", "w", encoding="utf-8") as f:
                f.write(content)
            return "Plan saved to PLAN.md"
        except Exception as e:
            return f"Error saving plan: {e}"


class EditFile:
    name = "edit_file"
    plan_safe = False
    description = "Replaces specific text in a file. Use for surgical edits instead of rewriting entire files."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "old_text": {"type": "string", "description": "Exact text to find and replace"},
            "new_text": {"type": "string", "description": "Text to replace it with"}
        },
        "required": ["path", "old_text", "new_text"]
    }

    def execute(self, context, path, old_text, new_text):
        print(f"  → Editing {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if old_text not in content:
                return f"Error: Could not find the specified text in {path}"
            new_content = content.replace(old_text, new_text, 1)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error editing file: {e}"


class ListFiles:
    name = "list_files"
    plan_safe = True
    description = "Lists all files in the project structure. Useful to understand the project layout."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The root path (default '.')"}
        }
    }

    def execute(self, context, path="."):
        print(f"  → Listing {path}")
        try:
            file_list = []
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "venv", ".venv", ".leif"}]

                level = root.replace(path, "").count(os.sep)
                indent = " " * 4 * level
                file_list.append(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 4 * (level + 1)
                for f in files:
                    file_list.append(f"{subindent}{f}")

            return "\n".join(file_list)
        except Exception as e:
            return f"Error listing files: {e}"


class SearchCodebase:
    name = "search_codebase"
    plan_safe = True
    description = "Searches the entire codebase for a text string. Useful to find where functions or variables are defined."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The string to search for"},
            "path": {"type": "string", "description": "The root path (default '.')"}
        },
        "required": ["query"]
    }

    def execute(self, context, query, path="."):
        print(f"  → Searching for '{query}'")
        results = []
        try:
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "venv", ".venv", ".leif"}]

                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            for i, line in enumerate(f):
                                if query.lower() in line.lower():
                                    results.append(f"{file_path}:{i+1}: {line.strip()}")
                    except Exception:
                        continue

            return "\n".join(results) if results else "No matches found."
        except Exception as e:
            return f"Error searching: {e}"


class SaveMemory:
    name = "save_memory"
    plan_safe = True
    description = "Updates your internal memory/scratchpad. Use this to remember user preferences."
    input_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The full text to save."}
        },
        "required": ["content"]
    }

    def execute(self, context, content):
        print("  → Saving memory")
        if context.memory is None:
            return "Error: Memory not available"
        context.memory.save(content)
        return "Memory updated successfully."


class RunCommand:
    name = "run_command"
    plan_safe = False
    description = "Executes a terminal command. Use this to run scripts, tests, or install packages."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run (e.g., 'python test.py')"}
        },
        "required": ["command"]
    }

    def execute(self, context, command):
        print(f"  → Running: {command[:50]}...")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=int(os.environ.get("LEIF_TIMEOUT", "30")),
                cwd=os.getcwd(),
            )

            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            if not output:
                output = "(No output)"

            return output.strip()

        except subprocess.TimeoutExpired:
            return "Error: Command timed out."
        except Exception as e:
            return f"Error executing command: {e}"


class SearchWeb:
    name = "search_web"
    plan_safe = True
    description = "Searches the internet for current information. Use when you need knowledge beyond your training data."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    }

    def execute(self, context, query):
        print(f"  → Searching web for '{query}'")
        if DDGS is None:
            return "Error: ddgs package not installed. Run: uv add ddgs"
        try:
            results = DDGS().text(query, max_results=3)
            if not results:
                return "No results found."

            formatted = []
            for r in results:
                formatted.append(f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}\n")

            return "\n".join(formatted)
        except Exception as e:
            return f"Error searching web: {e}"


def get_tool(tools, name):
    return next((t for t in tools if t.name == name), None)


def tool_definitions(tools):
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


tools = [
    ReadFile(),
    WritePlan(),
    SaveMemory(),
    ListFiles(),
    SearchCodebase(),
    SearchWeb(),
    WriteFile(),
    EditFile(),
    RunCommand(),
]


# --- Agent ---

class Agent:
    """A coding agent with tools, memory, and plan/act mode."""

    def __init__(self, brain, tools, memory=None, mode="plan"):
        self.brain = brain
        self.tools = list(tools)
        self.memory = memory
        self.mode = mode  # "plan" or "act"
        self.conversation = []
        self.brain.tools = self._tools_for_mode()
        self.brain.system = self._build_system_prompt()

    def _build_system_prompt(self):
        """Build system prompt from memory and current mode."""
        parts = [self.memory.content] if self.memory else []
        if self.mode == "plan":
            parts.append(
                "You are in PLAN mode. You cannot write code files. "
                "Use write_plan to save your plans to PLAN.md."
            )
        return "\n".join(parts)

    def _tools_for_mode(self):
        """Return tool definitions based on current mode."""
        if self.mode == "act":
            return tool_definitions(self.tools)
        return tool_definitions([t for t in self.tools if t.plan_safe])

    def handle_input(self, user_input):
        if user_input.strip() == "/q":
            raise AgentStop()

        if user_input.strip() == "/switch":
            return self._switch_brain()

        if not user_input.strip():
            return ""

        if user_input.strip().startswith("/mode"):
            return self._handle_mode_command(user_input)

        self.conversation.append({"role": "user", "content": user_input})

        try:
            return self._agentic_loop()
        except Exception as e:
            self.conversation.pop()
            return f"Error: {e}"

    def _handle_mode_command(self, user_input):
        """Handle /mode command to switch between plan and act."""
        parts = user_input.strip().split()
        if len(parts) > 1 and parts[1] == "act":
            self.mode = "act"
            self.brain.tools = self._tools_for_mode()
            self.brain.system = self._build_system_prompt()
            return "⚠️  Switched to ACT MODE (Writing Enabled)"
        self.mode = "plan"
        self.brain.tools = self._tools_for_mode()
        self.brain.system = self._build_system_prompt()
        return "🛡️  Switched to PLAN MODE (Code Read-Only)"

    def _agentic_loop(self):
        output_parts = []
        max_iterations = 50

        for _iteration in range(max_iterations):
            thought = self.brain.think(self.conversation)

            if thought.thinking:
                lines = thought.thinking.strip().split("\n")[:5]
                for i, line in enumerate(lines):
                    prefix = "  💭 " if i == 0 else "     "
                    print(f"\033[2m{prefix}{line}\033[0m")

            if self.brain.last_input_tokens > self.brain.context_limit * 0.75:
                self._compact_conversation()

            self.conversation.append(thought.raw_message)

            if thought.text:
                output_parts.append(thought.text)

            if not thought.tool_calls:
                break

            for tool_call in thought.tool_calls:
                result = self._execute_tool(tool_call.name, tool_call.args)
                self.conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            output_parts.append("(Stopped: too many iterations)")

        return "\n".join(output_parts)

    def _compact_conversation(self):
        """Summarize old messages to stay within context limits."""
        print("(Compacting conversation...)")
        history = "\n".join(
            f"{m.get('role', '?')}: {str(m.get('content', m))[:500]}"
            for m in self.conversation
        )
        prompt = [{
            "role": "user",
            "content": (
                "Summarize this conversation for continuity. "
                "Focus on what was accomplished, what's in progress, "
                f"and key decisions:\n\n{history}"
            ),
        }]
        saved_tools = self.brain.tools
        self.brain.tools = []
        try:
            thought = self.brain.think(prompt)
        finally:
            self.brain.tools = saved_tools
        self.conversation = [
            {"role": "user", "content": f"Previous conversation summary: {thought.text}"},
        ]

    def _execute_tool(self, name, args):
        tool = get_tool(self.tools, name)
        if tool is None:
            return f"Error: Tool '{name}' not found"
        if self.mode == "plan" and not tool.plan_safe:
            return f"Error: '{name}' is not available in PLAN mode. Use /mode act to enable writing."
        try:
            context = ToolContext(memory=self.memory)
            return tool.execute(context, **args)
        except TypeError as e:
            return f"Error: Invalid arguments - {e}"

    def _switch_brain(self):
        names = list(MODELS.keys())
        idx = names.index(self.brain.name)
        new_name = names[(idx + 1) % len(names)]
        self.brain = Brain(new_name, memory=self.memory, tools=self._tools_for_mode())
        self.brain.system = self._build_system_prompt()
        return f"Switched to: {new_name}"


# --- Main Loop ---

def main():
    mode = "act" if len(sys.argv) > 1 and sys.argv[1] == "--act" else "plan"
    name = os.getenv("LEIF_BRAIN", "claude")
    memory = Memory()
    brain = Brain(name, memory=memory, tools=tool_definitions(tools))
    agent = Agent(brain=brain, tools=tools, memory=memory, mode=mode)

    print("⚡ Leif v1.0 (Explore)")
    print("Commands: /q quit, /switch toggle brain, /mode [plan|act]")
    print(f"Brain: {agent.brain.name}")
    if mode == "act":
        print("Mode: ACT (Writing Enabled)")
    else:
        print("Mode: PLAN (Code Read-Only)")

    while True:
        try:
            user_input = input(f"[{agent.brain.name}:{agent.mode}] ❯ ")
            output = agent.handle_input(user_input)
            if output:
                print(f"\n{output}\n")
        except (AgentStop, KeyboardInterrupt):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
