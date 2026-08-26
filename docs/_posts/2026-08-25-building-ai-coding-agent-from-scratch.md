---
title: Building AI Coding Agent from Scratch
date: 2026-08-25 00:00:00
featured_image: https://images.unsplash.com/photo-1677095030827-950e69f0545a?q=90&fm=jpg&w=1000&fit=max
excerpt: I wanted to understand how a coding agent actually works so i built a small one in Python without frameworks. Each piece is small enough to hold in your head.
keywords: coding-agent, llm, openrouter, python, tool-calling, agents
---

![](https://images.unsplash.com/photo-1677095030827-950e69f0545a?q=90&fm=jpg&w=1000&fit=max)

I wanted to understand how a coding agent actually works, not by reading another architecture diagram, but by building one. No framework, no magic. Each piece is small enough to hold in your head.

### What a coding agent does?

A chatbot only talks. You type, it types back. An agent can also **do things**: open a file, search your project, run a command, look something up online.

You give it a job, like "what's in `app.py`?" A chatbot would guess from memory. An agent can go look.

Here is what actually happens, in order:

1. Your program sends the question to the model. The model cannot open files by itself. So it replies with a request: "run `read_file` on `app.py`." That is not the answer yet.
2. Your program runs that request, it opens the file on disk and sends the contents back to the model. Now the model has seen the file.
3. Your program asks the model again, with that extra information. This time the model can answer in words: "Here's what `app.py` does…" Or it can request another tool (edit the file, run a test). If it requests another tool, you go back to step 2. If it answers in words, you print that and stop.

The model only talks. Your program is what reads files, runs commands, and searches. The "agent" is this back-and-forth until the model is done asking for help.

### The Components

**The Shell.** A terminal loop. It waits for a line, hands it to the agent, prints the reply. `/q` for quit.

**The Brain.** A language model behind. You send the conversation; it returns a `Thought`: thought can be words to show you, optional thinking, and (later) a list of tools it wants to run.

**Brain Switching.** The brain should not be glued to one model but it should allow models switching, that's why i will use openrouter to be able to switch to any model easily.

**The Tools.** Small Python classes the brain can ask for: read a file, write one, search the repo, search the web, run a command, save a note. The model never runs them. It *requests* them. Your code runs and sends the result back.

**Plan and Act.** Plan mode can only look around and write a plan. Act mode can write files and run commands. `/mode act` unlocks writing.

**Wiring.** Last step is wiring everything together to get the final working agent.

```
you
  → shell (type a line)
       → Agent (loop)
            → Brain (OpenRouter)
            → tools (only if the brain asked)
       → reply printed in the shell
```

### The Shell

Before it can write code or call a model, the agent needs a place to live: a loop that waits for you, hands the line to a handler, and prints the reply. Empty input is ignored. `/q` (or `Ctrl+C`) is a stop signal that ends the process.

```python
class AgentStop(Exception):
    """Raised when the agent should stop processing."""
    pass


class Agent:
    def handle_input(self, user_input):
        if user_input.strip() == "/q":
            raise AgentStop()
        if not user_input.strip():
            return ""
        return f"You said {user_input}\n (Agent not yet connected)"


def main():
    agent = Agent()
    print("⚡ Leif v0.1 initialized.")
    print("Type '/q' to quit.")
    while True:
        try:
            user_input = input("\n❯ ")
            output = agent.handle_input(user_input)
            if output:
                print(output)
        except (AgentStop, KeyboardInterrupt):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
```

Nothing talks to a model yet. That is the point. You can sit in the loop and quit it before any API exists.

### The Brain

The shell can type. It cannot think. The brain is the language model that does the thinking part.

We reach it through OpenRouter. One URL, one API key, many models - Claude, GPT, DeepSeek. You do not need a separate client for each vendor.

```python
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/Clivern/Leif",
    "X-OpenRouter-Title": "Leif",
}
payload = {
    "model": "anthropic/claude-sonnet-4.6",
    "max_tokens": 4096,
    "reasoning": {"effort": "low"},
    "messages": [{"role": "user", "content": "Hello, are you ready to code?"}],
}

response = requests.post(url, headers=headers, json=payload, timeout=120)
print(json.dumps(response.json(), indent=2) if response.ok else response.text)
```

Then wrap the same `POST` so the rest of the program never sees raw JSON. A `Thought` is text, optional thinking, and a list of `ToolCall` - parsed now so the shape does not change when tools arrive.

```python
class ToolCall:
    def __init__(self, id, name, args):
        self.id = id
        self.name = name
        self.args = args


class Thought:
    def __init__(self, text=None, tool_calls=None, thinking=None):
        self.text = text
        self.tool_calls = tool_calls or []
        self.thinking = thinking


class Brain:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = "anthropic/claude-sonnet-4.6"
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def think(self, conversation):
        payload = {
            "model": self.model,
            "max_tokens": 16000,
            "reasoning": {"effort": "low"},
            "messages": conversation,
        }
        response = requests.post(self.url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return self._parse_response(response.json())

    def _parse_response(self, data):
        message = data["choices"][0]["message"]
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], args=args))
        return Thought(
            text=message.get("content") or None,
            tool_calls=tool_calls,
            thinking=message.get("reasoning") or None,
        )
```

### Brain Switching

A second provider should not mean a second `HTTP` client. OpenRouter is one door. Claude, DeepSeek, and GPT are slugs in a dict. `/switch` flips the slug and the brain.

We also need to stop dying on the first `429`: `request_with_retry` backs off on rate limits, 5xx, and network errors.

```python
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

MODELS = {
    "claude": "anthropic/claude-sonnet-4.6",
    "deepseek": "deepseek/deepseek-chat",
    "gpt": "openai/gpt-4o",
}


class Brain:
    def __init__(self, name="claude"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if name not in MODELS:
            raise ValueError(f"Unknown brain: {name}")
        self.name = name
        self.model = MODELS[name]
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def think(self, conversation):
        payload = {
            "model": self.model,
            "max_tokens": 16000,
            "reasoning": {"effort": "low"},
            "messages": conversation,
        }
        response = request_with_retry(self.url, headers, payload)
        return self._parse_response(response.json())


def _switch_brain(self):
    names = list(MODELS.keys())
    new_name = names[(names.index(self.brain.name) + 1) % len(names)]
    self.brain = Brain(new_name)
    return f"Switched to: {new_name}"
```

### The Tools

The model does not execute anything. It *asks*. Each tool is a class with four things: `name`, `description`, `input_schema`, `execute`. Each tool also has a `plan_safe` flag. The next step uses it.

```python
def tool_definitions(tools):
    return [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        },
    } for t in tools]
```

Every tool looks like `read_file`. The rest is the same pattern with different arguments.

#### Files

`read_file` returns the file with line numbers. `write_file` overwrites. `edit_file` finds a string and replaces it once, that's the right tool for a one-line fix.

```python
class ReadFile:
    name = "read_file"
    plan_safe = True
    description = "Reads a file from the filesystem. Use this to examine code."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "The path to the file"}},
        "required": ["path"],
    }

    def execute(self, context, path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(f"{i+1} | {line}" for i, line in enumerate(lines))


class EditFile:
    name = "edit_file"
    plan_safe = False

    def execute(self, context, path, old_text, new_text):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if old_text not in content:
            return f"Error: Could not find the specified text in {path}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.replace(old_text, new_text, 1))
        return f"Successfully edited {path}"
```

#### Look around

Plan mode can read a file if you name it. It cannot find that file. `list_files` walks the tree (skipping `.git`, caches, `.leif`). `search_codebase` greps for a string. `search_web` hits `DuckDuckGo` for what is not in the repo. something like a library that shipped last week or current docs.

```python
class ListFiles:
    name = "list_files"
    plan_safe = True

    def execute(self, context, path="."):
        file_list = []
        for root, dirs, files in os.walk(path):
            # Ideally it should escape the global ignore and local ignore of our coding agent
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "venv", ".venv", ".leif"}]
        return "\n".join(file_list)


class SearchWeb:
    name = "search_web"
    plan_safe = True

    def execute(self, context, query):
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        return "\n".join(
            f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}\n"
            for r in results
        )
```

#### Remember, plan, run

The conversation in memory lives only while the process is running. Close the terminal and it is gone. Preferences should survive that: how you like code formatted, what the project is. We keep those in a markdown file, `.leif/memory.md`. On every request the agent sends that file as the system prompt, so the model still knows who you are next time. `save_memory` overwrites the whole file. It is a scratchpad, not a database.

A plan is different. Before the agent is allowed to change your code, it should be able to write down what it intends to do. `write_plan` saves that outline to `PLAN.md`. That tool is allowed in plan mode, so the agent can think on disk without touching source files.

Then there is the shell. `run_command` really runs a command on your machine like tests, installers, scripts. Output comes back as text, stdout and stderr together. It stops after `LEIF_TIMEOUT` seconds (30 by default) so a hung process cannot sit forever. This tool is act-only. It is also not a sandbox: plan mode means "don't write until I say so," not "this cannot hurt the machine."

```python

class Memory:
    def __init__(self, path=".leif/memory.md"):
        self.path = path
        self._ensure_exists()
        self.content = self._load()

    def save(self, content):
        self.content = content
        with open(self.path, "w") as f:
            f.write(content)


class RunCommand:
    name = "run_command"
    plan_safe = False

    def execute(self, context, command):
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=int(os.environ.get("LEIF_TIMEOUT", "30")),
        )
        return ((result.stdout or "") + (result.stderr or "")).strip() or "(No output)"
```

| Tool | Plan? | Does |
|------|-------|------|
| `read_file` | yes | File with line numbers |
| `list_files` | yes | Tree of the project |
| `search_codebase` | yes | Substring grep |
| `search_web` | yes | DuckDuckGo, 3 hits |
| `save_memory` | yes | Overwrite `.leif/memory.md` |
| `write_plan` | yes | Write `PLAN.md` |
| `write_file` | no | Overwrite a file |
| `edit_file` | no | One find-and-replace |
| `run_command` | no | Shell, with a timeout |

### Plan & Act

Tools that write or run are dangerous on the first turn. Plan is the default. The agent can read, search, save memory, and write `PLAN.md`. It cannot overwrite your code. `/mode act` unlocks writing. `/mode plan` locks it again.

Each tool has `plan_safe`. We only *send* safe tools to the model in plan mode. If it asks for `write_file` anyway, we refuse. Two layers, because models improvise.

```python
def _tools_for_mode(self):
    if self.mode == "act":
        return tool_definitions(self.tools)
    return tool_definitions([t for t in self.tools if t.plan_safe])


def _execute_tool(self, name, args):
    tool = get_tool(self.tools, name)
    if self.mode == "plan" and not tool.plan_safe:
        return f"Error: '{name}' is not available in PLAN mode. Use /mode act to enable writing."
    return tool.execute(ToolContext(memory=self.memory), **args)
```

### Wiring things together

The parts above do nothing until something calls them in order. That something is  the `Agent`. It owns the conversation and the loop

```
you type
  → Agent.handle_input
       → Brain.think (calls model through OpenRouter)
       → Thought (text and/or tool_calls)
       → if tools: execute, append role:tool, think again
       → print text
```

#### The Agentic Loop

This loop is the agent. After the model replies, we save the **whole** reply in the conversation, not only the words it would print.

That matters when it asked for a tool. The next call has to show: the model asked for this tool, here is what we ran. If we drop the ask and only send the result, OpenRouter rejects it.

```python
def _agentic_loop(self):
    output_parts = []
    for _ in range(50):
        thought = self.brain.think(self.conversation)
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
```

#### Memory on every request

The scratchpad is the system prompt. Tools get a `ToolContext` so they can reach `Memory` without a global.

```python
def _build_system_prompt(self):
    parts = [self.memory.content] if self.memory else []
    if self.mode == "plan":
        parts.append(
            "You are in PLAN mode. You cannot write code files. "
            "Use write_plan to save your plans to PLAN.md."
        )
    return "\n".join(parts)
```

#### Compaction

Long sessions fill the context window. After each turn, if input tokens pass 75% of the model's limit, the agent asks the brain to summarize history and starts over from that summary. Tools are stripped for that call so the summary is text, not another tool round. Compaction is lossy. That is the trade.

```python
def _compact_conversation(self):
    history = "\n".join(
        f"{m.get('role', '?')}: {str(m.get('content', m))[:500]}"
        for m in self.conversation
    )
    saved_tools = self.brain.tools
    self.brain.tools = []
    try:
        thought = self.brain.think([{
            "role": "user",
            "content": f"Summarize this conversation...\n\n{history}",
        }])
    finally:
        self.brain.tools = saved_tools
    self.conversation = [
        {"role": "user", "content": f"Previous conversation summary: {thought.text}"},
    ]
```

#### Putting it in `main.py`

The shell is still that `while True` loop. `main` builds the parts and hands them to `Agent`. Run it with `uv run python main.py` (plan) or `uv run python main.py --act`.

```python
def main():
    mode = "act" if len(sys.argv) > 1 and sys.argv[1] == "--act" else "plan"
    name = os.getenv("LEIF_BRAIN", "claude")
    memory = Memory()
    brain = Brain(name, memory=memory, tools=tool_definitions(tools))
    agent = Agent(brain=brain, tools=tools, memory=memory, mode=mode)

    print("⚡ Leif v1.0 (Explore)")
    print("Commands: /q quit, /switch toggle brain, /mode [plan|act]")

    while True:
        try:
            user_input = input(f"[{agent.brain.name}:{agent.mode}] ❯ ")
            output = agent.handle_input(user_input)
            if output:
                print(f"\n{output}\n")
        except (AgentStop, KeyboardInterrupt):
            print("\nExiting...")
            break
```

That is the whole agent: a shell, a brain you can switch, tools, a plan/act gate, and a loop that wires them. This is still not a production coding assistant. There is no approval prompt per tool and no git isolation.

[Full source on GitHub](https://github.com/Clivern/Matrix/tree/main/docs/_code/building-ai-coding-agent-from-scratch).
