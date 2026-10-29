---
title: Human in the Loop with LangGraph
date: 2026-01-15 00:00:00
featured_image: https://images.unsplash.com/photo-1601950135061-76231010f960?q=90&fm=jpg&w=1000&fit=max
excerpt: An agent should not send an email or delete a record just because the model said so. LangGraph can pause a run, wait for a person, and continue from the same state.
keywords: langgraph, human-in-the-loop, agents, interrupt, langchain
---

![](https://images.unsplash.com/photo-1601950135061-76231010f960?q=90&fm=jpg&w=1000&fit=max)

An agent that acts on its own is useful until it sends the wrong email to a customer. Some steps need a person to say yes first. Sending a support reply, issuing a refund, or deleting a record are some examples. The agent does the work up to that step, then stops and waits.

`LangGraph` makes this easy because pausing is part of the graph. A run can stop, keep its state, and continue later. Even if the process restarts in between.

### How to achieve this?

- `interrupt(value)` pauses the graph and hands `value` to whoever is running it.
- `Command(resume=...)` continues the paused run. The value becomes the return of `interrupt()`.

A `checkpointer` saves the state at each step, keyed by a `thread_id`. Same thread id, same run.

A small graph shows the whole flow. It drafts a reply, pauses for a person, and sends only if the person approves:

```python
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    ticket: str
    draft: str
    action: str
    category: str


def draft(state: State) -> dict:
    return {"draft": f"Sorry about this. We will fix: {state['ticket']}", "category": "issue"}


def review(state: State) -> dict:
    response = interrupt({"draft": state["draft"]})
    return {"action": response["action"]}


def send(state: State) -> dict:
    print("sent:", state["draft"])
    return {}


def route(state: State) -> str:
    return "send" if state["action"] == "approve" else END


builder = StateGraph(State)
builder.add_node("draft", draft)
builder.add_node("review", review)
builder.add_node("send", send)
builder.add_edge(START, "draft")
builder.add_edge("draft", "review")
builder.add_conditional_edges("review", route, ["send", END])
builder.add_edge("send", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "ticket-42"}}

out = graph.invoke({"ticket": "My order arrived broken."}, config)
print(out["__interrupt__"][0].value)  # {'draft': '...'}

# Graph will get the state from MemorySaver with the thread_id and continue the graph
out = graph.invoke(Command(resume={"action": "approve"}), config)
print(out)
```

Running this prints the draft, pauses, and then sends after the auto approval. The resume call only supplies `{"action": "approve"}`. The rest of the state comes from the saved checkpoint, so `draft` step does not run again.

```
{'draft': 'Sorry about this. We will fix: My order arrived broken.'}
sent: Sorry about this. We will fix: My order arrived broken.
{'ticket': 'My order arrived broken.', 'draft': 'Sorry about this. We will fix: My order arrived broken.', 'action': 'approve', 'category': 'issue'}
```

The `interrupt()` line does pause on the first run, but it only pauses when there's no answer for it yet.

Think of it as asking a question that gets stored:

1. First run: review reaches `interrupt(...)`. No answer is stored for this question, so the graph stops and waits.
2. Resume: you call `Command(resume={"action": "approve"})`, which stores that answer. `LangGraph` runs review again from the top. When it reaches `interrupt(...)`, the answer is already there, so `interrupt()` returns it, and the next line runs.

One thing to watch. On resume, `LangGraph` runs the node that paused from its first line again. It does not jump into the middle of it. So anything before `interrupt()` in that node runs twice. Keep reads and LLM calls in their own nodes, and put only the human check in the node that pauses. Something like the following will be bad

```python
def review(state: State) -> dict:
    category = llm_call() # Should be in separate node otherwise it will run again on resume.
    response = interrupt({"draft": state["draft"]})
    return {"action": response["action"]}
```

In production you should use `postgres` or `sqlite` as `checkpointer`. also approval can be over many channels and once you get the approval or rejection or something else you initialize the graph with your `checkpointer` and the `interrupt` answer.

Use this pattern anywhere an agent does real work that a person needs to check. Support replies are the obvious case: the agent drafts, and a human approves, rephrases, or writes the reply.

The same applies to emails or documents that trigger work, where the agent reads and prepares, then waits before it sends or changes anything. Refunds, record deletions, and data changes fit too. The rule is the same: let the agent do the preparation, and make the step that can't be undone wait for a person.
