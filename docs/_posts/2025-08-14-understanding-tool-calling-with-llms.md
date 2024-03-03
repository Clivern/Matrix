---
title: Understanding Tool Calling with LLMs
date: 2025-08-14 00:00:00
featured_image: https://images.unsplash.com/photo-1748738293234-7788833c7578?q=90&fm=jpg&w=1000&fit=max
excerpt: Tool calling (also known as function calling) is a powerful feature in modern Large Language Models that enables them to interact with external systems and APIs.
keywords: llm, ai, tool calling, function calling, openai, gpt
---

![](https://images.unsplash.com/photo-1748738293234-7788833c7578?q=90&fm=jpg&w=1000&fit=max)

Tool calling is a powerful feature in modern Large Language Models that enables them to interact with external systems and APIs. This article demonstrates the difference between a standard LLM request and one that leverages tool calling.

When you ask an LLM about real-time information like Bitcoin prices, it can't directly access that data. Let's see what happens with a standard request:

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "o4-mini",
    "messages": [{"role": "user", "content": "What is the current price of Bitcoin in USD?"}]
  }'
```

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "I'm not able to pull live market data directly, but you can check Bitcoin's real-time USD price on any major exchange or data provider..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "completion_tokens": 287,
    "reasoning_tokens": 128
  }
}
```

Tool calling allows you to define functions that the LLM can "call" when it needs external data or actions. The LLM doesn't actually execute these functions—it identifies when they're needed and returns structured instructions for your application to execute them.

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "o4-mini",
    "messages": [{"role": "user", "content": "What is the current price of Bitcoin in USD?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_bitcoin_price",
          "description": "Get the current price of Bitcoin",
          "parameters": {
            "type": "object",
            "properties": {
              "currency": {
                "type": "string",
                "description": "The currency to get the price in"
              }
            },
            "required": ["currency"]
          }
        }
      }
    ]
  }'
```

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_GxQeC2epvqEUQyJbTn7LzLSE",
        "type": "function",
        "function": {
          "name": "get_bitcoin_price",
          "arguments": "{\"currency\":\"USD\"}"
        }
      }],
      "finish_reason": "tool_calls"
    }
  }],
  "usage": {
    "completion_tokens": 89,
    "reasoning_tokens": 64
  }
}
```

### How Tool Calling Works

#### Phase 1: Request & Analysis

**User → LLM**
- User asks: *"What is the current price of Bitcoin in USD?"*
- LLM analyzes the question and scans available tools
- LLM identifies `get_bitcoin_price` as the appropriate tool
- LLM extracts required parameters: `{"currency": "USD"}`
- LLM returns a structured `tool_call` instruction (not the actual result)

#### Phase 2: Execution

**Application → External API**
- Your application receives the `tool_call` from LLM
- Application executes the actual function (calls Bitcoin API)
- API returns the result: `{"price": 95234.50, "currency": "USD"}`
- Application sends the function result back to LLM

#### Phase 3: Final Response

**LLM → User**
- LLM receives the function execution result
- LLM formats a natural language response: *"The current price of Bitcoin is $95,234.50 USD"*

It is worth noting that the LLM doesn't execute functions. it identifies *which* function to call and *what* parameters to use. Your application handles the actual execution.
