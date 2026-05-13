// ReAct agent for the booking assistant. Plan 3 single-agent functions live here.

// One tool the agent can call. Tools are dispatched in Python by name.
class ToolCall {
  name string @description("One of: search_nannies, get_policy, check_availability, draft_email, finish")
  args map<string, string> @description("String-keyed args for the tool. Empty map for 'finish'.")
}

// One step of the agent's reasoning loop.
class AgentStep {
  thought string @description("Brief reasoning about what to do next.")
  tool_call ToolCall @description("The tool to invoke this turn. Use name='finish' when answer is ready.")
  final_answer string? @description("If tool_call.name == 'finish', this is the user-facing answer.")
}

// Single-tool ReAct primer (Section 1). Agent uses ONLY search_nannies + finish.
function DecideOneTool(user_message: string, history: string) -> AgentStep {
  client GPT4oMini
  prompt #"
    You are a nanny-agency booking assistant. You may only call: search_nannies, finish.

    Tool: search_nannies
      args: query (string) — natural-language description of needed nanny.
      returns: a list of candidate nannies.

    Decide the next step. If you have enough information, call finish with a user-facing
    final_answer. Otherwise call search_nannies.

    USER MESSAGE:
    {{ user_message }}

    HISTORY (prior thoughts/observations, may be empty):
    {{ history }}

    {{ ctx.output_format }}
  "#
}

// Multi-tool agent with all 4 tools exposed (Section 2 demo).
function DecideAllTools(user_message: string, history: string) -> AgentStep {
  client GPT4oMini
  prompt #"
    You are a nanny-agency booking assistant. You can call any of these tools:

    Tools available:
    - search_nannies(query: string) — find candidate nannies by description
    - get_policy(topic: string) — look up agency policy (e.g., "booking confirmation", "PII", "escalation")
    - check_availability(nanny_id: string, day: string) — returns true/false
    - draft_email(parent_id: string, nanny_id: string, day: string, hours: string) — produces email text
    - finish — when ready to reply to the user

    Choose one tool per step. If you have enough information, call finish with final_answer.

    USER MESSAGE:
    {{ user_message }}

    HISTORY:
    {{ history }}

    {{ ctx.output_format }}
  "#
}

// Tool-scoped agent (Section 2 fix). The CALLER decides which tools to expose.
function DecideWithTools(user_message: string, history: string, tools_doc: string) -> AgentStep {
  client GPT4oMini
  prompt #"
    You are a nanny-agency booking assistant. You can call only these tools:

    {{ tools_doc }}

    Choose one tool per step. If you have enough information, call finish with final_answer.

    USER MESSAGE:
    {{ user_message }}

    HISTORY:
    {{ history }}

    {{ ctx.output_format }}
  "#
}

// Intent classifier (Section 2 fix). Decides which tool family is relevant.
function ClassifyIntent(user_message: string) -> string {
  client GPT4oMini
  prompt #"
    Classify this user message into exactly ONE category:
    - search (they're looking for nannies)
    - policy (they're asking about agency rules / cancellation / safety)
    - booking (they're confirming or scheduling a session)
    - other (anything else)

    Reply with ONLY the category word.

    USER MESSAGE: {{ user_message }}
  "#
}
