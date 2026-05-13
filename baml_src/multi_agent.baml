// Multi-agent topology BAML functions. Used in sections 7 and 8.

class PlanStep {
  agent string @description("Which role handles this step: researcher, executor, or planner")
  task string @description("What that agent should do, in one sentence")
}

class Plan {
  goal string
  steps PlanStep[]
}

function PlanAgent(user_query: string) -> Plan {
  client GPT4oMini
  prompt #"
    You are the PLANNER for a nanny-agency booking assistant. Given a parent's request,
    produce a plan with concrete next steps for the team.

    Roles you can assign:
    - researcher: looks up nanny profiles or agency policy
    - executor: drafts emails, confirms bookings, sends replies

    Keep the plan short (1-4 steps). Each step's task should be a single sentence.

    USER QUERY: {{ user_query }}

    {{ ctx.output_format }}
  "#
}

class ResearchFinding {
  source string @description("Where this came from: 'nanny_db' or 'policies' or 'unknown'")
  excerpt string @description("Verbatim excerpt or structured summary that proves the claim")
  claim string @description("One-sentence summary of what was found")
}

function ResearcherAgent(task: string, evidence_context: string) -> ResearchFinding {
  client GPT4oMini
  prompt #"
    You are the RESEARCHER. Your role is to CITE sources, not invent facts.
    Read the evidence and produce a finding that cites where it came from.

    If the evidence does not directly support the task, your finding's source MUST be 'unknown'
    and excerpt empty. Never fabricate.

    TASK: {{ task }}

    EVIDENCE (verbatim — only cite from here):
    ---
    {{ evidence_context }}
    ---

    {{ ctx.output_format }}
  "#
}

function ExecutorAgent(task: string, findings: string) -> string {
  client GPT4oMini
  prompt #"
    You are the EXECUTOR. You draft user-facing replies based on what the researcher found.
    Stay within scope of the task. Do NOT re-research; trust the findings provided.

    Tone: warm, professional, reassuring. 2-5 sentences.

    TASK: {{ task }}

    RESEARCHER FINDINGS:
    {{ findings }}
  "#
}

function CriticAgent(draft_reply: string, findings: string, original_request: string) -> string {
  client GPT4oMini
  prompt #"
    You are the CRITIC. Review the draft reply. Verify:
    1. It addresses the original parent request.
    2. Every factual claim in the reply appears in the researcher findings.
    3. The tone is warm + professional.

    If the reply is acceptable, output exactly: APPROVED
    Otherwise, output a SHORT (1-2 sentence) critique of what's wrong, prefixed with REVISE:.

    ORIGINAL REQUEST: {{ original_request }}

    DRAFT REPLY:
    {{ draft_reply }}

    RESEARCHER FINDINGS:
    {{ findings }}
  "#
}

// Versioned shared-memory write (Section 8, Stale Shared Memory fix).
class VersionedFact {
  key string
  value string
  version int @description("Monotonic version; readers check this is the latest before trusting")
}
