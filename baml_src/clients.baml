// OpenAI clients used by the agent. Plan 3 adds more (with different reasoning efforts).
// Current OpenAI reasoning models reject temperature/top_p — reasoning effort is the knob instead.

client<llm> GPT6Luna {
  provider openai
  options {
    model "gpt-6-luna"
    reasoning_effort "low"
    api_key env.OPENAI_API_KEY
  }
}

client<llm> GPT61Sol {
  provider openai
  options {
    model "gpt-6.1-sol"
    reasoning_effort "low"
    api_key env.OPENAI_API_KEY
  }
}
