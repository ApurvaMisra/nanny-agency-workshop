// Stub so the BAML compiler emits a client. Real agent functions added in Plan 3.

function EchoNannyName(name: string) -> string {
  client GPT4oMini
  prompt #"
    Return only the name, exactly: {{ name }}
  "#
}
