from llama_cpp_agent.messages_formatter import MessagesFormatter, PromptMarkers, Roles

mistral_v1_markers = {
    Roles.system: PromptMarkers(""" [INST]""", """ [/INST] Understood.</s>"""),
    Roles.user: PromptMarkers(""" [INST]""", """ [/INST]"""),
    Roles.assistant: PromptMarkers(" ", "</s>"),
    Roles.tool: PromptMarkers("", ""),
}

mistral_v1_formatter = MessagesFormatter(
    pre_prompt="",
    prompt_markers=mistral_v1_markers,
    include_sys_prompt_in_first_user_message=False,
    default_stop_sequences=["</s>"]
)

mistral_v2_markers = {
    Roles.system: PromptMarkers("""[INST] """, """[/INST] Understood.</s>"""),
    Roles.user: PromptMarkers("""[INST] """, """[/INST]"""),
    Roles.assistant: PromptMarkers(" ", "</s>"),
    Roles.tool: PromptMarkers("", ""),
}

mistral_v2_formatter = MessagesFormatter(
    pre_prompt="",
    prompt_markers=mistral_v2_markers,
    include_sys_prompt_in_first_user_message=False,
    default_stop_sequences=["</s>"]
)

mistral_v3_tekken_markers = {
    Roles.system: PromptMarkers("""[INST]""", """[/INST]Understood.</s>"""),
    Roles.user: PromptMarkers("""[INST]""", """[/INST]"""),
    Roles.assistant: PromptMarkers("", "</s>"),
    Roles.tool: PromptMarkers("", ""),
}

mistral_v3_tekken_formatter = MessagesFormatter(
    pre_prompt="",
    prompt_markers=mistral_v3_tekken_markers,
    include_sys_prompt_in_first_user_message=False,
    default_stop_sequences=["</s>"]
)
