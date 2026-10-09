from alpha_lab.agents.llm import _tool_choice


def test_haiku_keeps_the_forced_tool_choice():
    assert _tool_choice("claude-haiku-5-5", "submit_idea") == {
        "type": "tool",
        "name": "submit_idea",
        "disable_parallel_tool_use": True,
    }


def test_sonnet_uses_auto_because_it_rejects_forced_tool_choice():
    assert _tool_choice("claude-sonnet-5-5", "submit_critique") == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }
