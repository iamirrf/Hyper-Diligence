from app.agent import loop


def test_extractive_agent_records_multi_tool_trace_for_ticker_question(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name: str, args: dict):
        calls.append((name, args))
        if name == "list_filings":
            return [{"ticker": "AAPL", "form": "10-K"}]
        return [
            {
                "chunk_id": 7,
                "ticker": "AAPL",
                "form": "10-K",
                "filed": "2025-10-31",
                "section": "Item 1A.",
                "content": (
                    "Apple described supply chain risks from natural disasters, interruptions, "
                    "manufacturing delays, and supplier concentration that can increase costs."
                ),
            }
        ]

    monkeypatch.setattr(loop, "execute_tool", fake_execute_tool)

    result = loop.run_extractive_agent("What supply chain risks did Apple discuss?")

    assert [call[0] for call in calls] == ["list_filings", "search_filings"]
    assert [item["tool"] for item in result["tool_trace"]] == ["list_filings", "search_filings"]
    assert result["citations"][0]["chunk_id"] == 7
