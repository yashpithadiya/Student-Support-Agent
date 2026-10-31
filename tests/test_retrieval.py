import retrieval


def ids(q):
    return [c["chunk_id"] for c in retrieval.search_policy(q)]


def test_handbook_is_split_into_sections_and_disclaimer_present():
    chunks = retrieval.load_chunks()
    assert len(chunks) >= 10
    assert "payment-portal-failure" in [c["chunk_id"] for c in chunks]
    assert "FICTIONAL".lower() in retrieval.config.POLICY_MD.read_text().lower()


def test_portal_failure_question_retrieves_the_right_chunk_first():
    assert ids("The payment portal failed. Can I pay on the next working day?")[0] == "payment-portal-failure"


def test_other_topics():
    assert ids("How do I change my elective course?") == ["course-change-procedure"]
    assert ids("How many days to appeal a rejected scholarship?")[0] == "scholarship-application-status-procedure"
    assert ids("minimum attendance required")[0] == "attendance-policy"
    assert ids("Can I get re-evaluation of my paper?")[0] == "examination-re-evaluation"


def test_only_relevant_chunks_returned_not_whole_document():
    total = len(retrieval.load_chunks())
    assert 1 <= len(retrieval.search_policy("What is the late fee?")) <= 2 < total


def test_unsupported_questions_return_nothing():
    assert ids("What is the hostel curfew time?") == []
    assert ids("What is the capital of France?") == []
    assert ids("placement policy") == []


def test_irrelevant_chunks_not_included():
    got = ids("How do I change my elective course?")
    assert "scholarship-eligibility" not in got and "late-fee-rules" not in got


def test_search_policy_tool_reports_no_result_clearly():
    import tools
    r = tools.execute_tool("search_policy", {"query": "hostel curfew time"})
    assert r["ok"] and r["chunks"] == [] and "No relevant section" in r["note"]
    assert tools.execute_tool("search_policy", {"query": "  "})["ok"] is False
