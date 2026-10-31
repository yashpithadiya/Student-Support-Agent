import tools


def test_fee_status_known_record():
    r = tools.execute_tool("check_fee_status", {"student_id": "S1002"})
    assert r["ok"] and r["fee_status"] == "Partially Paid"
    assert r["balance_inr"] == 225000 and r["next_due_date"] == "2026-11-05"


def test_fee_status_id_is_case_insensitive():
    assert tools.execute_tool("check_fee_status", {"student_id": " s1001 "})["fee_status"] == "Paid"


def test_scholarship_status_known_record():
    r = tools.execute_tool("check_scholarship_status", {"student_id": "S1004"})
    assert r["ok"] and r["status"] == "Rejected"


def test_student_record():
    r = tools.execute_tool("get_student_record", {"student_id": "S1005"})
    assert r["ok"] and r["programme"] == "PGDM" and r["course"] == "BDA" and r["semester"] == 4


def test_missing_student_record():
    r = tools.execute_tool("check_fee_status", {"student_id": "S9999"})
    assert r == {"ok": False, "error_type": "not_found", "error": "No student record found for ID S9999."}


def test_invalid_student_id():
    r = tools.execute_tool("check_scholarship_status", {"student_id": "hello"})
    assert not r["ok"] and r["error_type"] == "invalid_input"


def test_simulated_service_failure_is_graceful():
    r = tools.execute_tool("check_fee_status", {"student_id": "S1009"})
    assert not r["ok"] and r["error_type"] == "tool_failure" and "unavailable" in r["error"]


def test_missing_argument_does_not_crash():
    r = tools.execute_tool("check_fee_status", {})
    assert not r["ok"] and r["error_type"] == "invalid_input"


def test_unknown_tool():
    assert tools.execute_tool("delete_everything", {})["error_type"] == "unknown_tool"


def test_create_ticket_and_idempotency():
    args = {"student_id": "S1003", "category": "fee", "description": "Payment portal failed twice"}
    first = tools.execute_tool("create_support_request", args)
    assert first["ok"] and first["ticket_id"] == "TKT-1001" and first["priority"] == "Urgent"
    again = tools.execute_tool("create_support_request", args)
    assert again["ticket_id"] == "TKT-1001"                       # no duplicate ticket
    other = tools.execute_tool("create_support_request", {**args, "description": "A different problem here"})
    assert other["ticket_id"] == "TKT-1002"


def test_ticket_validation():
    bad_cat = tools.execute_tool("create_support_request",
                                 {"student_id": "S1003", "category": "weather", "description": "long enough text"})
    short = tools.execute_tool("create_support_request",
                               {"student_id": "S1003", "category": "fee", "description": "help"})
    secret = tools.execute_tool("create_support_request", {"student_id": "S1003", "category": "fee",
                                                           "description": "my password is abc123 please fix"})
    unknown = tools.execute_tool("create_support_request",
                                 {"student_id": "S9999", "category": "fee", "description": "long enough text"})
    assert all(not r["ok"] for r in (bad_cat, short, secret, unknown))


def test_schemas_match_registry():
    names = [s["function"]["name"] for s in tools.TOOL_SCHEMAS]
    assert sorted(tools.TOOL_REGISTRY) == sorted(tools.TOOL_META)
    # the model must NOT be able to create tickets directly: it may only prepare a draft
    assert sorted(names) == sorted(set(tools.TOOL_REGISTRY) - {"create_support_request"})
    assert "prepare_support_ticket" in names


def test_at_least_two_student_tools_exist():
    assert {"check_fee_status", "check_scholarship_status", "create_support_request",
            "get_student_record"} <= set(tools.TOOL_REGISTRY)
