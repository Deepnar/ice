"""Explicit terminal cloud faults, never fabricated answers or grades."""
from scripts.z1.campaign_recovery import TRANSIENT

POLICY = "ice-v3-isolated-cloud-failures-1"


def answer_fault(row):
    if row.get("error_status") in (401, 402, 403, 429) or row.get("error_type") == "ProviderAccessError":
        return False
    return (row.get("error_type") in TRANSIENT | {"CloudCompletionError", "UnconfirmedCloudCall"}
            or row.get("error_status") in (408, 500, 502, 503, 504))


def terminal_answer_fault(row):
    return (bool(row.get("error")) and row.get("fault_disposition") == "degraded_final"
            and row.get("cloud_attempts") == 2 and answer_fault(row))


def judge_fault(raw):
    if raw.get("operator_required"):
        return False
    return (raw.get("verdict") == "ERROR"
            and (raw.get("error_type") in TRANSIENT or raw.get("reason") in
                 {"api_http_408", "api_http_500", "api_http_502", "api_http_503", "api_http_504",
                  "unconfirmed_call", "incomplete_completion", "unparseable", "bad_verdict", "bad_absolute_grade",
                  "bad_reason", "inconsistent_failure_verdict", "inconsistent_grade_preference"}))


def terminal_judge_fault(row):
    return (row.get("winner") == "ERROR" and row.get("fault_disposition") == "degraded_final"
            and (row.get("reason") == "answer_failed" or any(
                judge_fault(order.get("raw") or {}) and order["raw"].get("cloud_attempts") == 2
                for order in row.get("order_verdicts") or [])))
