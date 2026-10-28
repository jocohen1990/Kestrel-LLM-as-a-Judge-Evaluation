"""Offline tests for the agent's tools (app/tools.py). No model, no API."""
import pytest

from app.paths import CORPUS_DIR
from app.tools import (
    METRIC_TABLE,
    METRICS,
    PERIODS,
    TOOL_DEFINITIONS,
    TOOLS,
    calculator,
    format_number,
    get_financial_metric,
    run_tool,
    search_documents,
)


# ---- Metric table: every figure must really be in the documents ---------------

@pytest.mark.parametrize("key", sorted(METRIC_TABLE))
def test_every_metric_value_is_quoted_from_its_document(key):
    entry = METRIC_TABLE[key]
    text = (CORPUS_DIR / entry.doc).read_text(encoding="utf-8")
    assert entry.evidence in text, f"{key}: evidence not found in {entry.doc}"
    # and the figure shown to the model is the one in the evidence
    assert entry.display.replace("$", "").split()[0] in entry.evidence


def test_metric_table_only_uses_known_metrics_and_periods():
    for metric, period in METRIC_TABLE:
        assert metric in METRICS and period in PERIODS


def test_lookup_returns_figure_and_source():
    r = get_financial_metric("revenue", "Q2 2026")
    assert not r.is_error
    assert "$116.0 million" in r.output and "Q2 2026 Earnings Release" in r.output


def test_gap_in_the_documents_is_not_reported_not_an_error():
    # FY2024 net income is the golden set's g031 trap: the documents don't give it.
    r = get_financial_metric("net_income", "FY2024")
    assert not r.is_error
    assert r.output.startswith("Not reported")


def test_not_reported_names_the_metric_the_way_a_reader_would():
    # Tools t2 (after Run 15): the agent copied "net_income" from this message
    # into its answer. Written out by hand, not built from METRIC_NAMES.
    assert (get_financial_metric("net_income", "FY2024").output
            == "Not reported: the documents don't give net income for FY2024.")
    assert (get_financial_metric("enterprise_customers", "Q1 2026").output
            == "Not reported: the documents don't give enterprise customers for Q1 2026.")


def test_no_not_reported_message_contains_a_metric_key():
    from app.tools import PERIODS
    for metric in METRICS:
        for period in PERIODS:
            if (metric, period) not in METRIC_TABLE:
                out = get_financial_metric(metric, period).output
                assert "_" not in out, out


def test_tools_version_is_recorded():
    from app.tools import TOOLS_VERSION
    assert TOOLS_VERSION == "t2"


@pytest.mark.parametrize("metric, period", [("revenues", "Q2 2026"), ("revenue", "Q2-2026"),
                                            ("revenue", "2026"), ("Revenue", "Q2 2026")])
def test_bad_arguments_are_errors_and_are_not_silently_fixed(metric, period):
    r = get_financial_metric(metric, period)
    assert r.is_error
    assert "Use one of" in r.output   # tells the model what IS allowed


# ---- Search ---------------------------------------------------------------------

def test_search_finds_the_right_paragraph():
    r = search_documents("revolving credit facility drawn")
    assert not r.is_error
    assert "$40 million was drawn" in r.output
    assert r.output.count("Kestrel Freight Analytics") >= 3   # 3 paragraphs, each titled


def test_search_tolerates_typos():   # retriever r2
    assert "Bramblecote" in search_documents("Bramblecoat aquisition").output


def test_empty_search_is_an_error():
    assert search_documents("   ").is_error


# ---- Calculator -------------------------------------------------------------------

@pytest.mark.parametrize("expression, expected", [
    ("108.3 + 116.0", "224.3"),
    ("(116.0 - 97.5) / 97.5 * 100", "18.974359"),
    ("72.1 - 71.4", "0.7"),          # floating point gives 0.699999..., shown rounded
    ("-3 + 2", "-1"),
    ("2 ** 3", "8"),
    ("10 / 4", "2.5"),
])
def test_calculator(expression, expected):
    r = calculator(expression)
    assert not r.is_error
    assert r.output == f"{expression} = {expected}"


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo hacked')",   # code, not arithmetic
    "open('secrets.txt').read()",
    "x + 1",                                   # names
    "$108.3 + $116.0",                         # dollar signs
    "2,850 + 1",                               # thousands separators parse as a tuple
    "10 ** 1000",                              # runaway exponent
    "1 / 0",
    "",
    "1 +",
    "'a' * 3",
])
def test_calculator_rejects_anything_but_arithmetic(expression):
    assert calculator(expression).is_error


def test_format_number():
    assert format_number(12.0) == "12"
    assert format_number(-0.0000001) == "0"
    assert format_number(1 / 3) == "0.333333"


# ---- Dispatch and definitions -------------------------------------------------------

def test_run_tool_turns_model_mistakes_into_error_results():
    assert run_tool("web_search", {"query": "x"}).is_error                 # no such tool
    assert run_tool("calculator", {}).is_error                             # missing argument
    assert run_tool("calculator", {"expr": "1+1"}).is_error                # misnamed argument
    assert run_tool("calculator", {"expression": "1+1", "x": 1}).is_error  # extra argument
    assert run_tool("calculator", {"expression": "1+1"}).output == "1+1 = 2"


def test_every_definition_matches_a_tool_and_its_arguments():
    import inspect
    assert {d["name"] for d in TOOL_DEFINITIONS} == set(TOOLS)
    for d in TOOL_DEFINITIONS:
        params = set(inspect.signature(TOOLS[d["name"]]).parameters)
        assert set(d["input_schema"]["properties"]) == params
        assert set(d["input_schema"]["required"]) == params


def test_metric_definition_enums_match_the_table():
    props = TOOL_DEFINITIONS[0]["input_schema"]["properties"]
    assert props["metric"]["enum"] == list(METRICS)
    assert props["period"]["enum"] == list(PERIODS)
