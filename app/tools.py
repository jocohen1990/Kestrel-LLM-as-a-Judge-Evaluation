"""The agent's tools (Phase 4).

Three tools, chosen so that each kind of question has one RIGHT tool, which is
what makes tool choice gradable:

  get_financial_metric  one reported figure for one period, from a small table.
                        The right tool for "What was Q2 2026 revenue?".
  search_documents      BM25 keyword search over the four filings. The right
                        tool for people, products, risks, guidance, debt.
  calculator            arithmetic. The right tool for sums, differences and
                        growth rates. Run 4's hardest questions (g023, g024)
                        were the ones that needed arithmetic, which LLMs get
                        wrong in their heads; with a calculator available,
                        "did it use the calculator?" becomes a checkable step.

Every tool is plain, deterministic Python: same arguments, same result, no API
calls. So the only thing that varies between runs is the model's decisions,
and the tools themselves can be tested exhaustively offline.

Tool results are strings, because that is what the model reads. A result is
either ordinary text or an error (is_error=True), which the API passes to the
model so it can correct its arguments and try again. Bad arguments are never
silently "fixed" here: a retry after an error is a visible step in the
trajectory, and the eval can count it.
"""
import ast
import operator
from dataclasses import dataclass

from app.paths import CORPUS_DIR
from app.retriever import Retriever, load_chunks

# ---- Metric table -----------------------------------------------------------
# Every figure is copied from the corpus, and `evidence` is the exact text it
# came from. tests/test_tools.py checks that each evidence string really is in
# the named document, so a typo here fails the suite instead of quietly
# teaching the agent a wrong number.

PERIODS = ("FY2024", "FY2025", "Q1 2025", "Q2 2025", "Q1 2026", "Q2 2026")

METRICS = {
    "revenue": "Total revenue, in millions of US dollars",
    "net_income": "Net income, in millions of US dollars",
    "diluted_eps": "Diluted earnings per share, in US dollars",
    "gross_margin": "Gross margin, as a percentage",
    "enterprise_customers": "Enterprise customers at the end of the period",
}

DOC_TITLES = {
    "annual_report_2025.md": "Annual Report 2025",
    "q1_2026_earnings.md": "Q1 2026 Earnings Release",
    "q2_2026_earnings.md": "Q2 2026 Earnings Release",
    "risk_factors_2026.md": "Risk Factors 2026",
}


@dataclass(frozen=True)
class MetricValue:
    display: str     # how the figure is shown to the model
    value: float     # the number itself, for graders
    doc: str         # the corpus file it came from
    evidence: str    # exact text in that file


METRIC_TABLE: dict[tuple[str, str], MetricValue] = {
    ("revenue", "FY2024"): MetricValue("$349.7 million", 349.7, "annual_report_2025.md", "from $349.7 million in fiscal year 2024"),
    ("revenue", "FY2025"): MetricValue("$412.6 million", 412.6, "annual_report_2025.md", "total revenue was $412.6 million"),
    ("revenue", "Q1 2025"): MetricValue("$89.5 million", 89.5, "q1_2026_earnings.md", "from $89.5 million in Q1 2025"),
    ("revenue", "Q1 2026"): MetricValue("$108.3 million", 108.3, "q1_2026_earnings.md", "Q1 2026 revenue of $108.3 million"),
    ("revenue", "Q2 2025"): MetricValue("$97.5 million", 97.5, "q2_2026_earnings.md", "from $97.5 million in Q2 2025"),
    ("revenue", "Q2 2026"): MetricValue("$116.0 million", 116.0, "q2_2026_earnings.md", "Q2 2026 revenue of $116.0 million"),
    ("net_income", "FY2025"): MetricValue("$38.9 million", 38.9, "annual_report_2025.md", "Net income for fiscal year 2025 was $38.9 million"),
    ("net_income", "Q1 2026"): MetricValue("$9.2 million", 9.2, "q1_2026_earnings.md", "Net income was $9.2 million"),
    ("net_income", "Q2 2026"): MetricValue("$11.4 million", 11.4, "q2_2026_earnings.md", "Net income was $11.4 million"),
    ("diluted_eps", "Q1 2026"): MetricValue("$0.31", 0.31, "q1_2026_earnings.md", "(EPS) were $0.31"),
    ("diluted_eps", "Q2 2026"): MetricValue("$0.38", 0.38, "q2_2026_earnings.md", "(EPS) were $0.38"),
    ("gross_margin", "Q1 2025"): MetricValue("69.8%", 69.8, "q1_2026_earnings.md", "compared with 69.8% in Q1 2025"),
    ("gross_margin", "Q1 2026"): MetricValue("71.4%", 71.4, "q1_2026_earnings.md", "gross margin for Q1 2026 was 71.4%"),
    ("gross_margin", "Q2 2026"): MetricValue("72.1%", 72.1, "q2_2026_earnings.md", "gross margin for Q2 2026 was 72.1%"),
    ("enterprise_customers", "FY2025"): MetricValue("2,850", 2850, "annual_report_2025.md", "2,850 enterprise customers at the end of 2025"),
    ("enterprise_customers", "Q2 2026"): MetricValue("3,120", 3120, "q2_2026_earnings.md", "ended Q2 2026 with 3,120 enterprise customers"),
}


@dataclass
class ToolResult:
    output: str
    is_error: bool = False


def get_financial_metric(metric: str, period: str) -> ToolResult:
    """Look up one figure. A real gap in the documents (net income for FY2024)
    is NOT an error: it's a true answer, "not reported", and the agent should
    then say it doesn't know rather than estimate."""
    if metric not in METRICS:
        return ToolResult(f"Unknown metric {metric!r}. Use one of: {', '.join(METRICS)}.", True)
    if period not in PERIODS:
        return ToolResult(f"Unknown period {period!r}. Use one of: {', '.join(PERIODS)}.", True)
    found = METRIC_TABLE.get((metric, period))
    if found is None:
        return ToolResult(f"Not reported: the documents don't give {metric} for {period}.")
    return ToolResult(f"{metric}, {period}: {found.display} (source: {DOC_TITLES[found.doc]})")


# ---- Document search ----------------------------------------------------------

SEARCH_K = 3
_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    """Built once, on first use. Retriever r2 (typo-tolerant BM25). No query
    rewrite (r3): the agent writes its own search queries, and how well it does
    that is part of what Phase 4 measures."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever(load_chunks(CORPUS_DIR), typo_tolerance=True)
    return _retriever


def search_documents(query: str) -> ToolResult:
    if not isinstance(query, str) or not query.strip():
        return ToolResult("The query is empty. Search with a few keywords.", True)
    chunks = _get_retriever().search(query, k=SEARCH_K)
    return ToolResult("\n\n".join(c.text for c in chunks))


# ---- Calculator ---------------------------------------------------------------
# A safe calculator: the expression is parsed into a syntax tree and only
# numbers, + - * / ** % and parentheses are allowed. Python's eval() would run
# ANY code the model wrote (or that a prompt injection got it to write), such as
# __import__('os').system(...). Parsing first means anything that isn't plain
# arithmetic is rejected before it can run.

_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
CALC_HELP = "Use plain numbers (no $, % or commas) with + - * / ** and parentheses."


class _NotArithmetic(ValueError):
    pass


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise _NotArithmetic("Exponents above 10 aren't supported.")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise _NotArithmetic(CALC_HELP)


def format_number(x: float) -> str:
    """At most 6 decimal places, no trailing zeros: 224.3, 0.192308, 12."""
    text = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def calculator(expression: str) -> ToolResult:
    if not isinstance(expression, str) or not expression.strip():
        return ToolResult(f"The expression is empty. {CALC_HELP}", True)
    if len(expression) > 200:
        return ToolResult("The expression is too long (200 characters at most).", True)
    try:
        value = _evaluate(ast.parse(expression, mode="eval"))
    except ZeroDivisionError:
        return ToolResult("Division by zero.", True)
    except _NotArithmetic as exc:
        return ToolResult(str(exc), True)
    except SyntaxError:
        return ToolResult(f"Couldn't read {expression!r}. {CALC_HELP}", True)
    return ToolResult(f"{expression} = {format_number(value)}")


# ---- Definitions sent to the model --------------------------------------------
# The descriptions are part of the prompt: they are how the model decides which
# tool fits a question. Like prompts, they are versioned with the agent (see
# app/agent.py), so a tool-choice change can be traced to a wording change.

TOOL_DEFINITIONS = [
    {
        "name": "get_financial_metric",
        "description": ("Look up one reported figure for one period from Kestrel's financial "
                        "filings: revenue, net income, diluted EPS, gross margin, or enterprise "
                        "customers. Returns the figure and its source document, or 'Not "
                        "reported' when the filings don't give it. Actual results only, not "
                        "guidance or forecasts."),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": list(METRICS),
                           "description": "; ".join(f"{k}: {v}" for k, v in METRICS.items())},
                "period": {"type": "string", "enum": list(PERIODS),
                           "description": "Fiscal year (FY2025) or quarter (Q2 2026)"},
            },
            "required": ["metric", "period"],
        },
    },
    {
        "name": "search_documents",
        "description": ("Keyword search over Kestrel's four filings (Annual Report 2025, Q1 and "
                        "Q2 2026 earnings releases, Risk Factors 2026). Returns the 3 most "
                        "relevant paragraphs. Use it for anything the metric tool doesn't "
                        "cover: people, products, acquisitions, guidance, debt, risks."),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Keywords in English, e.g. 'credit facility drawn'"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": ("Evaluate an arithmetic expression, such as '(116.0 - 97.5) / 97.5 * 100'. "
                        "Use it for every sum, difference, ratio or growth rate. " + CALC_HELP),
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]

TOOLS = {
    "get_financial_metric": get_financial_metric,
    "search_documents": search_documents,
    "calculator": calculator,
}


def run_tool(name: str, args: dict) -> ToolResult:
    """Run a tool by name. Anything the model gets wrong (a tool that doesn't
    exist, a missing or extra argument) comes back as an error result, never an
    exception, so one bad call can't crash an eval run."""
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(f"There is no tool named {name!r}.", True)
    try:
        return tool(**(args or {}))
    except TypeError as exc:   # missing, extra or misnamed arguments
        return ToolResult(f"Bad arguments for {name}: {exc}", True)
