"""API prices, so a scorecard can say what each model costs, not just how it scores.

Prices change, so each one is stored with the date it was checked. A results
file only stores token counts; cost is worked out from this table when the
scorecard is built. If a price changes, update the table and rebuild the
scorecard (`python compare_models.py --from-files`): no API calls needed.

All prices are US dollars per million tokens (MTok), standard (not batch) API.

Source: https://platform.claude.com/docs/en/about-claude/pricing, checked by
hand on 2026-10-29 for every model below (every column matched). Cache writes
use the 5-minute rate; the 1-hour rate isn't used anywhere in this project.
"""
from dataclasses import dataclass

PRICES_CHECKED = "2026-10-29"


@dataclass(frozen=True)
class Price:
    input: float          # uncached input tokens
    output: float
    cache_read: float     # input tokens read from the prompt cache
    cache_write: float    # input tokens written to the cache (5-minute cache)
    note: str = ""


PRICES: dict[str, Price] = {
    "claude-haiku-4-5": Price(1.00, 5.00, 0.10, 1.25),
    # Haiku 5.5 is priced by prompt length. Every prompt in this project is far
    # under 100,000 tokens (the biggest are about 2,000), so only the lower tier
    # is used. Above 100K it's $0.50 / $2.50.
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01, 0.125,
                              note="prompts up to 100K tokens"),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.10, 2.50),
    "claude-opus-5-5": Price(4.00, 20.00, 0.20, 5.00),
}


def base_name(model: str) -> str:
    """'claude-haiku-4-5-20251001' -> 'claude-haiku-4-5'.

    Model IDs sometimes carry a date snapshot on the end. The price is the
    same for the snapshot and the alias, so the table is keyed by the alias.
    """
    parts = model.split("-")
    if parts and len(parts[-1]) == 8 and parts[-1].isdigit():
        parts = parts[:-1]
    return "-".join(parts)


def price_for(model: str) -> Price | None:
    """The model's price, or None if it isn't in the table (cost shows as n/a)."""
    return PRICES.get(base_name(model))


def cost_usd(model: str, input_tokens: int, output_tokens: int,
             cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float | None:
    """Dollar cost of a set of calls.

    input_tokens is ALL input, cached or not, the way app/llm.py records it
    (see _input_usage there). So the cached parts are taken out before the
    normal input price is applied, and charged at their own prices.
    """
    price = price_for(model)
    if price is None:
        return None
    uncached = input_tokens - cache_read_tokens - cache_write_tokens
    if uncached < 0:
        raise ValueError("cached tokens can't be more than the total input tokens")
    total = (uncached * price.input
             + cache_read_tokens * price.cache_read
             + cache_write_tokens * price.cache_write
             + output_tokens * price.output)
    return total / 1_000_000
