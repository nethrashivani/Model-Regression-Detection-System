"""
Fills the entity placeholders (e.g. {{Order Number}}) that Bitext leaves in
its instruction text. These are template slots in the *real* dataset, not
content we're inventing — we're just substituting plausible concrete values
so the text reads like an actual message instead of a template.

Deterministic + seeded, so re-running produces the same filled dataset.
"""

from __future__ import annotations

import random
import re

_SEED = 42

_ACCOUNT_TYPES = ["Premium", "Standard", "Business", "Free Tier", "Pro"]
_ACCOUNT_CATEGORIES = ["personal", "business", "student", "family"]
_CURRENCY_SYMBOLS = ["$", "€", "£"]
_CITIES = ["Austin", "Manchester", "Toronto", "Berlin", "Chennai", "Sydney", "Denver"]
_COUNTRIES = ["the United States", "the United Kingdom", "Canada", "Germany", "India", "Australia"]
_PERSON_NAMES = ["Alex Morgan", "Priya Nair", "Daniel Kim", "Sofia Reyes", "James Whitfield", "Mei Lin"]


def _random_order_number(rng: random.Random) -> str:
    return f"ORD-{rng.randint(100000, 999999)}"


def _random_invoice_number(rng: random.Random) -> str:
    return f"INV-{rng.randint(10000, 99999)}"


def _random_refund_amount(rng: random.Random, currency: str) -> str:
    return f"{currency}{rng.choice([19.99, 45.00, 89.50, 120.00, 12.75, 250.00])}"


def fill_entities(text: str, row_id: str) -> str:
    """Replace {{Entity Name}} placeholders with a deterministic realistic value."""
    rng = random.Random(f"{_SEED}-{row_id}")
    currency = rng.choice(_CURRENCY_SYMBOLS)

    replacements = {
        "Order Number": lambda: _random_order_number(rng),
        "Invoice Number": lambda: _random_invoice_number(rng),
        "Account Type": lambda: rng.choice(_ACCOUNT_TYPES),
        "Account Category": lambda: rng.choice(_ACCOUNT_CATEGORIES),
        "Currency Symbol": lambda: currency,
        "Delivery City": lambda: rng.choice(_CITIES),
        "Delivery Country": lambda: rng.choice(_COUNTRIES),
        "Person Name": lambda: rng.choice(_PERSON_NAMES),
        "Refund Amount": lambda: _random_refund_amount(rng, currency),
    }

    def _sub(match: re.Match) -> str:
        key = match.group(1)
        fn = replacements.get(key)
        return fn() if fn else match.group(0)

    return re.sub(r"\{\{(.+?)\}\}", _sub, text)
