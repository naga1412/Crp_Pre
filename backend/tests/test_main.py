"""Tests for the API surface that don't hit the network."""
import pytest
from fastapi import HTTPException

from main import (
    HTF_MAP,
    SYMBOL_RE,
    VALID_TIMEFRAMES,
    _build_symbol_terms,
    _news_cache,
    _news_cache_get,
    _news_cache_set,
    _word_re,
    is_crypto,
    validate_symbol,
)


# ── symbol validation ──────────────────────────────────────────────────────

@pytest.mark.parametrize("sym", [
    "BTC/USDT", "AAPL", "^GSPC", "GC=F", "BRK.B", "RELIANCE.NS", "ETH/BTC",
])
def test_validate_symbol_accepts(sym):
    assert validate_symbol(sym) == sym


@pytest.mark.parametrize("bad", [
    "", "<script>", "../etc/passwd", "foo//bar", "A" * 21,
    "BTC USDT", "rm -rf /", "; DROP TABLE",
])
def test_validate_symbol_rejects(bad):
    with pytest.raises(HTTPException) as exc:
        validate_symbol(bad)
    assert exc.value.status_code == 400


# ── word-boundary symbol terms (the bug that boosted "magic" for "GC") ─────

@pytest.mark.parametrize("sym, expected", [
    ("BTC/USDT", ["BTC", "bitcoin"]),
    ("ETH/USDT", ["ETH", "ethereum"]),
    ("V",        []),         # too short — must NOT leak
    ("MS",       []),
    ("GC=F",     []),         # short futures suffix — must NOT leak
    ("MSFT",     ["MSFT"]),
    ("^GSPC",    ["GSPC"]),
    ("RELIANCE.NS", ["RELIANCE"]),
])
def test_build_symbol_terms(sym, expected):
    assert _build_symbol_terms(sym) == expected


def test_word_re_does_word_boundary_match():
    pat = _word_re("GC")
    assert pat.search("Gold (GC) hits new high")
    assert not pat.search("the magic word")
    assert not pat.search("encrypted")


# ── caches ─────────────────────────────────────────────────────────────────

def test_news_cache_evicts_when_full():
    _news_cache.clear()
    from main import NEWS_CACHE_MAX_KEYS
    for i in range(NEWS_CACHE_MAX_KEYS + 50):
        _news_cache_set(f"k{i}", {"i": i})
    assert len(_news_cache) <= NEWS_CACHE_MAX_KEYS
    # First inserted must have been evicted
    assert _news_cache_get("k0") is None
    # Most recent insert should still be there
    assert _news_cache_get(f"k{NEWS_CACHE_MAX_KEYS + 49}") == {"i": NEWS_CACHE_MAX_KEYS + 49}


# ── routing helpers ────────────────────────────────────────────────────────

def test_is_crypto():
    assert is_crypto("BTC/USDT")
    assert is_crypto("ETH/BTC")
    assert not is_crypto("AAPL")
    assert not is_crypto("^GSPC")


def test_htf_map_has_no_self_references():
    for primary, htfs in HTF_MAP.items():
        assert primary not in htfs, f"{primary} lists itself as HTF"
        for h in htfs:
            assert h in VALID_TIMEFRAMES


def test_symbol_re_rejects_traversal_directly():
    """Belt-and-braces — even before the explicit `..` check, regex bounds it."""
    assert SYMBOL_RE.match("BTC/USDT")
    assert not SYMBOL_RE.match("BTC USDT")  # space disallowed
    assert not SYMBOL_RE.match("\x00BTC")
