import pytest
from portfolio_rl.data import fetch_daily

def test_fetch_daily(monkeypatch):
    dummy = {"Time Series (Daily)": {"2025-01-01": {"5. adjusted close": "100.0"}}}
    class DummyResp:
        def raise_for_status(self): pass
        def json(self): return dummy
    monkeypatch.setattr("requests.get", lambda *a, **k: DummyResp())
    df = fetch_daily("FAKE", outputsize="compact")
    assert df.iloc[0]["5. adjusted close"] == 100.0
