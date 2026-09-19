import pandas as pd
import pytest
from src import data as d


def test_clean_prices_drops_bad_rows_and_duplicates():
    df = pd.DataFrame({"Close": [1.0, None, 3.0, -1.0, 5.0]},
                      index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-03"]))
    out = d.clean_prices(df)
    assert out.index.is_unique and out.index.is_monotonic_increasing and (out["Close"] > 0).all()


def test_no_silent_synthetic_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(d, "_download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(d.DataUnavailable):
        d.download_asset("NVDA", "2024-01-01", "2024-03-01", cache_dir=str(tmp_path))


def test_synthetic_is_opt_in_and_labelled(monkeypatch, tmp_path):
    monkeypatch.setattr(d, "_download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    df, source = d.download_asset("NVDA", "2024-01-01", "2024-03-01", cache_dir=str(tmp_path), allow_synthetic=True)
    assert source == "synthetic" and not df.empty
    assert not list(tmp_path.glob("*.csv"))            # fake data is never written to the cache
