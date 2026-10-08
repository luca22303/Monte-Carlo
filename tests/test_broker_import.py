from pathlib import Path

import pandas as pd
import pytest

from mcengine.broker_import import detect_columns, parse_depot_csv, parse_number, template_csv
from mcengine.config import SimConfig
from mcengine.etfs import load_candidates

DATA = Path(__file__).parent / "data"
CFG = SimConfig()
KINDS = {a.key: a.kind for a in CFG.assets}
ISIN_MAP = dict(zip(load_candidates()["isin"], load_candidates()["asset_key"], strict=True))


@pytest.mark.parametrize("text,decimal,expected", [
    ("1.234,56", ",", 1234.56), ("1.234,56 €", None, 1234.56), ("1,234.56", None, 1234.56),
    ("35.575,00", ",", 35575.0), ("12,5", ",", 12.5), ("12.5", ",", 12.5), ("1.234", ",", 1234.0),
    ("-3,40 %", ",", -3.4), ("(3.4)", None, -3.4), ("250,000", ",", 250.0), ("1,000", None, 1000.0),
    ("", ",", None), ("-", None, None), ("n/a", None, None), ("EUR 1.000,00", ",", 1000.0),
])
def test_parse_number(text, decimal, expected):
    assert parse_number(text, decimal) == (pytest.approx(expected) if expected is not None else None)


def test_german_export_with_preamble_and_totals():
    res = parse_depot_csv((DATA / "depot_de_cp1252.csv").read_bytes(), ISIN_MAP, KINDS)
    pos = res.positions.set_index("isin")
    assert len(pos) == 5  # totals row skipped
    assert pos.loc["IE00BK5BQT80", "value"] == pytest.approx(35575.0)
    assert pos.loc["IE00BK5BQT80", "basis"] == pytest.approx(29525.0)
    assert pos.loc["IE00BK5BQT80", "asset_key"] == "equity"
    assert pos.loc["DE000EWG2LD7", "asset_key"] == "gold"
    assert pd.isna(pos.loc["LU1234567890", "asset_key"])  # unknown fund stays unassigned
    assert res.columns["value"] == "Kurswert in EUR" and res.columns["basis"] == "Einstandswert in EUR"
    values, basis = res.totals()
    assert values["gov_bonds"] == pytest.approx(12108.25) and basis["linkers"] == pytest.approx(5760.0)
    assert "LU1234567890" not in values and any("could not be assigned" in w for w in res.warnings)


def test_english_export_quantity_times_price_and_name_guess():
    res = parse_depot_csv((DATA / "depot_en_qty.csv").read_bytes(), ISIN_MAP, KINDS)
    pos = res.positions.set_index("isin")
    assert pos.loc["IE00B4L5Y983", "value"] == pytest.approx(1000.5 * 95.20)
    assert pos.loc["IE00B4L5Y983", "basis"] == pytest.approx(1000.5 * 80.10)
    assert pos.loc["XS1234567891", "asset_key"] == "gold"
    assert "guessed" in pos.loc["XS1234567891", "how"]


def test_template_roundtrip():
    raw = template_csv({}).encode()
    res = parse_depot_csv(raw, ISIN_MAP, KINDS)
    assert res.positions["value"].sum() == pytest.approx(40000.0)
    assert not res.warnings


def test_missing_basis_warns_and_defaults_to_value():
    raw = b"ISIN;Bezeichnung;Kurswert\nIE00BK5BQT80;Vanguard;1.000,00\n"
    res = parse_depot_csv(raw, ISIN_MAP, KINDS)
    assert res.positions["basis"].iloc[0] == pytest.approx(1000.0)
    assert any("cost-basis" in w for w in res.warnings)


def test_errors():
    with pytest.raises(ValueError, match="ISIN"):
        parse_depot_csv(b"Name;Wert\nFoo;1\n", ISIN_MAP, KINDS)
    with pytest.raises(ValueError, match="market-value"):
        parse_depot_csv(b"ISIN;Name\nIE00BK5BQT80;x\n", ISIN_MAP, KINDS)


def test_detect_columns_does_not_confuse_wertpapierart():
    df = pd.DataFrame(columns=["Wertpapierart", "Bezeichnung", "ISIN", "Kurs (EUR)", "Stück"])
    cols = detect_columns(df)
    assert "value" not in cols and cols["price"] == "Kurs (EUR)" and cols["name"] == "Bezeichnung"
