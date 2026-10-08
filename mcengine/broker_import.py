"""Import a depot export (CSV) from a broker for the annual review.

Broker exports differ a lot, so the parser is tolerant:

* separator ``;``, ``,`` or tab; encodings UTF-8 / UTF-8-BOM / Windows-1252
* intro lines above the header: the first line that names an ISIN column is taken as the header
* German (``1.234,56``) and English (``1,234.56``) number formats, ``€``/``EUR``/``%`` suffixes
* column names in German or English. The market value comes from a value column or from
  quantity × price; the cost basis from a cost column or from quantity × average purchase price.

Each position is mapped to one of the configured asset classes by ISIN (your ETF selection
and the shipped shortlist), then by a keyword guess on the name. Anything else stays
unassigned for you to choose. If your broker has no export, fill in ``template_csv()``.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

import pandas as pd

ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

SYNONYMS: dict[str, list[str]] = {
    "isin": ["isin", "isin/wkn", "wertpapierkennnummer isin"],
    "name": ["bezeichnung", "name", "wertpapier", "wertpapiername", "instrument", "security", "titel",
             "produkt", "description", "beschreibung", "position"],
    "value": ["kurswert in eur", "kurswert", "marktwert", "marktwert in eur", "wert in eur", "aktueller wert",
              "depotwert", "bewertung", "market value", "current value", "value", "wert", "positionswert",
              "gesamtwert", "kurswert eur"],
    "basis": ["einstandswert", "einstandswert in eur", "kaufwert", "kaufwert in eur", "anschaffungswert",
              "anschaffungskosten", "einstand", "einstandskurswert", "investiert", "investierter betrag",
              "cost basis", "purchase value", "invested", "book cost", "buy value", "kaufsumme"],
    "quantity": ["stück", "stueck", "stück/nominale", "anzahl", "nominale", "nominal", "menge", "bestand",
                 "quantity", "shares", "units", "stk", "stk."],
    "price": ["aktueller kurs", "kurs", "letzter kurs", "schlusskurs", "price", "last price",
              "current price"],
    "avg_price": ["einstandskurs", "kaufkurs", "durchschnittskurs", "ø kaufkurs", "ø-kaufkurs",
                  "durchschnittlicher kaufkurs", "avg price", "average price", "average cost", "cost price"],
}

KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("linker", ("inflation", "linker", "inflationsgesch")),
    ("gold", ("gold",)),
    ("cash", ("overnight", "money market", "geldmarkt", "€str", "estr", "tagesgeld")),
    ("nominal_bond", ("government bond", "govt bond", "staatsanleihe", "treasury", "sovereign", "govies",
                      "euro government", "anleihen")),
    ("equity", ("all-world", "all world", "acwi", "msci world", "ftse", "world", "s&p", "stoxx", "equity",
                "aktien", "msci")),
]


@dataclass
class ImportResult:
    positions: pd.DataFrame            # isin, name, value, basis, asset_key, how
    columns: dict[str, str]            # field -> column name used
    warnings: list[str] = field(default_factory=list)

    def totals(self) -> tuple[dict[str, float], dict[str, float]]:
        """Market value and cost basis per asset class (unassigned/ignored positions excluded)."""
        p = self.positions[self.positions["asset_key"].notna() & (self.positions["asset_key"] != "ignore")]
        g = p.groupby("asset_key")[["value", "basis"]].sum()
        return g["value"].to_dict(), g["basis"].to_dict()


# ----------------------------------------------------------------------------- parsing helpers
def parse_number(text, decimal: str | None = None) -> float | None:
    """Parse '1.234,56 €', '1,234.56', '-12,5 %', '(3.4)'; returns None if empty/unparseable."""
    if text is None:
        return None
    if isinstance(text, int | float):
        return None if pd.isna(text) else float(text)
    s = str(text).strip().replace(" ", "").replace(" ", "")
    s = re.sub(r"(EUR|€|USD|\$|%|Stk\.?|St\.?)", "", s, flags=re.IGNORECASE)
    if not s or s in {"-", "–", "n/a", "N/A"}:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
    elif "," in s:
        dec = decimal or ("." if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s) else ",")
    elif "." in s:
        dec = "." if decimal != "," or not re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s) else ","
    else:
        dec = "."
    thou = "." if dec == "," else ","
    s = s.replace(thou, "").replace(dec, ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _norm(col: str) -> str:
    """Lower-case header without trailing currency notes ('Kurswert in EUR' -> 'kurswert')."""
    s = re.sub(r"\s+", " ", str(col).strip().strip('"').lower().replace("_", " ")).strip()
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\s*(\([^)]*\)|\bin eur|\beur|€)\s*$", "", s).strip()
    return s


def read_table(raw: bytes) -> tuple[pd.DataFrame, str]:
    """Locate the header row (first line mentioning ISIN) and read the table. Returns (df, separator)."""
    text = _decode(raw)
    lines = text.splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if "isin" in ln.lower()), None)
    if header_idx is None:
        raise ValueError("no column named 'ISIN' found – is this a depot/position export?")
    header = lines[header_idx]
    sep = max([";", "\t", ","], key=header.count)
    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(body), sep=sep, dtype=str, quoting=csv.QUOTE_MINIMAL, skip_blank_lines=True,
                     on_bad_lines="skip", engine="python")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed")]]
    return df, sep


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Field -> column name, by exact synonym first, then by 'starts with' a synonym."""
    norm = {_norm(c): c for c in df.columns}
    out: dict[str, str] = {}
    for fld, names in SYNONYMS.items():
        for n in names:  # exact matches in priority order
            if n in norm and norm[n] not in out.values():
                out[fld] = norm[n]
                break
        if fld in out:
            continue
        for n in (n for n in names if len(n) >= 6):  # short names ('wert', 'kurs') only match exactly
            hit = next((orig for k, orig in norm.items()
                        if k.startswith(n) and orig not in out.values()), None)
            if hit:
                out[fld] = hit
                break
    return out


def guess_asset(name: str, kinds: dict[str, str]) -> str | None:
    """Asset key whose kind matches a keyword in the product name."""
    low = (name or "").lower()
    by_kind = {kind: key for key, kind in kinds.items()}
    for kind, words in KEYWORDS:
        if any(w in low for w in words) and kind in by_kind:
            return by_kind[kind]
    return None


# ----------------------------------------------------------------------------- main entry
def parse_depot_csv(raw: bytes, isin_map: dict[str, str], kinds: dict[str, str]) -> ImportResult:
    """Parse a broker export. ``isin_map``: ISIN -> asset key; ``kinds``: asset key -> asset kind."""
    df, sep = read_table(raw)
    cols = detect_columns(df)
    if "isin" not in cols:
        raise ValueError("could not identify the ISIN column")
    decimal = "," if sep == ";" else None
    warnings: list[str] = []

    def num(field_name: str) -> pd.Series:
        if field_name not in cols:
            return pd.Series([None] * len(df), index=df.index, dtype=object)
        return df[cols[field_name]].map(lambda x: parse_number(x, decimal))

    isin = df[cols["isin"]].fillna("").astype(str).str.strip().str.upper().str.extract(
        r"([A-Z]{2}[A-Z0-9]{9}[0-9])")[0]
    value, basis = num("value"), num("basis")
    qty, price, avg = num("quantity"), num("price"), num("avg_price")
    value = value.where(value.notna(), qty.astype(float) * price.astype(float))
    basis = basis.where(basis.notna(), qty.astype(float) * avg.astype(float))
    if "value" not in cols and not {"quantity", "price"} <= set(cols):
        raise ValueError("found no market-value column (or quantity and price) – columns: "
                         + ", ".join(df.columns))
    if "basis" not in cols and not {"quantity", "avg_price"} <= set(cols):
        warnings.append("No cost-basis column found: cost basis set to market value (no unrealised gain). "
                        "Enter it by hand for an accurate tax estimate.")

    if "name" in cols:
        names = df[cols["name"]].fillna("").astype(str).str.strip()
    else:
        names = pd.Series("", index=df.index)
    rows = []
    for i in df.index:
        if pd.isna(isin[i]) or value[i] is None or pd.isna(value[i]):
            continue  # totals, sub-headers, cash lines without ISIN
        key, how = isin_map.get(isin[i]), "ISIN match"
        if key is None:
            key = guess_asset(names[i], kinds)
            how = "guessed from name – please check" if key else "unassigned"
        b = basis[i]
        rows.append({"isin": isin[i], "name": names[i], "value": float(value[i]),
                     "basis": float(value[i]) if b is None or pd.isna(b) else float(b),
                     "asset_key": key, "how": how})
    pos = pd.DataFrame(rows, columns=["isin", "name", "value", "basis", "asset_key", "how"])
    if pos.empty:
        raise ValueError("no positions with an ISIN and a value found")
    pos = pos.groupby(["isin"], as_index=False).agg(
        {"name": "first", "value": "sum", "basis": "sum", "asset_key": "first", "how": "first"})
    if pos["asset_key"].isna().any():
        n = int(pos["asset_key"].isna().sum())
        warnings.append(f"{n} position(s) could not be assigned to an asset class.")
    return ImportResult(pos, cols, warnings)


def template_csv(asset_names: dict[str, str]) -> str:
    """Simple template for brokers without an export (one line per product)."""
    lines = ["ISIN;Name;Market value;Cost basis"]
    lines.append("IE00BK5BQT80;Vanguard FTSE All-World UCITS ETF (USD) Acc;30.000,00;27.000,00")
    lines.append("LU0290355717;Xtrackers II Eurozone Government Bond UCITS ETF 1C;10.000,00;9.800,00")
    return "\n".join(lines) + "\n"
