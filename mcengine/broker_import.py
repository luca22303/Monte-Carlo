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
    positions: pd.DataFrame            # isin, name, value, basis, asset_key, how (+ shares, price_date)
    columns: dict[str, str]            # field -> column name used
    warnings: list[str] = field(default_factory=list)
    kind: str = "depot"                # "depot" (holdings overview) or "transactions" (rebuilt from bookings)

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
    def is_header(ln: str) -> bool:
        low = ln.lower()
        return "isin" in low or ("symbol" in low and "shares" in low)

    header_idx = next((i for i, ln in enumerate(lines) if is_header(ln)), None)
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


# ----------------------------------------------------------------------------- transaction exports
TX_REQUIRED = {"symbol", "shares", "amount", "type"}


def is_transaction_export(df: pd.DataFrame) -> bool:
    """Trade Republic style export: one row per booking with symbol (ISIN), shares and amount."""
    cols = {c.strip().lower() for c in df.columns}
    return TX_REQUIRED <= cols and ("category" in cols or "transaction_id" in cols)


def parse_transactions(df: pd.DataFrame, isin_map: dict[str, str], kinds: dict[str, str]) -> ImportResult:
    """Rebuild holdings from a transaction export.

    * Units per ISIN = sum of all share changes of trades (buys +, sells -).
    * Cost basis = FIFO over the purchase lots, purchase fees included (German tax rule).
    * Market value = units x the most recent trade price in the export. The export has no
      current prices, so these are dated and should be overwritten with today's values.
    * Cash = sum of amount + fee + tax over all bookings (card payments, transfers, interest,
      dividends, trades...), i.e. the cash balance at the broker.
    """
    d = df.rename(columns={c: c.strip().lower() for c in df.columns}).copy()
    num = {c: pd.to_numeric(d[c], errors="coerce").fillna(0.0) if c in d else pd.Series(0.0, index=d.index)
           for c in ("shares", "price", "amount", "fee", "tax")}
    when = d["datetime"] if "datetime" in d else d.get("date", pd.Series("", index=d.index))
    order = when.fillna("").astype(str).sort_values(kind="stable").index
    def text(col: str) -> pd.Series:
        return d[col].fillna("").astype(str).str.strip() if col in d else pd.Series("", index=d.index)

    isin, category, ttype = text("symbol").str.upper(), text("category").str.upper(), text("type").str.upper()
    names, aclass = text("name"), text("asset_class").str.upper()
    dates = text("date") if "date" in d else when.fillna("").astype(str).str[:10]

    is_trade = ((category == "TRADING") | ttype.isin(["BUY", "SELL"])) & isin.str.match(ISIN_RE) \
        & (num["shares"] != 0)
    lots: dict[str, list[list[float]]] = {}
    last_px: dict[str, tuple[float, str]] = {}
    meta: dict[str, tuple[str, str]] = {}
    n_trades = 0
    for i in order:
        if not is_trade[i]:
            continue
        n_trades += 1
        code, sh = isin[i], float(num["shares"][i])
        meta[code] = (names[i], aclass[i])
        if num["price"][i] > 0:
            last_px[code] = (float(num["price"][i]), dates[i])
        book = lots.setdefault(code, [])
        if sh > 0:
            book.append([sh, max(-(num["amount"][i] + num["fee"][i]), 0.0)])
        else:
            q = -sh
            while q > 1e-12 and book:
                take = min(q, book[0][0])
                book[0][1] *= 1.0 - take / book[0][0]
                book[0][0] -= take
                q -= take
                if book[0][0] <= 1e-12:
                    book.pop(0)
    if n_trades == 0:
        raise ValueError("the transaction export contains no buy or sell trades with an ISIN")

    equity_key = next((k for k, kind in kinds.items() if kind == "equity"), None)
    cash_key = next((k for k, kind in kinds.items() if kind == "cash"), None)
    rows, stale = [], []
    for code, book in lots.items():
        units = sum(b[0] for b in book)
        if units <= 1e-9:
            continue  # fully sold
        basis = sum(b[1] for b in book)
        name, cls = meta[code]
        px, px_date = last_px.get(code, (0.0, ""))
        value = units * px if px > 0 else basis
        stale.append(px_date)
        key, how = isin_map.get(code), "ISIN match"
        if key is None and cls == "STOCK" and equity_key:
            key, how = equity_key, "single stock → equity"
        elif key is None:
            key = guess_asset(name, kinds)
            how = "guessed from name – please check" if key else "unassigned"
        rows.append({"isin": code, "name": name, "value": value, "basis": basis, "asset_key": key, "how": how,
                     "shares": units, "price_date": px_date})
    cash = float((num["amount"] + num["fee"] + num["tax"]).sum())
    if abs(cash) > 0.005:
        rows.append({"isin": "", "name": "Cash balance at the broker", "value": cash, "basis": cash,
                     "asset_key": cash_key, "how": "sum of all bookings", "shares": float("nan"),
                     "price_date": ""})
    columns = ["isin", "name", "value", "basis", "asset_key", "how", "shares", "price_date"]
    pos = pd.DataFrame(rows, columns=columns)

    first, last = dates[order[0]], dates[order[-1]]
    warnings = [
        f"Transaction export recognised ({len(d)} bookings, {n_trades} trades, {first} to {last}). "
        "Holdings and FIFO cost basis were rebuilt from the trades. **The export contains no current "
        "prices:** market values use each product's last trade price (see 'Price from'). Overwrite them "
        "with today's values from your app.",
    ]
    if cash < 0:
        warnings.append("The computed cash balance is negative; check the export covers your whole history.")
    if pos["asset_key"].isna().any():
        n = int(pos["asset_key"].isna().sum())
        warnings.append(f"{n} position(s) could not be assigned to an asset class.")
    cols = {"isin": "symbol", "name": "name", "quantity": "shares", "price": "price",
            "basis": "amount + fee (FIFO)"}
    return ImportResult(pos, cols, warnings, kind="transactions")


# ----------------------------------------------------------------------------- main entry
def parse_depot_csv(raw: bytes, isin_map: dict[str, str], kinds: dict[str, str]) -> ImportResult:
    """Parse a broker export: a holdings overview, or a transaction export (rebuilt into holdings).

    ``isin_map``: ISIN -> asset key; ``kinds``: asset key -> asset kind."""
    df, sep = read_table(raw)
    if is_transaction_export(df):
        return parse_transactions(df, isin_map, kinds)
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
