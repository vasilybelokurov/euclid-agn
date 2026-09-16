"""Minimal, dependency-light IVOA TAP client.

astroquery is available in the environment, but its Euclid/IRSA interfaces move
between releases.  A direct TAP/sync POST is stable, easy to log for provenance
and easy to mock in tests.
"""

from __future__ import annotations

import io
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

import pandas as pd


class TapError(RuntimeError):
    """A TAP service returned an error document or an unusable payload."""


@dataclass
class TapService:
    """Synchronous TAP endpoint.

    Parameters
    ----------
    sync_url : str
        Full URL of the ``/sync`` endpoint.
    timeout : float
        Socket timeout in seconds.  Cone searches over the 400-million-row MER
        catalogue can take minutes; box searches on ``ra``/``dec`` are much
        faster and are what the manifest builder uses.
    """

    sync_url: str
    timeout: float = 600.0
    retries: int = 2
    last_query: str | None = None

    def query(self, adql: str) -> pd.DataFrame:
        """Run ADQL and return the result as a DataFrame (CSV transport)."""
        self.last_query = adql
        payload = urllib.parse.urlencode(
            {
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "csv",
                "QUERY": adql,
            }
        ).encode()
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.sync_url, data=payload)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                break
            except Exception as exc:  # network flakiness, not logic errors
                last_exc = exc
                if attempt == self.retries:
                    raise TapError(f"TAP request failed: {exc}\nQuery was:\n{adql}") from exc
                time.sleep(2.0 * (attempt + 1))
        if text.lstrip().startswith("<"):
            raise TapError(f"TAP returned an error document:\n{text[:2000]}\nQuery was:\n{adql}")
        if not text.strip():
            raise TapError(f"TAP returned an empty payload for query:\n{adql}")
        try:
            return pd.read_csv(io.StringIO(text))
        except Exception as exc:
            raise TapError(f"could not parse TAP CSV payload: {exc}\n{text[:1000]}") from exc


def quote_columns(columns) -> str:
    """Render a SELECT list with every plain identifier double-quoted.

    IRSA's ADQL parser rejects some perfectly ordinary Euclid column names.
    VERIFIED failure: ``SELECT position_angle FROM euclid_q1_mer_catalogue``
    returns ``BAD_REQUEST: Invalid or unsupported ADQL query string`` because
    the lexer sees the reserved word ``POSITION``; ``SELECT "position_angle"``
    works.  Quoting the whole SELECT list is uniform and harmless.

    Identifiers are quoted only in the SELECT list.  Quoting them in the WHERE
    clause as well was measured to break the same query, so WHERE clauses keep
    bare names.
    """
    if isinstance(columns, str):
        columns = [c.strip() for c in columns.split(",")]
    out = []
    for column in columns:
        name = column.strip()
        if name.isidentifier() and not name.startswith('"'):
            out.append(f'"{name}"')
        else:
            out.append(name)
    return ", ".join(out)


def in_list_clause(column: str, values) -> str:
    """ADQL ``IN`` clause for integer identifiers."""
    joined = ", ".join(str(int(v)) for v in values)
    return f"{column} IN ({joined})"
