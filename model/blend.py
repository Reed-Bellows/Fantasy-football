"""Blends your own positional rankings into the model's projections.

Each rankings file lists players by position, one per line: position,rank,player
Your rank becomes the points the model projects for that slot at that position
(your RB5 is worth the model's RB5 projection), and then

    final points = weight × your points + (1 − weight) × model points

Players you don't list keep the model's projection.
"""
import csv
import difflib
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ROS_FILE = ROOT / "my_rankings_ros.csv"
WEEK_FILE = ROOT / "my_rankings_week.csv"
WEIGHT = 0.7  # your share of the final points; 0 means model only
POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
WEEK_LINE = re.compile(r"#\s*week\s*:?\s*(\d+)", re.I)


def _key(name: str) -> str:
    """Lowercase, no accents, punctuation or suffixes: "Amon-Ra St. Brown" -> "amon ra st brown"."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’]", "", s)
    words = re.sub(r"[^a-z0-9]+", " ", s).split()
    return " ".join(w for w in words if w not in SUFFIXES)


def _aliases(r) -> set:
    keys = {_key(r.name)}
    if r.position == "DST":  # "Ravens D/ST", "Ravens", "BAL" all work
        keys |= {_key(r.name.replace("D/ST", "")), _key(r.team)}
    return keys


def load(path: Path) -> tuple[pd.DataFrame, int | None]:
    """Your rankings and, if the file has a "# week: N" line, the week they're for."""
    rows, week = [], None
    if not path.exists():
        return pd.DataFrame(columns=["position", "rank", "player", "line"]), None
    with path.open(newline="") as fh:
        for n, rec in enumerate(csv.reader(fh), start=1):
            if not rec or not "".join(rec).strip():
                continue
            if rec[0].lstrip().startswith("#"):
                if m := WEEK_LINE.match(",".join(rec).strip()):
                    week = int(m.group(1))
                continue
            if rec[0].strip().lower() == "position":  # header
                continue
            if len(rec) < 3:
                print(f"  {path.name} line {n}: expected position,rank,player — skipped")
                continue
            pos, rank, player = (x.strip() for x in rec[:3])
            pos = pos.upper().replace("D/ST", "DST").replace("DEF", "DST")
            if pos not in POSITIONS or not rank.isdigit() or int(rank) < 1:
                print(f"  {path.name} line {n}: can't read '{','.join(rec)}' — skipped")
                continue
            rows.append(dict(position=pos, rank=int(rank), player=player, line=n))
    return pd.DataFrame(rows, columns=["position", "rank", "player", "line"]), week


def _suggest(df, pos, player) -> str:
    names = df[df.position == pos].name.tolist()
    close = difflib.get_close_matches(player, names, n=1, cutoff=0.6)
    return f"no match, did you mean {close[0]}?" if close else "no match"


def apply(df: pd.DataFrame, mine: pd.DataFrame, cols: list[str], weight: float, label: str) -> pd.DataFrame:
    """Blend `mine` into the point columns `cols` of `df` (one row per player)."""
    if mine.empty or weight <= 0:
        return df
    df = df.copy()
    lookup = {}
    for idx, r in zip(df.index, df.itertuples()):
        for k in _aliases(r):
            lookup.setdefault((r.position, k), []).append(idx)

    matched, seen, missed = [], set(), []
    for r in mine.itertuples():
        hits = lookup.get((r.position, _key(r.player)), [])
        if len(hits) != 1:
            why = _suggest(df, r.position, r.player) if not hits else "matches " + ", ".join(f"{df.at[i, 'name']} ({df.at[i, 'team']})" for i in hits)
            missed.append(f"line {r.line}: {r.position} {r.player} — {why}")
        elif hits[0] in seen:
            missed.append(f"line {r.line}: {r.position} {r.player} — listed twice, kept the first")
        else:
            seen.add(hits[0])
            matched.append((hits[0], r.position, r.rank))

    for col in cols:
        slots = {pos: g[col].sort_values(ascending=False).to_numpy() for pos, g in df.groupby("position")}
        for idx, pos, rank in matched:
            yours = slots[pos][min(rank, len(slots[pos])) - 1]
            df.at[idx, col] = weight * yours + (1 - weight) * df.at[idx, col]

    print(f"{label}: blended {len(matched)} of your rankings at {weight:.0%} your weight")
    for m in missed:
        print(f"  not used, {m}")
    return df
