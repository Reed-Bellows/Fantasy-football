"""Rebuild the board's rankings from nflverse data.

    .venv/bin/python -m model.build              # next unplayed week of the current season
    .venv/bin/python -m model.build --week 6     # a specific week
"""
import argparse
from datetime import date
from pathlib import Path

from . import data, site
from .projections import Model

HTML = Path(__file__).resolve().parent.parent / "index.html"


def current_season(today: date) -> int:
    return today.year if today.month >= 8 else today.year - 1


def next_week(d: data.Data) -> int:
    unplayed = d.sched[d.sched.result.isna()]
    if unplayed.empty:
        raise SystemExit(f"The {d.season} regular season is over.")
    return int(unplayed.week.min())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=current_season(date.today()))
    ap.add_argument("--week", type=int, help="week to project (default: the next week with unplayed games)")
    ap.add_argument("--html", type=Path, default=HTML)
    args = ap.parse_args()

    d = data.load(args.season)
    week = args.week or next_week(d)
    payload = site.build(Model(d, week))
    site.write(payload, args.html)
    print(f"Wrote {len(payload['ROS'])} rest-of-season and {len(payload['WK'])} Week {week} players to {args.html}")


if __name__ == "__main__":
    main()
