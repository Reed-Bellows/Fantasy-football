"""Rebuild the board's rankings from nflverse data.

    .venv/bin/python -m model.build              # next unplayed week of the current season
    .venv/bin/python -m model.build --week 6     # a specific week
    .venv/bin/python -m model.build --weight 0   # ignore your rankings files
    .venv/bin/python -m model.build --rankings-only   # re-blend your rankings into the last full build's
                                                      # projections: no downloads, takes seconds
"""
import argparse
from datetime import date
from pathlib import Path

from . import blend, data, site
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
    ap.add_argument("--weight", type=float, default=blend.WEIGHT,
                    help=f"your rankings' share of the final points, 0 to 1 (default: {blend.WEIGHT})")
    ap.add_argument("--rankings-only", action="store_true",
                    help="reuse the last full build's projections instead of downloading new data")
    args = ap.parse_args()
    if not 0 <= args.weight <= 1:
        ap.error("--weight must be between 0 and 1")

    if args.rankings_only:
        if args.season != current_season(date.today()) or args.week:
            ap.error("--rankings-only reuses the last full build's season and week; drop --season and --week")
        proj = site.load()
        week = proj["info"]["week"]
        print(f"Using projections from the full build on {proj['info']['built']} (Week {week})")
    else:
        d = data.load(args.season)
        week = args.week or next_week(d)
        proj = site.project(Model(d, week))
        site.save(proj)
    payload = site.rank(proj, args.weight)
    site.write(payload, args.html)
    print(f"Wrote {len(payload['ROS'])} rest-of-season and {len(payload['WK'])} Week {week} players to {args.html}")


if __name__ == "__main__":
    main()
