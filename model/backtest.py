"""Check the model against a past season: project each week using only data from before it.

    .venv/bin/python -m model.backtest --season 2025 --weeks 5-17

Compares PPR projections for QB/RB/WR/TE with a simple baseline (each player's average over
his last 4 games) on players who played. Kickers and defenses aren't scored here yet.
"""
import argparse

import numpy as np
import pandas as pd

from . import data
from .projections import Model, SKILL
from .site import WK_POOL


def baseline(d: data.Data, week: int) -> pd.Series:
    h = d.stats[d.stats.position.isin(SKILL) & ((d.stats.season < d.season) | (d.stats.week < week))]
    h = h.sort_values(["season", "week"])
    return h.groupby("player_id").fantasy_points_ppr.apply(lambda s: s.tail(4).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--weeks", default="5-17")
    args = ap.parse_args()
    lo, hi = map(int, args.weeks.split("-"))

    d = data.load(args.season)
    rows = []
    for week in range(lo, hi + 1):
        m = Model(d, week)
        p = m.project([week])
        p = p[p.position.isin(SKILL)][["player_id", "position", "pts_ppr"]].rename(columns={"pts_ppr": "model"})
        p["base"] = p.player_id.map(baseline(d, week))
        actual = d.stats[(d.stats.season == d.season) & (d.stats.week == week)].set_index("player_id").fantasy_points_ppr
        p["actual"] = p.player_id.map(actual)
        p = p.dropna(subset=["actual", "base"])
        # Fantasy-relevant: either method ranks him inside the weekly board's pool
        keep = pd.Series(False, index=p.index)
        for col in ["model", "base"]:
            rank = p.groupby("position")[col].rank(ascending=False)
            keep |= rank <= p.position.map(WK_POOL)
        p = p[keep].assign(week=week)
        rows.append(p)
        print(f"week {week}: {len(p)} players", flush=True)

    r = pd.concat(rows)
    out = []
    for pos, g in r.groupby("position"):
        # Spearman rank correlation within each week, averaged over weeks
        corr = lambda col: g.groupby("week").apply(lambda x: x[col].rank().corr(x.actual.rank())).mean()
        out.append(dict(pos=pos, players=len(g),
                        mae_model=np.abs(g.model - g.actual).mean(), mae_base=np.abs(g.base - g.actual).mean(),
                        rank_corr_model=corr("model"), rank_corr_base=corr("base")))
    print(f"\n{args.season} weeks {lo}-{hi}, PPR. Lower error and higher rank correlation are better.\n")
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
