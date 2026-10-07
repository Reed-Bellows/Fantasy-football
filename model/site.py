"""Turns projections into the board's ROS and WK data and writes them into index.html."""
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import scoring
from .projections import Model

# Replacement level for value over replacement in a 12-team, one-QB league
REPLACEMENT = {"QB": 12, "RB": 30, "WR": 36, "TE": 14, "K": 12, "DST": 12}
# How many players per position make the board
ROS_POOL = {"QB": 36, "RB": 70, "WR": 90, "TE": 36, "K": 32, "DST": 32}
WK_POOL = {"QB": 32, "RB": 64, "WR": 80, "TE": 32, "K": 30, "DST": 30}
FLEX = ["RB", "WR", "TE"]
Z80 = 0.84  # the range shown is the middle 60% of outcomes (20th to 80th percentile)


def _r1(x):
    return round(float(x), 1)


def _age(birth, today):
    if birth is None or pd.isna(birth):
        return None
    b = pd.Timestamp(birth).date()
    return today.year - b.year - ((today.month, today.day) < (b.month, b.day))


def _top(df, pool, key):
    return pd.concat([df[df.position == pos].nlargest(n, key) for pos, n in pool.items()])


def _stat_line(r) -> str:
    if r.position == "QB":
        return f"{r.att:.0f} att · {r.pass_yd:.0f} pass yd · {r.pass_td:.1f} TD · {r.rush_yd:.0f} rush yd"
    if r.position == "RB":
        return f"{r.car:.1f} car · {r.rush_yd:.0f} yd · {r.rec:.1f} rec · {r.rec_yd:.0f} yd · {r.rush_td + r.rec_td:.1f} TD"
    if r.position in ("WR", "TE"):
        return f"{r.tgt:.1f} tgt · {r.rec:.1f} rec · {r.rec_yd:.0f} yd · {r.rec_td + r.rush_td:.1f} TD"
    if r.position == "K":
        return f"{r.fga:.1f} FGA · {r.xpa:.1f} XP"
    return f"{r.sacks:.1f} sacks · {r.to:.1f} TO · opp. {r.opp_implied:.0f} pts"


def build(m: Model) -> dict:
    proj = m.project(range(m.week, 19))
    today = datetime.now().date()
    weeks_played = m.d.sched.melt(id_vars="week", value_vars=["home_team", "away_team"]).groupby("value").week.apply(set)
    bye = {t: next((w for w in range(1, 19) if w not in ws), None) for t, ws in weeks_played.items()}

    # ---- rest of season: expected points over remaining games, discounted by availability ----
    for f in scoring.FORMATS:
        proj[f"ev_{f}"] = proj[f"pts_{f}"] * proj.avail
    ros = (proj.groupby("player_id")
           .agg(name=("name", "first"), position=("position", "first"), team=("team", "first"),
                inj=("inj", "first"), inj_week=("inj_week", "first"), birth=("birth_date", "first"),
                gp=("avail", "sum"), **{f"ev_{f}": (f"ev_{f}", "sum") for f in scoring.FORMATS})
           .reset_index())
    ros = _top(ros, ROS_POOL, "ev_ppr")
    for f in scoring.FORMATS:
        ros[f"pr_{f}"] = ros.groupby("position")[f"ev_{f}"].rank(ascending=False, method="first").astype(int)
        repl = {pos: ros[(ros.position == pos) & (ros[f"pr_{f}"] == n)][f"ev_{f}"].max() for pos, n in REPLACEMENT.items()}
        ros[f"vor_{f}"] = ros[f"ev_{f}"] - ros.position.map(repl).fillna(0)
        ros[f"r_{f}"] = ros[f"vor_{f}"].rank(ascending=False, method="first").astype(int)
    ros = ros.sort_values("r_ppr")

    ros_out = [dict(
        n=r.name, p=r.position, t=r.team, b=bye.get(r.team), i=r.inj or "",
        iw=int(r.inj_week) if pd.notna(r.inj_week) else None, a=_age(r.birth, today),
        f={f: dict(pts=_r1(getattr(r, f"ev_{f}")), pg=_r1(getattr(r, f"ev_{f}") / r.gp) if r.gp > 0 else None,
                   gp=_r1(r.gp), v=_r1(getattr(r, f"vor_{f}")), r=int(getattr(r, f"r_{f}")),
                   pr=int(getattr(r, f"pr_{f}"))) for f in scoring.FORMATS})
        for r in ros.itertuples()]

    # ---- this week: points if he plays, with a typical range ----
    wk = proj[(proj.week == m.week) & (proj.avail > 0)].copy()
    wk = _top(wk, WK_POOL, "pts_ppr")
    for f in scoring.FORMATS:
        wk[f"pr_{f}"] = wk.groupby("position")[f"pts_{f}"].rank(ascending=False, method="first").astype(int)
        fx = wk[wk.position.isin(FLEX)][f"pts_{f}"].rank(ascending=False, method="first")
        wk[f"fx_{f}"] = fx.reindex(wk.index)
    wk = wk.sort_values("pts_ppr", ascending=False)

    def wk_fmt(r, f):
        pts = getattr(r, f"pts_{f}")
        sd = pts * r.cv
        lo = pts - Z80 * sd if r.position in ("K", "DST") else max(0.0, pts - Z80 * sd)
        fx = getattr(r, f"fx_{f}")
        return dict(pts=_r1(pts), lo=_r1(lo), hi=_r1(pts + Z80 * sd), pr=int(getattr(r, f"pr_{f}")),
                    fx=int(fx) if pd.notna(fx) else None)

    wk_out = [dict(
        n=r.name, p=r.position, t=r.team, o=("vs " if r.home else "@") + r.opp, ko=int(r.ko), i=r.inj or "",
        iw=int(r.inj_week) if pd.notna(r.inj_week) else None, s=_stat_line(r),
        f={f: wk_fmt(r, f) for f in scoring.FORMATS})
        for r in wk.itertuples()]

    line_weeks = sorted(int(w) for w in m.games[m.games.line].week.unique())
    info = dict(season=m.season, week=m.week, built=today.isoformat(),
                stats_through=m.week - 1, line_weeks=line_weeks)
    return dict(ROS=ros_out, WK=wk_out, INFO=info)


BEGIN, END = "// BEGIN MODEL DATA", "// END MODEL DATA"


def write(payload: dict, html_path: Path):
    html = html_path.read_text()
    dump = lambda x: json.dumps(x, separators=(",", ":"), ensure_ascii=False, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    block = "\n".join(f"const {k} = {dump(v)};" for k, v in payload.items())
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pattern.search(html):
        raise SystemExit(f"{html_path} has no '{BEGIN}' ... '{END}' block to replace")
    html_path.write_text(pattern.sub(lambda _: f"{BEGIN}\n{block}\n{END}", html))
