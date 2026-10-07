"""Loads the nflverse tables the model uses (CC-BY 4.0, credit nflverse on the site)."""
from dataclasses import dataclass

import nflreadpy as nfl
import pandas as pd

FANTASY_POS = ["QB", "RB", "WR", "TE", "K"]


@dataclass
class Data:
    season: int
    stats: pd.DataFrame      # one row per player-game, regular season, with expected TDs joined on
    team: pd.DataFrame       # one row per team-game, with points scored and allowed
    sched: pd.DataFrame      # regular-season schedule for `season`, with betting lines
    rosters: pd.DataFrame    # weekly rosters for `season`
    injuries: pd.DataFrame   # official injury reports for `season`
    teams: pd.DataFrame      # team names


def _pd(df) -> pd.DataFrame:
    return df.to_pandas()


def load(season: int) -> Data:
    """Load `season` plus the season before it, which the model uses as a prior."""
    seasons = [season - 1, season]

    stats = _pd(nfl.load_player_stats(seasons))
    stats = stats[stats.season_type == "REG"].copy()

    # Expected TDs from the ffverse opportunity model steady the TD rates
    opp = _pd(nfl.load_ff_opportunity(seasons))
    opp = opp[["season", "week", "player_id", "rec_touchdown_exp", "rush_touchdown_exp", "pass_touchdown_exp"]]
    opp = opp.assign(season=opp.season.astype(int), week=opp.week.astype(int)).groupby(["season", "week", "player_id"], as_index=False).sum()
    stats = stats.merge(opp, on=["season", "week", "player_id"], how="left")
    # Games the opportunity model skipped fall back to actual TDs
    for exp, actual in [("rec_touchdown_exp", "receiving_tds"), ("rush_touchdown_exp", "rushing_tds"),
                        ("pass_touchdown_exp", "passing_tds")]:
        stats[exp] = stats[exp].fillna(stats[actual])

    sched_all = _pd(nfl.load_schedules(seasons))
    sched_all = sched_all[sched_all.game_type == "REG"].copy()

    team = _pd(nfl.load_team_stats(seasons))
    team = team[team.season_type == "REG"].copy()
    # Points for and against come from the schedule
    home = sched_all[["game_id", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "pf", "away_score": "pa"})
    away = sched_all[["game_id", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "pf", "home_score": "pa"})
    team = team.merge(pd.concat([home, away]), on=["game_id", "team"], how="left")

    return Data(
        season=season,
        stats=stats,
        team=team,
        sched=sched_all[sched_all.season == season].copy(),
        rosters=_pd(nfl.load_rosters_weekly([season])),
        injuries=_pd(nfl.load_injuries([season])),
        teams=_pd(nfl.load_teams()),
    )
