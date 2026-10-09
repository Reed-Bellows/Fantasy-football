"""Projects fantasy points per player per game: opportunity x efficiency, adjusted for game context.

Every rate is a recency-weighted average of past games, pulled toward the league average
by a pseudo-count, so a player with a few big games doesn't get an extreme projection.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import scoring
from .data import Data

DECAY = 0.85            # each older team-week counts 15% less
PRIOR_SEASON = 0.6      # last season's games count a further 40% less
SKILL = ["QB", "RB", "WR", "TE"]
FUMBLE_RATE = 0.004     # fumbles lost per touch, league-wide
QB_PASS_SHARE = 0.97    # the starter's share of team pass attempts
HOME_EDGE = 0.8         # points added to the home side (and taken from the away side) when there's no betting line
# Chance a player suits up in a future week, beyond known injuries
AVAILABILITY = {"QB": 0.95, "RB": 0.90, "WR": 0.93, "TE": 0.93, "K": 0.99, "DST": 1.0}
# Typical game-to-game swing (sd / mean); each player's own history is blended toward these
CV = {"QB": 0.40, "RB": 0.55, "WR": 0.60, "TE": 0.65, "K": 0.50, "DST": 0.75}
# Share of team targets and carries assumed for a player with no history
PRIOR_SHARE = {"QB": (0.0, 0.03), "RB": (0.015, 0.03), "WR": (0.02, 0.002), "TE": (0.015, 0.001)}
INJURY_AVAIL = {"Out": 0.0, "Doubtful": 0.2, "Questionable": 0.8}
IR_WEEKS = 4            # minimum stint on injured reserve
TE_SCALE = 0.95         # tight ends' projected stat lines (and so points) are scaled by this; 1.0 means no change
TEAM_COLS = ["attempts", "carries", "targets", "pf", "pa", "def_sacks", "def_interceptions", "fumble_recovery_opp",
             "def_tds", "special_teams_tds", "def_safeties", "sacks_suffered", "passing_interceptions",
             "fumbles_lost_total", "fg_att", "pat_att"]
RARE_TEAM_COLS = {"def_tds", "special_teams_tds", "def_safeties"}
ET = ZoneInfo("America/New_York")


def _shrink(num, den, prior, k):
    """Rate num/den pulled toward `prior`, as if `k` units of league-average data were added."""
    return (num + prior * k) / (den + k)


def _fill_factor(target, total):
    """Scale shares that overshoot the team total all the way down; fill a shortfall only partway."""
    ratio = target / total.where(total > 0, 1)
    return pd.Series(np.where(ratio < 1, ratio, np.minimum(ratio ** 0.7, 1.35)), index=total.index)


class Model:
    def __init__(self, d: Data, week: int):
        self.d, self.season, self.week = d, d.season, week

        def past(df):
            df = df[(df.season < self.season) | (df.week < week)]
            return df.assign(w=self._weight(df))

        self.hist = past(d.stats)
        self.team_hist = past(d.team)
        self._league()
        self._team_profiles()
        self.games = self._games()
        self.roster = self._roster()
        self.skill = self._skill_profiles()
        self.kickers = self._kicker_profiles()
        self._matchups()
        self._starters()

    def _weight(self, df):
        gap = self.week - 1 - ((df.season - self.season) * 19 + df.week)
        return DECAY ** gap * np.where(df.season < self.season, PRIOR_SEASON, 1.0)

    # ---- league and team baselines ----

    def _league(self):
        t, h = self.team_hist, self.hist
        self.lg = {c: t[c].mean() for c in TEAM_COLS}
        self.lg["total"] = 2 * self.lg["pf"]
        self.pos_rates = {}
        for pos in SKILL:
            p = h[h.position == pos]
            tg, car, att = max(p.targets.sum(), 1), max(p.carries.sum(), 1), max(p.attempts.sum(), 1)
            self.pos_rates[pos] = dict(
                catch=p.receptions.sum() / tg, ypt=p.receiving_yards.sum() / tg,
                rec_td=(p.receiving_tds.sum() + p.rec_touchdown_exp.sum()) / 2 / tg,
                ypc=p.rushing_yards.sum() / car,
                rush_td=(p.rushing_tds.sum() + p.rush_touchdown_exp.sum()) / 2 / car,
                cmp=p.completions.sum() / att, ypa=p.passing_yards.sum() / att,
                pass_td=(p.passing_tds.sum() + p.pass_touchdown_exp.sum()) / 2 / att,
                int=p.passing_interceptions.sum() / att)

    def _team_profiles(self):
        t = self.team_hist
        n = t.groupby("team").w.sum()
        prof = {c: _shrink((t[c] * t.w).groupby(t.team).sum(), n, self.lg[c], 8 if c in RARE_TEAM_COLS else 2)
                for c in TEAM_COLS}
        self.tp = pd.DataFrame(prof)
        self.tp["dst_tds"] = self.tp.def_tds + self.tp.special_teams_tds
        self.lg["dst_tds"] = self.lg["def_tds"] + self.lg["special_teams_tds"]

    def _games(self) -> pd.DataFrame:
        """One row per team per remaining game: opponent, implied points, kickoff, listed starting QB."""
        tp, rows = self.tp, []
        for g in self.d.sched[self.d.sched.week >= self.week].itertuples():
            if pd.notna(g.spread_line) and pd.notna(g.total_line):
                h_imp = g.total_line / 2 + g.spread_line / 2   # positive spread = home team favored
                a_imp = g.total_line - h_imp
            else:  # no line yet: average each offense with the defense it faces
                h_imp = (tp.pf[g.home_team] + tp.pa[g.away_team]) / 2 + HOME_EDGE
                a_imp = (tp.pf[g.away_team] + tp.pa[g.home_team]) / 2 - HOME_EDGE
            ko = int(datetime.strptime(f"{g.gameday} {g.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET).timestamp())
            for team, opp, imp, opp_imp, home, qb in [
                    (g.home_team, g.away_team, h_imp, a_imp, True, g.home_qb_id),
                    (g.away_team, g.home_team, a_imp, h_imp, False, g.away_qb_id)]:
                rows.append(dict(week=g.week, team=team, opp=opp, implied=imp, opp_implied=opp_imp, home=home,
                                 ko=ko, qb=qb if pd.notna(qb) else None, line=pd.notna(g.spread_line)))
        return pd.DataFrame(rows)

    # ---- players ----

    def _roster(self) -> pd.DataFrame:
        # Each team's latest roster up to this week: teams on bye have no roster for this week
        r = self.d.rosters[self.d.rosters.week <= self.week]
        r = r[r.week == r.groupby("team").week.transform("max")]
        # Game-day inactives count as active (the injury report handles this week); the exempt list counts as reserve
        r = r.assign(status=r.status.replace({"INA": "ACT", "EXE": "RES"}))
        r = r[r.position.isin(SKILL + ["K"]) & r.status.isin(["ACT", "RES"])]
        r = r.drop_duplicates("gsis_id").rename(columns={"gsis_id": "player_id", "full_name": "name"})
        r = r[["player_id", "name", "position", "team", "status", "birth_date"]].copy()

        # This week's official injury report; if a team hasn't filed one yet, carry last week's "Out"
        # forward at half weight for players who sat out last week's game
        inj = self.d.injuries.dropna(subset=["report_status"])
        now = inj[inj.week == self.week].set_index("gsis_id").report_status
        prev = inj[(inj.week == self.week - 1) & (inj.report_status == "Out")].set_index("gsis_id").report_status
        filed = set(inj[inj.week == self.week].team)
        played_last = set(self.hist[(self.hist.season == self.season) & (self.hist.week == self.week - 1)].player_id)

        def status(row):
            if row.status == "RES":
                return "IR", None, 0.0
            if row.player_id in now.index:
                s = now[row.player_id]
                return s, None, INJURY_AVAIL.get(s, 1.0)
            if row.team not in filed and row.player_id in prev.index and row.player_id not in played_last:
                return "Out", self.week - 1, 0.5
            return "", None, 1.0

        r[["inj", "inj_week", "avail_now"]] = pd.DataFrame(r.apply(status, axis=1).tolist(), index=r.index)
        return r

    def _skill_profiles(self) -> pd.DataFrame:
        tm = (self.team_hist[["game_id", "team", "attempts", "carries", "targets"]]
              .rename(columns={"attempts": "t_att", "carries": "t_car", "targets": "t_tgt"}))
        h = self.hist.merge(tm, on=["game_id", "team"], how="left")
        h["rec_td_obs"] = (h.receiving_tds + h.rec_touchdown_exp) / 2
        h["rush_td_obs"] = (h.rushing_tds + h.rush_touchdown_exp) / 2
        h["pass_td_obs"] = (h.passing_tds + h.pass_touchdown_exp) / 2
        h["pts2"] = h.fantasy_points_ppr ** 2
        cols = ["targets", "receptions", "receiving_yards", "rec_td_obs", "carries", "rushing_yards", "rush_td_obs",
                "attempts", "completions", "passing_yards", "pass_td_obs", "passing_interceptions",
                "t_att", "t_car", "t_tgt", "fantasy_points_ppr", "pts2"]
        sums = h[cols].fillna(0).mul(h.w, axis=0).groupby(h.player_id).sum()
        sums["n"] = h.groupby("player_id").w.sum()

        p = self.roster[self.roster.position.isin(SKILL)].merge(sums, left_on="player_id", right_index=True, how="left")
        p[cols + ["n"]] = p[cols + ["n"]].fillna(0)
        prior = lambda key: p.position.map(lambda pos: self.pos_rates[pos][key])
        qb = self.pos_rates["QB"]
        p["tgt_share"] = _shrink(p.targets, p.t_tgt, p.position.map(lambda x: PRIOR_SHARE[x][0]), self.lg["targets"])
        p["car_share"] = _shrink(p.carries, p.t_car, p.position.map(lambda x: PRIOR_SHARE[x][1]), self.lg["carries"])
        p["catch"] = _shrink(p.receptions, p.targets, prior("catch"), 40)
        p["ypt"] = _shrink(p.receiving_yards, p.targets, prior("ypt"), 60)
        p["rec_td"] = _shrink(p.rec_td_obs, p.targets, prior("rec_td"), 100)
        p["ypc"] = _shrink(p.rushing_yards, p.carries, prior("ypc"), 80)
        p["rush_td"] = _shrink(p.rush_td_obs, p.carries, prior("rush_td"), 100)
        p["cmp"] = _shrink(p.completions, p.attempts, qb["cmp"], 150)
        p["ypa"] = _shrink(p.passing_yards, p.attempts, qb["ypa"], 200)
        p["pass_td"] = _shrink(p.pass_td_obs, p.attempts, qb["pass_td"], 300)
        p["int"] = _shrink(p.passing_interceptions, p.attempts, qb["int"], 400)

        # Volatility: this player's game-to-game swing, blended toward his position's typical swing
        n = p.n.where(p.n > 0, 1)
        mean = p.fantasy_points_ppr / n
        sd = np.sqrt(np.maximum(p.pts2 / n - mean ** 2, 0))
        cv_obs = np.where(mean > 1, sd / mean.where(mean > 1, 1), p.position.map(CV))
        p["cv"] = (p.n * cv_obs + 6 * p.position.map(CV)) / (p.n + 6)
        return p

    def _kicker_profiles(self) -> pd.DataFrame:
        made_cols = ["fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49", "fg_made_50_59", "fg_made_60_"]
        cols = ["fg_made", "fg_att", "pat_made", "pat_att"]
        h = self.hist[self.hist.position == "K"].fillna({c: 0 for c in made_cols + cols})
        short = h.fg_made_0_19.sum() + h.fg_made_20_29.sum() + h.fg_made_30_39.sum()
        mid, long_ = h.fg_made_40_49.sum(), h.fg_made_50_59.sum() + h.fg_made_60_.sum()
        self.pts_per_fg = (3 * short + 4 * mid + 5 * long_) / max(short + mid + long_, 1)
        lg_fg, lg_pat = h.fg_made.sum() / max(h.fg_att.sum(), 1), h.pat_made.sum() / max(h.pat_att.sum(), 1)

        sums = h[cols].mul(h.w, axis=0).groupby(h.player_id).sum()
        k = self.roster[self.roster.position == "K"].merge(sums, left_on="player_id", right_index=True, how="left")
        k[cols] = k[cols].fillna(0)
        k["fg_acc"] = _shrink(k.fg_made, k.fg_att, lg_fg, 20)
        k["pat_acc"] = _shrink(k.pat_made, k.pat_att, lg_pat, 30)
        # One kicker per team: the active one with the most recent attempts
        return k[k.status == "ACT"].sort_values("fg_att", ascending=False).drop_duplicates("team")

    def _matchups(self):
        """How many PPR points each defense allows to each position, relative to league average."""
        h = self.hist[self.hist.position.isin(SKILL)]
        g = (h.groupby(["game_id", "opponent_team", "position"])
             .agg(pts=("fantasy_points_ppr", "sum"), w=("w", "first")).reset_index())
        lg = g.groupby("position").pts.mean()
        g["wp"] = g.pts * g.w
        a = g.groupby(["opponent_team", "position"]).agg(wp=("wp", "sum"), w=("w", "sum"))
        base = a.index.get_level_values("position").map(lg).to_numpy()
        ratio = _shrink(a.wp, a.w, base, 4) / base
        self.matchup = np.clip(ratio ** 0.5, 0.9, 1.1).to_dict()

    def _starters(self):
        """Each team's starting QB: the schedule's listed starter, else the active QB with the most recent attempts."""
        listed = self.games.dropna(subset=["qb"]).sort_values("week").groupby("team").qb.first()
        qbs = self.skill[(self.skill.position == "QB") & (self.skill.status == "ACT")]
        fallback = qbs.sort_values("attempts", ascending=False).drop_duplicates("team").set_index("team").player_id
        self.default_qb = listed.combine_first(fallback)

    # ---- projections ----

    def _availability(self, p, week):
        """(expected share of the game played, whether he counts when splitting team volume)."""
        if week == self.week:
            return p.avail_now.to_numpy(), p.avail_now.to_numpy()
        base = p.position.map(AVAILABILITY).to_numpy()
        on_ir = (p.status == "RES").to_numpy()
        back = week >= self.week + IR_WEEKS
        return np.where(on_ir, 0.6 * base if back else 0.0, base), np.where(on_ir & ~back, 0.0, 1.0)

    def _context(self, p):
        tp = self.tp.reindex(p.team)
        ratio = p.implied.to_numpy() / tp.pf.to_numpy()
        margin = (p.implied - p.opp_implied).to_numpy()
        pace = ((p.implied + p.opp_implied).to_numpy() / self.lg["total"]) ** 0.35
        pass_f = np.clip(1 - 0.009 * margin, 0.85, 1.15) * pace   # favorites run more, underdogs throw more
        rush_f = np.clip(1 + 0.014 * margin, 0.80, 1.25) * pace
        return dict(tp=tp, ratio=ratio, team_att=tp.attempts.to_numpy() * pass_f,
                    team_tgt=tp.targets.to_numpy() * pass_f, team_car=tp.carries.to_numpy() * rush_f,
                    td_env=np.clip(ratio ** 0.8, 0.65, 1.5), yd_env=np.clip(ratio ** 0.25, 0.85, 1.15))

    def _skill_week(self, week, g) -> pd.DataFrame:
        p = self.skill.merge(g, on="team")
        ev, counts = self._availability(p, week)
        p["avail"] = ev
        is_qb = (p.position == "QB").to_numpy()
        listed = g.set_index("team").qb.dropna()
        starter = p.team.map(lambda t: listed.get(t, self.default_qb.get(t)))
        p["start"] = is_qb & (p.player_id == starter).to_numpy()
        pass_share = np.where(p.start, QB_PASS_SHARE, 0.0)
        t_share = np.where(is_qb, 0.0, p.tgt_share)
        c_share = np.where(is_qb & ~p.start, 0.0, p.car_share)

        # Split team volume among the players who are actually available: when someone sits,
        # teammates' shares scale up (partly), and overlapping shares from role changes scale down
        nq = ~is_qb
        team = p.team.to_numpy()
        tsum = pd.Series(t_share * counts * nq).groupby(team).sum()
        csum = pd.Series(c_share * counts * nq).groupby(team).sum()
        qb_c = pd.Series(np.where(p.start, c_share, 0.0)).groupby(team).sum()
        t_fac, c_fac = _fill_factor(1.0, tsum), _fill_factor(1.0 - qb_c, csum)
        t_share = np.where(nq, t_share * pd.Series(team).map(t_fac).to_numpy(), t_share)
        c_share = np.where(nq, c_share * pd.Series(team).map(c_fac).to_numpy(), c_share)

        cx = self._context(p)
        mu = np.array([self.matchup.get((o, pos), 1.0) for o, pos in zip(p.opp, p.position)])
        tgt, car, att = t_share * cx["team_tgt"], c_share * cx["team_car"], pass_share * cx["team_att"]
        rec = tgt * p.catch.to_numpy()
        line = dict(
            tgt=tgt, rec=rec, rec_yd=tgt * p.ypt.to_numpy() * mu * cx["yd_env"],
            rec_td=tgt * p.rec_td.to_numpy() * cx["td_env"] * mu,
            car=car, rush_yd=car * p.ypc.to_numpy() * mu * cx["yd_env"],
            rush_td=car * p.rush_td.to_numpy() * cx["td_env"] * mu,
            att=att, pass_yd=att * p.ypa.to_numpy() * mu * cx["yd_env"],
            pass_td=att * p.pass_td.to_numpy() * cx["td_env"] * mu, int=att * p.int.to_numpy(),
            fum=(rec + car) * FUMBLE_RATE)
        te = np.where(p.position == "TE", TE_SCALE, 1.0)
        line = {k: v * te for k, v in line.items()}
        for k, v in line.items():
            p[k] = v
        for f, ppr in scoring.FORMATS.items():
            p[f"pts_{f}"] = scoring.offense(line, ppr)
        return p

    def _kicker_week(self, week, g) -> pd.DataFrame:
        k = self.kickers.merge(g, on="team")
        ev, _ = self._availability(k, week)
        cx = self._context(k)
        fga = cx["tp"].fg_att.to_numpy() * cx["ratio"] ** 0.5
        xpa = cx["tp"].pat_att.to_numpy() * cx["ratio"]
        acc, pacc = k.fg_acc.to_numpy(), k.pat_acc.to_numpy()
        pts = scoring.kicker(fga * acc, fga * (1 - acc), self.pts_per_fg, xpa * pacc, xpa * (1 - pacc))
        k = k.assign(avail=ev, fga=fga, xpa=xpa, cv=CV["K"], **{f"pts_{f}": pts for f in scoring.FORMATS})
        return k

    def _dst_week(self, g) -> pd.DataFrame:
        tp, lg = self.tp, self.lg
        nick = self.d.teams.set_index("team_abbr").team_nick
        rows = []
        for r in g.itertuples():
            t, o = tp.loc[r.team], tp.loc[r.opp]
            sacks = t.def_sacks * np.sqrt(o.sacks_suffered / lg["sacks_suffered"])
            ints = t.def_interceptions * np.sqrt(o.passing_interceptions / lg["passing_interceptions"])
            fum = t.fumble_recovery_opp * np.sqrt(o.fumbles_lost_total / lg["fumbles_lost_total"])
            pts = scoring.defense(sacks, ints, fum, t.dst_tds, t.def_safeties, r.opp_implied)
            rows.append(dict(player_id=f"DST-{r.team}", name=f"{nick.get(r.team, r.team)} D/ST", position="DST",
                             team=r.team, status="ACT", inj="", inj_week=None, birth_date=None, avail=1.0,
                             sacks=sacks, to=ints + fum, cv=CV["DST"], **{f"pts_{f}": pts for f in scoring.FORMATS}))
        return g.merge(pd.DataFrame(rows), on="team") if rows else pd.DataFrame()

    def project(self, weeks) -> pd.DataFrame:
        """One row per player per game in `weeks`; points assume he plays, `avail` is the chance he does."""
        out = []
        for week in weeks:
            g = self.games[self.games.week == week]
            if g.empty:
                continue
            out += [self._skill_week(week, g), self._kicker_week(week, g), self._dst_week(g)]
        return pd.concat(out, ignore_index=True)
