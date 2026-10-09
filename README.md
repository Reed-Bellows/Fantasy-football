# Fantasy Website

**Big Board** is a fantasy football rankings site for the 2026 season. The rankings come from this project's own projection model, built on open data from [nflverse](https://github.com/nflverse) (CC BY 4.0).

## Features

- **Two views**:
  - **Rest of Season**: projected points for the remaining weeks, discounted for injury risk. Positional and FLEX ranks sort by projected points; a value column shows points over replacement for a 12-team, one-QB league.
  - **Weekly**: positional and FLEX start/sit ranks, with a likely range for each player and his projected stat line.
- **Scoring formats**: PPR, Half PPR and Standard
- **Filters**: by position (QB, RB, WR, TE, FLEX, K, DST), by player or team search, and a toggle to hide players who are Out or on IR
- **Sorting**: click any column header to sort by it
- **Trade Analyzer** (`trade.html`): pick any players from the rest-of-season board for each side of a trade. Each side's value is the players' value over replacement, plus a 10% depth credit for points up to replacement level so bench players keep their order, with the best player counted in full and each extra player discounted (85%, 70%, 55%, 40%, then 25%). Sides within 10% of each other are a fair trade; otherwise the side with more value wins.

## How the model works

Each player's projection is **opportunity × efficiency**, in his team's context:

- **Opportunity**: his share of team targets, carries or pass attempts. Recent games count most, and last season's games count less.
- **Efficiency**: catch rate, yards per target or carry, TD rate, and for quarterbacks completion rate, yards per attempt, TD rate and interceptions. Every rate is pulled toward the average for his position, so small samples don't swing it. Touchdown rates also blend in expected TDs from the ffopportunity model.
- **Team context**: betting lines set each team's expected points and game script. Weeks without a line use team scoring and points-allowed averages instead. Each defense's points allowed to a position nudge yards and TDs by up to 10%.
- **Availability**: official injury reports and injured-reserve status. When a player is out, his teammates take on more of the team's volume.

The model code lives in [`model/`](model/):

| File | What it does |
|---|---|
| `data.py` | Loads nflverse tables |
| `projections.py` | The projection model, with its tuning constants at the top |
| `scoring.py` | Fantasy scoring rules |
| `site.py` | Builds the board's data and writes it to `data.js` |
| `build.py` | Command line entry point |
| `backtest.py` | Tests the model on a past season |

## Updating the rankings

One-time setup:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Each week, after the games finish (and again late in the week, once injury reports are out):

```sh
.venv/bin/python -m model.build
```

This projects the next week that has unplayed games and rewrites `data.js`, which both pages load. Use `--week 6` to pick a specific week.

The full build also saves the model's projections to `.cache/projections.pkl`. After editing only your rankings files, you can re-blend them into those saved projections without downloading anything:

```sh
.venv/bin/python -m model.build --rankings-only
```

The projections stay exactly as the last full build left them, so the only changes on the board come from your rankings.

## Adding your own rankings

The board shows one set of rankings: the model's projections blended with your own. Your rankings live in two files at the top of the repo:

| File | What it holds |
|---|---|
| `my_rankings_ros.csv` | Rest-of-season rankings. Update whenever your outlook changes. |
| `my_rankings_week.csv` | Rankings for one week. Set the `# week: N` line to the week you're ranking; if it doesn't match the week being built, the weekly board uses the model only. |

Each line is `position,rank,player`, ranked within the position:

```
RB,1,Bijan Robinson
WR,3,Amon-Ra St. Brown
DST,2,Ravens
```

List only the players you have an opinion on; everyone else keeps the model's projection. Capitalization, punctuation and suffixes like "Jr." don't matter, and a D/ST can be its nickname or team abbreviation.

When you run `model.build`, each player you ranked gets the points the model projects for that slot (your RB5 is worth the model's RB5 projection), blended with his own projection:

```
final points = 80% your points + 20% model points
```

The blend runs separately for PPR, Half PPR and Standard, before value over replacement and the positional and FLEX ranks are computed. Change the weight with `--weight` (`--weight 0` ignores your files). The build prints any lines it couldn't match, with a suggestion when it looks like a typo.

Only the blended rankings reach the site, so only someone who can push to this repo can change them.

## Checking accuracy

```sh
.venv/bin/python -m model.backtest --season 2025 --weeks 5-17
```

This replays a past season, projecting each week from only the data available before it, and compares the results with a simple baseline (each player's average over his last 4 games). On 2025 Weeks 5–17 (PPR, QB/RB/WR/TE), the model beat the baseline at every position:

| Position | Avg. error, model | Avg. error, baseline | Rank correlation, model | Rank correlation, baseline |
|---|---|---|---|---|
| QB | 5.85 | 7.15 | 0.51 | 0.33 |
| RB | 4.87 | 5.40 | 0.66 | 0.60 |
| WR | 5.08 | 5.51 | 0.50 | 0.44 |
| TE | 4.78 | 5.41 | 0.43 | 0.29 |

## Viewing locally

The site is `index.html` (the Big Board), `trade.html` (the Trade Analyzer), `style.css` and `data.js`. You can open `index.html` directly, or serve the folder:

```sh
python3 -m http.server 8000
```

Then visit http://localhost:8000.
