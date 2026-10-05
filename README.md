# Fantasy Website

**Consensus Big Board** is a single-page fantasy football rankings site for the 2026 season. It combines three sources into one sortable board:

- **FantasyPros**: expert consensus rankings
- **ESPN**: season and weekly projections and analyst ranks
- **Sleeper**: projected points, plus player ages and injury tags

Each player's board rank is the average of the three source ranks, so players all three sources like rise to the top.

## Features

- **Two views**:
  - **Rest of Season**: overall ranks. Projected points are converted to value over replacement for a 12-team, one-QB league.
  - **Weekly**: positional and FLEX start/sit ranks, with actual points shown once a game kicks off.
- **Scoring formats**: PPR, Half PPR and Standard
- **Filters**: by position (QB, RB, WR, TE, FLEX, K, DST), by player or team search, and a toggle to hide players who are Out or on IR
- **Sorting**: click any column header to sort by it
- **Spread strip**: shows how much the three sources agree or disagree on each player

## Running locally

Everything lives in `index.html`: the HTML, CSS, JavaScript and data. You can open the file directly in a browser, or serve it locally:

```sh
python3 -m http.server 8000
```

Then visit http://localhost:8000.
