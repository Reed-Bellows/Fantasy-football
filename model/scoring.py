"""Fantasy scoring rules. Offense differs only in points per catch; K and DST score the same in every format."""
from math import erf, sqrt

FORMATS = {"ppr": 1.0, "half": 0.5, "std": 0.0}


def offense(line: dict, ppr: float) -> float:
    """Points for a projected stat line (expected values, so fractions are fine)."""
    g = line.get
    return (g("pass_yd", 0) * 0.04 + g("pass_td", 0) * 4 - g("int", 0) * 2
            + (g("rush_yd", 0) + g("rec_yd", 0)) * 0.1
            + (g("rush_td", 0) + g("rec_td", 0)) * 6
            + g("rec", 0) * ppr - g("fum", 0) * 2)


# Kicker: 3 points for a field goal under 40 yards, 4 for 40-49, 5 for 50+; misses cost 1
def kicker(fg_made: float, fg_miss: float, pts_per_fg: float, pat_made: float, pat_miss: float) -> float:
    return fg_made * pts_per_fg - fg_miss + pat_made - pat_miss


# Defense: points-allowed tiers, as (upper bound inclusive, points)
PA_TIERS = [(0, 10), (6, 7), (13, 4), (20, 1), (27, 0), (34, -1), (999, -4)]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def expected_pa_points(mean_pa: float, sd: float = 9.5) -> float:
    """Expected tier points when points allowed is roughly normal around the opponent's implied total."""
    total, lo = 0.0, -1e9
    for hi, pts in PA_TIERS:
        p = _norm_cdf((hi + 0.5 - mean_pa) / sd) - _norm_cdf((lo + 0.5 - mean_pa) / sd)
        total += p * pts
        lo = hi
    return total


def defense(sacks: float, ints: float, fum_rec: float, tds: float, safeties: float, mean_pa: float) -> float:
    return sacks + 2 * ints + 2 * fum_rec + 6 * tds + 2 * safeties + expected_pa_points(mean_pa)
