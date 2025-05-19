import math

BROADLEAF = dict(K=30.0, Y=0.67)


def est_dbh_radius(height_m: float, K: float = BROADLEAF["K"], Y: float = BROADLEAF["Y"]):
    if height_m <= 0:
        return 0.05
    dbh = (height_m / K) ** (1.0 / Y)
    return max(dbh * 0.5, 0.02)