#!/usr/bin/env python3
"""Predict pump.fun top-5 coins by market cap, 24h out — make.fun bounty.

Reads $MAKEFUN_INPUT_DIR (else /input): task.json + history.ndjson (10-min
marks of ranked pump.fun coins). Writes $MAKEFUN_OUTPUT_DIR (else /output)/
forecast.json with exactly {probabilities, top5, sizes}.

Method: log-cap dynamics. Each coin's log10(market cap in SOL) series is
fit with a (shrunk, blended long/short) linear velocity; at each answer
read from task.json its forecast mean is last log-cap + velocity integrated
with exponential decay (saturating displacement), and its sd grows with
sqrt(horizon). Places come from thresholds midway between neighbouring
forecast means, so P(coin at place k) is the normal mass of its log-cap
between the two thresholds around place k, averaged over the seven reads.
A generous share of every place is left unnamed: coins born after the
history ends (and mayhem coins still on their curve) can take places, and
in the practice 2 of the top 5 at the last read were such newcomers.
Sizes are a normal on log10 cap centred on the place-k probability-weighted
forecast mean, sd growing with horizon.

All timing (t0, history cut, answer reads) is read from task.json; nothing
about the horizon is hardcoded, so the same code runs the 1-hour practice
and the 7-13-hour final. Stdlib only, no network, fully deterministic.
"""

import json
import math
import os
import statistics
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tunables (v3: tuned on cut-at-mark-3 backtest; cap 0.30 chosen as the
# balanced point -- lower caps kept gaining on the 3-read backtest but that
# backtest has only 2-3 points per coin, so the final (66 marks) deserves
# more weight on the model than the tiny backtest suggests) ---
TAU_H = 1.5            # velocity decay time (h); displacement saturates at v*TAU
VEL_CLAMP_LO = -0.6    # dex/h
VEL_CLAMP_HI = 2.0     # dex/h
DISP_CLAMP_LO = -1.0   # dex total displacement
DISP_CLAMP_HI = 1.5    # dex total displacement
SD0 = 0.20             # base log-cap sd (dex)
SD_RATE = 0.25         # sd growth per sqrt(hour)
SD_DISP = 0.08         # extra sd per dex of |displacement|
SD_NEW = 0.15          # extra sd for coins seen only once
SHORT_W = 0.35         # weight of short-window velocity in the blend
SHORT_H = 2.0          # short window length (h)
CAND_TOP_N = 80        # candidates from the last mark's top ranks
CAND_RISER_N = 30      # extra fast risers from deeper ranks
PLACE_CAP = 0.30       # max named mass per place (rest unnamed, for newcomers)
MIN_P = 0.012          # drop probability rows below this
SIZE_SD0 = 0.40        # base size sd (dex)
SIZE_SD_RATE = 0.20    # size sd growth per sqrt(hour)
MICRO = 1000000        # probabilities floored to 6 decimals


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_range(mu, sd, lo, hi):
    if sd <= 0:
        return 1.0 if lo <= mu < hi else 0.0
    a = 0.0 if lo == float("-inf") else phi((lo - mu) / sd)
    b = 1.0 if hi == float("inf") else phi((hi - mu) / sd)
    return max(0.0, b - a)


def linfit_vel(pts):
    """Slope of uniform linear fit of log-cap vs time (hours). pts: [(t, y)]."""
    if len(pts) < 2:
        return None
    n = len(pts)
    mt = sum(t for t, _ in pts) / n
    my = sum(y for _, y in pts) / n
    den = sum((t - mt) ** 2 for t, _ in pts)
    if den <= 1e-12:
        return 0.0
    num = sum((t - mt) * (y - my) for t, y in pts)
    return num / den


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task.get("places", 5))
    edges = task["size"]["edges"]
    output = task.get("output", "forecast.json")
    reads = [parse_time(s) for s in task.get("reads", [])]

    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                m = json.loads(line)
                if m.get("ok") is True and isinstance(m.get("rows"), list):
                    marks.append(m)
    marks.sort(key=lambda m: m["at"])
    if not marks:
        raise SystemExit("the history holds no ok mark")
    last = marks[-1]
    t_last = parse_time(last["at"])
    t_first = parse_time(marks[0]["at"])
    if not reads:
        # Fallback: hourly reads starting 1h after the last mark.
        from datetime import timedelta
        reads = [t_last + timedelta(hours=h) for h in range(1, 8)]
    horizons = [max(0.0, (r - t_last).total_seconds() / 3600.0) for r in reads]

    # --- per-coin log-cap series ---
    series = {}   # mint -> [(t_h, logcap)]
    last_rank = {}
    last_lc = {}
    last_row = {}
    for m in marks:
        t = (parse_time(m["at"]) - t_first).total_seconds() / 3600.0
        for r in m["rows"]:
            try:
                lc = math.log10(int(r["market_cap_lamports"])) - 9
            except (KeyError, TypeError, ValueError):
                continue
            series.setdefault(r["mint"], []).append((t, lc))
    for r in sorted(last["rows"], key=lambda x: x["rank"]):
        mint = r["mint"]
        last_rank[mint] = r["rank"]
        try:
            last_lc[mint] = math.log10(int(r["market_cap_lamports"])) - 9
        except (KeyError, TypeError, ValueError):
            continue
        last_row[mint] = r
    if len(last_lc) < places:
        raise SystemExit("the last mark ranks fewer coins than places")

    t0 = (t_last - t_first).total_seconds() / 3600.0
    # velocities
    raw_vel = {}
    for mint, pts in series.items():
        v = linfit_vel(pts)
        if v is not None:
            raw_vel[mint] = clamp(v, VEL_CLAMP_LO, VEL_CLAMP_HI)
    v_med = statistics.median(raw_vel.values()) if raw_vel else 0.0
    vel = {}
    npts = {}
    for mint in last_lc:
        pts = series.get(mint, [])
        npts[mint] = len(pts)
        if len(pts) >= 2:
            v_long = raw_vel.get(mint, v_med)
            short = [(t, y) for (t, y) in pts if t >= t0 - SHORT_H]
            v_short = linfit_vel(short)
            if v_short is None:
                v_short = v_long
            v_short = clamp(v_short, VEL_CLAMP_LO, VEL_CLAMP_HI)
            v = (1.0 - SHORT_W) * v_long + SHORT_W * v_short
            # shrink toward the population median by observation count
            v = (len(pts) * v + 2.0 * v_med) / (len(pts) + 2.0)
        else:
            v = 0.5 * v_med
        vel[mint] = clamp(v, VEL_CLAMP_LO, VEL_CLAMP_HI)

    # --- candidates: top ranks + fast risers ---
    by_rank = sorted(last_lc, key=lambda m: last_rank[m])
    cand = list(by_rank[:CAND_TOP_N])
    cand_set = set(cand)
    risers = sorted((m for m in last_lc if m not in cand_set),
                    key=lambda m: -vel.get(m, 0.0))
    for m in risers[:CAND_RISER_N]:
        if vel.get(m, 0.0) > 0.15:
            cand.append(m)
            cand_set.add(m)

    def mu_sd(mint, t):
        v = vel.get(mint, 0.0)
        disp = v * TAU_H * (1.0 - math.exp(-t / TAU_H)) if t > 0 else 0.0
        disp = clamp(disp, DISP_CLAMP_LO, DISP_CLAMP_HI)
        mu = last_lc[mint] + disp
        sd = SD0 + SD_RATE * math.sqrt(t) + SD_DISP * abs(disp)
        if npts.get(mint, 0) < 2:
            sd += SD_NEW
        return mu, sd, disp

    # --- P(coin at place k), averaged over reads ---
    prob = {k: {} for k in range(1, places + 1)}
    size_center = {k: [] for k in range(1, places + 1)}  # (t, centre, weight)
    for t in horizons:
        mus = {}
        sds = {}
        for m in cand:
            mu, sd, _ = mu_sd(m, t)
            mus[m] = mu
            sds[m] = sd
        order = sorted(cand, key=lambda m: (-mus[m], m))
        # thresholds between consecutive forecast means
        thr = {}  # thr[k] = boundary between place k and k+1, k=1..places
        for k in range(1, places + 1):
            if k < len(order):
                thr[k] = 0.5 * (mus[order[k - 1]] + mus[order[k]])
            else:
                thr[k] = mus[order[-1]] - 0.3
        for m in cand:
            mu, sd = mus[m], sds[m]
            for k in range(1, places + 1):
                lo = thr[k]
                hi = float("inf") if k == 1 else thr[k - 1]
                p = norm_range(mu, sd, lo, hi)
                if p > 1e-6:
                    prob[k][m] = prob[k].get(m, 0.0) + p
        # size centres: midpoint of the place-k log-cap band at this read
        # (place 1: the top forecast mean). The probability-weighted mean
        # was dragged down by the long tail of low coins with small p.
        for k in range(1, places + 1):
            if k == 1:
                centre = mus[order[0]]
            else:
                centre = 0.5 * (thr[k - 1] + thr[k])
            size_center[k].append((t, centre))

    for k in prob:
        for m in prob[k]:
            prob[k][m] /= len(horizons)

    # --- probabilities rows: cap named mass, floor to 6 decimals ---
    rows = []
    for k in range(1, places + 1):
        items = sorted(prob[k].items(), key=lambda kv: (-kv[1], kv[0]))
        items = [(m, p) for (m, p) in items if p >= MIN_P]
        if not items:
            continue
        s = sum(p for _, p in items)
        scale = min(1.0, PLACE_CAP / s) if s > 0 else 0.0
        for m, p in items:
            q = math.floor(p * scale * MICRO) / MICRO
            if q > 0:
                rows.append({"mint": m, "place": k, "p": q})
    rows.sort(key=lambda r: (r["place"], -r["p"], r["mint"]))

    # --- top5: order by forecast mean at the median horizon ---
    t_med = sorted(horizons)[len(horizons) // 2]
    top5 = sorted(cand, key=lambda m: (-mu_sd(m, t_med)[0], last_rank[m]))[:places]
    # fill from last-mark ranks if short (should not happen)
    if len(top5) < places:
        for m in by_rank:
            if m not in top5:
                top5.append(m)
            if len(top5) == places:
                break

    # --- sizes: normal around the place centre, sd growing with horizon ---
    sizes = []
    for k in range(1, places + 1):
        centres = size_center[k]
        if not centres:
            # fallback: last mark's place-k log-cap
            centres = [(0.0, last_lc[by_rank[k - 1]])]
        bin_p = {}
        for (t, centre) in centres:
            sd = SIZE_SD0 + SIZE_SD_RATE * math.sqrt(t)
            for b in range(0, len(edges) + 1):
                lo = float("-inf") if b == 0 else edges[b - 1]
                hi = float("inf") if b == len(edges) else edges[b]
                p = norm_range(centre, sd, lo, hi)
                if p > 1e-7:
                    bin_p[b] = bin_p.get(b, 0.0) + p / len(centres)
        for b in sorted(bin_p):
            q = math.floor(bin_p[b] * MICRO) / MICRO
            if q > 0:
                sizes.append({"place": k, "bin": b, "p": q})
    sizes.sort(key=lambda r: (r["place"], r["bin"]))

    result = {"probabilities": rows, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output)
    with open(out_path, "w") as f:
        f.write(json.dumps(result) + "\n")
    print(f"Wrote {out_path} ({os.path.getsize(out_path)} bytes, "
          f"{len(rows)} prob rows, {len(sizes)} size rows, top5={top5})")


if __name__ == "__main__":
    main()
