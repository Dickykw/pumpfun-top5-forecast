# Predict the top 5 new Solana coins by market cap, 24 hours out

Entry for make.fun bounty **pumpfun-top5-forecast-01** — forecast the top 5
new Solana coins by market cap, 24 hours out, from pump.fun history.

## What it does

`main.py` (Python 3.13, standard library only, no network) reads
`$MAKEFUN_INPUT_DIR` (`task.json` + `history.ndjson`) and writes
`$MAKEFUN_OUTPUT_DIR/forecast.json` with exactly three keys:
`probabilities`, `top5` and `sizes`.

Model: log-cap dynamics. Per coin, the log10 market cap (SOL) series is fit
with a shrunk, long/short-blend linear velocity (clamped), integrated with
exponential decay, plus diffusion (sd grows with sqrt of horizon). The
probability of each place comes from normal mass between thresholds midway
between neighbouring forecast means, averaged over the 7 answer reads in
`task.json`. Candidates are the last mark's top 80 plus fast risers. Each
place keeps at most 0.30 of its probability on named coins so at least 0.70
stays unnamed for coins born after the cut. Sizes are normal on log10 cap
around the place-k band midpoint. All timing comes from `task.json`; the
program is deterministic (two runs write identical bytes).

## Local check

Backtest on the kit practice history (cut at mark 3, scored on marks 4-6):
model -44.094 vs house persistence baseline -49.958 (order -23.713 vs
-26.249, sizes -20.381 vs -23.709) — better on all three reads.

Run it with:

```
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=<out> python3 main.py
```
