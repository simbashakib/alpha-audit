# Finance project test report

**Participant adapter**

Dataset: 25 cases; seed: 42; source: Price returns derived from Binance Vision public data (CC BY-NC-SA 4.0). Not endorsed by Binance. Practice set; months: ADAUSDT 1h 2022-01, ADAUSDT 1h 2024-06, ADAUSDT 1h 2026-09, ADAUSDT 4h 2024-05, ADAUSDT 4h 2024-11, BNBUSDT 1h 2025-07, BTCUSDT 1h 2023-08, BTCUSDT 1h 2024-04, BTCUSDT 4h 2024-04, BTCUSDT 4h 2024-07, BTCUSDT 4h 2025-03, DOGEUSDT 1h 2022-04, DOGEUSDT 1h 2022-11, DOGEUSDT 1h 2023-05, DOGEUSDT 4h 2024-12, ETHUSDT 1h 2022-11, LTCUSDT 1h 2021-03, LTCUSDT 4h 2023-03, SOLUSDT 1h 2022-01, SOLUSDT 1h 2024-09, SOLUSDT 1h 2025-02, SOLUSDT 1h 2026-03, SOLUSDT 4h 2022-11, XRPUSDT 1h 2024-10, XRPUSDT 4h 2022-04.

Exact checks: **0 passed, 0 failed, 0 execution errors**.
Market simulations completed: **25**. Cases not run: **0**.

Only the configured adapter and defined checks were exercised. A baseline run tests the kit, not your submission.

## Coverage

| Task / scenario family | Pass | Fail | Error | Simulated |
| --- | ---: | ---: | ---: | ---: |
| markets / real_anonymised | 0 | 0 | 0 | 25 |

## Strategy results after costs

| Case | Final cash | Profit / loss | Buy-and-hold final | Cash baseline | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5e5fb830aede001a789e | £2,391.35 | £66.40 | £2,357.15 | £2,324.95 | 3.24% |
| 42b387135299009e3c6c | £6,361.53 | -£203.88 | £6,207.63 | £6,565.41 | 3.60% |
| 876c4a93ed71991e03b5 | £7,126.81 | -£418.56 | £7,090.33 | £7,545.37 | 22.36% |
| 6de315ad2605d7191f3c | £8,199.58 | £0.00 | £9,936.78 | £8,199.58 | 0.00% |
| 31a4bccf2cc628142234 | £4,245.12 | £0.00 | £5,322.62 | £4,245.12 | 0.00% |
| 7b762969b17ba793b268 | £9,257.06 | £0.00 | £10,307.70 | £9,257.06 | 0.00% |
| fde2e4765ddb42c5136e | £9,892.03 | £637.34 | £9,563.51 | £9,254.69 | 4.81% |
| 6d3f06e6140bf04fdd22 | £8,996.30 | £410.66 | £8,664.68 | £8,585.64 | 6.82% |
| 246a26e4481ab7b42725 | £5,414.93 | -£176.50 | £5,254.62 | £5,591.43 | 8.50% |
| 73fc955659752f7ff702 | £3,799.12 | -£185.85 | £3,708.85 | £3,984.97 | 10.78% |
| bd33f82af3a7c1470c0c | £3,780.77 | £0.00 | £4,124.91 | £3,780.77 | 0.00% |
| d188ed4139a160626ce2 | £2,277.89 | £0.00 | £2,423.65 | £2,277.89 | 0.00% |
| 70c03cb6a1c262ec889c | £6,536.34 | -£477.74 | £6,232.51 | £7,014.08 | 16.03% |
| 311c87e74d4bd63dfa4d | £5,693.72 | -£1,012.96 | £5,560.47 | £6,706.68 | 20.45% |
| b59f350c59db14b3e310 | £7,951.56 | £326.95 | £7,544.62 | £7,624.61 | 5.86% |
| c5715d7bdab26c5ecee8 | £9,803.99 | £0.00 | £10,991.45 | £9,803.99 | 0.00% |
| 6e0682a3046e42e5b7fd | £2,967.23 | -£307.48 | £2,827.72 | £3,274.71 | 9.44% |
| b88a734a37626ac2f2da | £6,247.12 | -£274.20 | £6,123.36 | £6,521.32 | 6.55% |
| 1e9d1dacfacfe9cce9fc | £5,404.04 | -£321.78 | £5,278.84 | £5,725.82 | 30.00% |
| e61f21169e300f59c534 | £4,382.48 | £0.00 | £4,375.39 | £4,382.48 | 0.00% |
| 41ff790722c429eff141 | £6,612.73 | £689.08 | £6,442.03 | £5,923.65 | 4.72% |
| f12b01faeba86d0ea399 | £7,046.82 | £273.02 | £7,037.46 | £6,773.80 | 8.01% |
| 0211e4fb99bd883bf54e | £3,139.34 | £0.00 | £3,252.06 | £3,139.34 | 0.00% |
| 11c60ee2772891ecebe8 | £17,310.63 | £11,272.13 | £17,068.31 | £6,038.50 | 17.67% |
| 0100473942282d64b13b | £4,043.72 | -£44.62 | £3,889.34 | £4,088.34 | 1.20% |

Showing 25 of 25 paths; every result and trade is in case-results.jsonl.
Completing a simulation does not mean the strategy is good. Compare losses, costs, baselines and drawdown. These price paths do not model real market liquidity or predict returns.

**Across 25 paths:** 18 wins, 7 losses, 0 ties vs buy-and-hold; beat cash on 7. Median excess 1.98% of capital; mean max drawdown 7.20%; sign-test p = 0.043.
**Beat buy-and-hold more often than chance would explain on these paths.**

## Next action

Run your actual submission adapter if this was a baseline. Add project-specific checks, then rerun with a fresh seed. Anything outside these checks remains untested.

Expected answers stayed with the evaluator; adapter requests contained inputs only. The local answer files remain visible to developers, so this practice run is not cheat-proof. Use `bench.py challenge` for a result you can claim.
