# Finance project test report

**Participant adapter**

Dataset: 25 cases; seed: 42; source: AgentVerse seeded generator v2.

Exact checks: **0 passed, 0 failed, 0 execution errors**.
Market simulations completed: **25**. Cases not run: **0**.

Only the configured adapter and defined checks were exercised. A baseline run tests the kit, not your submission.

## Coverage

| Task / scenario family | Pass | Fail | Error | Simulated |
| --- | ---: | ---: | ---: | ---: |
| markets / falling | 0 | 0 | 0 | 4 |
| markets / flat | 0 | 0 | 0 | 4 |
| markets / jump | 0 | 0 | 0 | 4 |
| markets / regime_change | 0 | 0 | 0 | 4 |
| markets / rising | 0 | 0 | 0 | 5 |
| markets / volatile | 0 | 0 | 0 | 4 |

## Strategy results after costs

| Case | Final cash | Profit / loss | Buy-and-hold final | Cash baseline | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| 227a5d19d073bbd23bfc | £5,035.91 | £533.98 | £5,158.97 | £4,501.93 | 0.41% |
| 2b58cd3c7f4c4cca9964 | £8,095.40 | -£90.05 | £5,229.93 | £8,185.45 | 1.10% |
| 8988628061d7affd5ac2 | £4,264.77 | £0.00 | £4,232.47 | £4,264.77 | 0.00% |
| 47eafd5785610731f72b | £2,588.62 | -£639.38 | £2,210.37 | £3,228.00 | 19.80% |
| 6a0961c541a0e9d214e4 | £11,993.25 | £2,629.62 | £9,066.69 | £9,363.63 | 7.33% |
| 8720ab66e154cb2dc50e | £2,324.00 | -£957.03 | £2,430.83 | £3,281.03 | 31.81% |
| e3df28507919de41aa43 | £5,301.60 | £1,458.39 | £5,545.57 | £3,843.21 | 0.65% |
| 619230dc8dbd2abe3baa | £5,548.17 | £0.00 | £4,697.27 | £5,548.17 | 0.00% |
| d8118a2a0e9b07cc77c9 | £9,459.08 | £0.00 | £9,417.42 | £9,459.08 | 0.00% |
| a72ad17f19b4b896922d | £8,009.42 | -£925.54 | £12,616.64 | £8,934.96 | 23.66% |
| eb2e17b1f756730d6b0e | £3,588.98 | -£246.86 | £3,771.06 | £3,835.84 | 6.43% |
| e619be0a65b6cad688a6 | £1,443.91 | -£54.38 | £1,927.87 | £1,498.29 | 7.05% |
| 1a4808eba764ca7d0e0f | £8,376.74 | £2,798.52 | £8,885.70 | £5,578.22 | 0.40% |
| 968fae5acb576ac5e398 | £9,074.95 | -£252.15 | £7,368.96 | £9,327.10 | 2.70% |
| b2b2bcaa50661c76efbd | £7,921.28 | £0.00 | £7,855.14 | £7,921.28 | 0.00% |
| 2e12379a3310489635d8 | £6,265.74 | £1,195.67 | £9,216.10 | £5,070.07 | 26.89% |
| f38c505acab42aac7419 | £2,553.83 | £439.54 | £1,950.01 | £2,114.29 | 7.63% |
| 9959f1cf1ab6d82e0c05 | £7,348.34 | £0.00 | £5,113.92 | £7,348.34 | 0.00% |
| 78f125674e8df796346e | £13,375.33 | £4,889.35 | £13,870.42 | £8,485.98 | 0.79% |
| d8f0752f819c01ae40ef | £3,964.07 | -£80.34 | £2,788.19 | £4,044.41 | 1.98% |
| b3062579f60f8e00de25 | £1,647.63 | £0.00 | £1,638.91 | £1,647.63 | 0.00% |
| 336e02b12c25c2599b49 | £9,324.65 | -£436.13 | £9,309.41 | £9,760.78 | 20.43% |
| 27b4d914fe76f5795dce | £5,876.45 | -£178.48 | £6,042.47 | £6,054.93 | 8.68% |
| 0871e8121cb419668fdf | £5,147.27 | £978.86 | £5,204.93 | £4,168.41 | 3.69% |
| 52b93bce5a534aa985ab | £9,879.29 | £1,646.18 | £10,356.68 | £8,233.11 | 0.38% |

Showing 25 of 25 paths; every result and trade is in case-results.jsonl.
Completing a simulation does not mean the strategy is good. Compare losses, costs, baselines and drawdown. These price paths do not model real market liquidity or predict returns.

**Across 25 paths:** 13 wins, 12 losses, 0 ties vs buy-and-hold; beat cash on 9. Median excess 0.15% of capital; mean max drawdown 6.87%; sign-test p = 1.000.
**Not distinguishable from buy-and-hold on these paths; more paths or a better edge is needed.**

## Next action

Run your actual submission adapter if this was a baseline. Add project-specific checks, then rerun with a fresh seed. Anything outside these checks remains untested.

Expected answers stayed with the evaluator; adapter requests contained inputs only. The local answer files remain visible to developers, so this practice run is not cheat-proof. Use `bench.py challenge` for a result you can claim.
