# Finance project test report

**CHALLENGE (self-serve): hidden real-markets cases**

Dataset: 60 cases; seed: None; source: Price returns derived from Binance Vision public data (CC BY-NC-SA 4.0). Not endorsed by Binance. Months drawn: 58 (hidden until reveal)..

Exact checks: **0 passed, 0 failed, 0 execution errors**.
Market simulations completed: **60**. Cases not run: **0**.

Only the configured adapter and defined checks were exercised. A baseline run tests the kit, not your submission.

## Challenge integrity

- Cases were generated in memory after the adapter was fixed; no input or answer file existed during the run.
- Seed commitment (published before scoring): `3f56972b0230824e0a777b50d339949481239ebb1bc444bc38b68e2c0359ad07`
- Seed: fresh random; revealed in SEED-REVEAL.txt after this run
- Kit fingerprint: `5c90bf815848f4edd85bb2446e00382a24f003d936b3bad72f297a3b7885dad9` (compare with the organisers' published value)
- Project commit: `None`, uncommitted changes: 55
- Provenance review: none flagged.

## Coverage

| Task / scenario family | Pass | Fail | Error | Simulated |
| --- | ---: | ---: | ---: | ---: |
| markets / real_anonymised | 0 | 0 | 0 | 60 |

## Strategy results after costs

| Case | Final cash | Profit / loss | Buy-and-hold final | Cash baseline | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| 699a032ea9d4d070991b | £5,549.43 | -£6.56 | £5,437.08 | £5,555.99 | 0.11% |
| 1e0198b97044a3f6fefa | £4,881.01 | -£195.69 | £4,639.56 | £5,076.70 | 8.65% |
| ce6156d474a54620bef9 | £6,080.45 | £225.76 | £5,825.91 | £5,854.69 | 9.61% |
| 40478e7219c1b2178bda | £4,844.57 | £0.00 | £6,870.16 | £4,844.57 | 0.00% |
| 052f9a8e8ebfa66b402a | £2,954.87 | £0.00 | £3,032.04 | £2,954.87 | 0.00% |
| ec54943adfa26520bd0d | £5,875.78 | £0.00 | £7,262.61 | £5,875.78 | 0.00% |
| 316c7c360d60d8617efc | £7,700.41 | £346.34 | £7,578.46 | £7,354.07 | 2.57% |
| 76be350e4c14c2f924f9 | £2,992.37 | -£618.02 | £2,966.85 | £3,610.39 | 26.38% |
| 3067088d7989074ab84d | £3,743.61 | £486.59 | £3,560.66 | £3,257.02 | 9.05% |
| c4958f453a3f6fcfcdde | £1,866.69 | -£11.29 | £1,836.03 | £1,877.98 | 3.85% |
| b07e8ade58c69bc56f9b | £2,088.34 | £119.41 | £2,047.99 | £1,968.93 | 2.88% |
| b38376cc54a7625c5562 | £1,399.21 | -£14.00 | £1,375.91 | £1,413.21 | 5.48% |
| 1e87582708103c906f25 | £9,737.69 | £467.34 | £9,246.33 | £9,270.35 | 20.97% |
| af09ee8412c8f6694d5e | £4,273.92 | £0.00 | £4,338.02 | £4,273.92 | 0.00% |
| 75f36a930e0cdb2f27cf | £8,620.95 | -£1,218.54 | £8,369.85 | £9,839.49 | 16.31% |
| 85b7813488d922d9c8a0 | £5,136.22 | £0.00 | £5,528.44 | £5,136.22 | 0.00% |
| 31b46ac01d73ac688528 | £6,153.59 | £120.93 | £5,522.20 | £6,032.66 | 21.07% |
| 1da36f61e89d987e3c3f | £1,507.24 | £212.12 | £1,449.76 | £1,295.12 | 16.85% |
| d136cdede19e97299ebd | £3,802.76 | -£78.90 | £3,745.66 | £3,881.66 | 12.54% |
| e23ed4c306ce70e24b27 | £4,426.01 | -£214.02 | £4,370.51 | £4,640.03 | 6.38% |
| 85c436c0f96b071c52e8 | £2,269.73 | £80.10 | £2,230.37 | £2,189.63 | 3.87% |
| 328be660fd473ace58c9 | £3,453.48 | £974.71 | £3,317.40 | £2,478.77 | 12.11% |
| 3032c21286734edfccf9 | £10,664.61 | £1,170.71 | £10,568.22 | £9,493.90 | 13.88% |
| 06b1e6ca1f0ff5c4389f | £5,433.87 | -£64.33 | £5,123.88 | £5,498.20 | 11.49% |
| fd8aecab19aa10c7d0e5 | £3,546.03 | -£140.66 | £3,457.83 | £3,686.69 | 7.59% |
| 7888065e43640b65a97a | £10,285.08 | £1,329.64 | £10,218.96 | £8,955.44 | 11.73% |
| 0f1244fb3523c618f2e7 | £9,801.91 | £0.00 | £10,879.14 | £9,801.91 | 0.00% |
| c7346ce9026a35c3b29e | £7,940.28 | -£186.00 | £7,724.31 | £8,126.28 | 2.62% |
| a873893e0b5d780bd256 | £9,111.73 | -£604.64 | £8,679.68 | £9,716.37 | 9.54% |
| 43f072fc6342d239bc85 | £8,984.50 | £544.00 | £8,652.74 | £8,440.50 | 1.92% |

Showing 30 of 60 paths; every result and trade is in case-results.jsonl.
Completing a simulation does not mean the strategy is good. Compare losses, costs, baselines and drawdown. These price paths do not model real market liquidity or predict returns.

**Across 60 paths:** 47 wins, 13 losses, 0 ties vs buy-and-hold; beat cash on 25. Median excess 2.03% of capital; mean max drawdown 7.56%; sign-test p = 0.000.
**Beat buy-and-hold more often than chance would explain on these paths.**

## Next action

Run your actual submission adapter if this was a baseline. Add project-specific checks, then rerun with a fresh seed. Anything outside these checks remains untested.

Hidden challenge run: no input or answer file existed while the adapter ran, and the kit's reference calculations were disabled for it.
