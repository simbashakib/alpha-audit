# Finance project test report

**CHALLENGE (self-serve): hidden markets cases**

Dataset: 60 cases; seed: None; source: AgentVerse seeded generator v2 (hidden challenge).

Exact checks: **0 passed, 0 failed, 0 execution errors**.
Market simulations completed: **60**. Cases not run: **0**.

Only the configured adapter and defined checks were exercised. A baseline run tests the kit, not your submission.

## Challenge integrity

- Cases were generated in memory after the adapter was fixed; no input or answer file existed during the run.
- Seed commitment (published before scoring): `6ff34ce16a48a5beca362854ccd51b3b6e9f577d27af45a8494c509175ed5d86`
- Seed: fresh random; revealed in SEED-REVEAL.txt after this run
- Kit fingerprint: `5c90bf815848f4edd85bb2446e00382a24f003d936b3bad72f297a3b7885dad9` (compare with the organisers' published value)
- Project commit: `None`, uncommitted changes: 55
- Provenance review: none flagged.

## Coverage

| Task / scenario family | Pass | Fail | Error | Simulated |
| --- | ---: | ---: | ---: | ---: |
| markets / falling | 0 | 0 | 0 | 10 |
| markets / flat | 0 | 0 | 0 | 10 |
| markets / jump | 0 | 0 | 0 | 10 |
| markets / regime_change | 0 | 0 | 0 | 10 |
| markets / rising | 0 | 0 | 0 | 10 |
| markets / volatile | 0 | 0 | 0 | 10 |

## Strategy results after costs

| Case | Final cash | Profit / loss | Buy-and-hold final | Cash baseline | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0b963ad02548e84a7223 | £7,469.74 | £0.00 | £12,378.12 | £7,469.74 | 0.00% |
| 82df100fc150aa7713a8 | £3,363.50 | -£2,297.64 | £3,254.69 | £5,661.14 | 40.58% |
| c211454c9a05c7bcc2d8 | £3,408.58 | £0.00 | £3,377.46 | £3,408.58 | 0.00% |
| 0a97c94ac73299d90c50 | £8,203.83 | £64.94 | £7,091.83 | £8,138.89 | 3.62% |
| cbc1da8dff058fe434b3 | £2,088.84 | £0.00 | £1,968.24 | £2,088.84 | 0.00% |
| ae2fc6a159d12effcb95 | £8,385.36 | -£30.43 | £5,856.05 | £8,415.79 | 1.47% |
| e0152debcc07b319d7c6 | £2,817.97 | £0.00 | £4,278.25 | £2,817.97 | 0.00% |
| d5e0babb2dfa1102e71a | £7,590.09 | -£2,302.40 | £7,221.09 | £9,892.49 | 23.27% |
| e9ddb9cf2a640f60f290 | £3,283.61 | £0.00 | £3,264.91 | £3,283.61 | 0.00% |
| 4d778dbb3d741b4fb5bd | £5,448.58 | £0.00 | £8,359.52 | £5,448.58 | 0.00% |
| f64ad30426b2c6690ccf | £4,467.70 | £0.00 | £4,329.26 | £4,467.70 | 0.00% |
| d87135df3ba98a07c3d6 | £1,181.80 | -£538.82 | £1,160.74 | £1,720.62 | 31.94% |
| 3c40135c60ab95919fff | £3,226.85 | £0.00 | £3,508.81 | £3,226.85 | 0.00% |
| 217bce7146f1fad56cf4 | £4,996.75 | -£2,126.48 | £4,351.91 | £7,123.23 | 29.85% |
| eb73dd8b8e25f30c0ae4 | £8,264.87 | £0.00 | £8,238.25 | £8,264.87 | 0.00% |
| d6d26d0f04667df7557b | £5,936.49 | -£535.22 | £5,045.21 | £6,471.71 | 26.79% |
| 3469883439c5c8da3909 | £2,361.66 | £0.00 | £2,187.26 | £2,361.66 | 0.00% |
| f252dc4bfe54d2728a04 | £5,527.17 | -£47.40 | £3,789.33 | £5,574.57 | 0.85% |
| 306552a82b6e5ea2297d | £3,243.40 | £0.00 | £3,908.43 | £3,243.40 | 0.00% |
| a45d418de96963203e6c | £2,194.19 | -£268.47 | £2,069.31 | £2,462.66 | 10.90% |
| 50283151b8824c338b33 | £9,527.44 | £0.00 | £9,467.92 | £9,527.44 | 0.00% |
| d0087dd6f1e0db80395a | £4,015.58 | £234.64 | £3,732.64 | £3,780.94 | 6.79% |
| cee0770fcb6b263e1440 | £7,085.24 | £0.00 | £6,595.29 | £7,085.24 | 0.00% |
| 944329873bff49e5ecca | £7,035.61 | -£2,201.88 | £6,914.93 | £9,237.49 | 30.30% |
| 52d0566068bbb2ee726e | £5,514.06 | £0.00 | £8,084.18 | £5,514.06 | 0.00% |
| 788c41af33cc8e5218fe | £5,388.03 | -£2,130.37 | £4,467.98 | £7,518.40 | 28.33% |
| 6d8216cd8e0026788460 | £7,444.68 | £0.00 | £7,392.16 | £7,444.68 | 0.00% |
| 2687ba66c426ef0f9ecd | £3,587.63 | -£211.72 | £3,078.39 | £3,799.35 | 22.29% |
| 9960b966b9ab63bf87d4 | £7,825.35 | £0.00 | £6,948.55 | £7,825.35 | 0.00% |
| 812e6048cdc375938593 | £11,349.03 | £2,390.02 | £11,075.23 | £8,959.01 | 3.78% |

Showing 30 of 60 paths; every result and trade is in case-results.jsonl.
Completing a simulation does not mean the strategy is good. Compare losses, costs, baselines and drawdown. These price paths do not model real market liquidity or predict returns.

**Across 60 paths:** 43 wins, 17 losses, 0 ties vs buy-and-hold; beat cash on 6. Median excess 1.26% of capital; mean max drawdown 8.26%; sign-test p = 0.001.
**Beat buy-and-hold more often than chance would explain on these paths.**

## Next action

Run your actual submission adapter if this was a baseline. Add project-specific checks, then rerun with a fresh seed. Anything outside these checks remains untested.

Hidden challenge run: no input or answer file existed while the adapter ran, and the kit's reference calculations were disabled for it.
