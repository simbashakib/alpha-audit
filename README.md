# Alpha Auditor

Alpha Auditor is a Python research workbench built for the finance track of the UCL AgentVerse hackathon. It turns a market hypothesis into a trading rule a researcher can inspect, test on a chronological holdout, and replay in a Jupyter notebook.

## A worked example

**Do Coca-Cola and Pepsi move together, and can you trade the divergence?** The included example uses adjusted daily prices from 2018 through 2025. It fits a log-price hedge during formation, freezes it for the holdout, and trades deviations beyond two standard deviations. Orders fill at the next open. Each entry and exit costs 10 basis points of portfolio notional.

| Question | Result on the included data |
| --- | --- |
| Do the daily returns move together? | Formation log-return correlation was 0.739. |
| Is there evidence of a long-run price relationship? | The Engle–Granger p-value was 0.101. The formation data did not establish cointegration at the 5% level. |
| Did the trading rule make money in the holdout? | It lost 23.59% net of modelled costs across 30 trades entered in the holdout, which began on 30 December 2021. |

The example shows why the app reports correlation, cointegration and trading performance separately. The parameters and dates are in the [example specification](examples/ko_pep.json).

## Run it locally

Use Python 3.12 or newer. From the repository root, create a virtual environment and install the dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Run the example without an API key.** The repository includes the market data for this replay. The first command prints formation and holdout results and exports the report. The second computes the same SPY experiment twice and compares the complete reports, excluding the creation timestamp.

```bash
python replay.py examples/ko_pep.json --pdf output/ko_pep.pdf --ipynb output/ko_pep.ipynb
python replay.py --verify-repeatability
```

To execute all notebook cells in a fresh Jupyter kernel and save their output, add `--execute-ipynb` to the first command. To run an exported notebook manually, open it inside this checkout with a kernel that has the project dependencies. It includes the aligned market data and imports the Python engine to recompute the results.

**Run the web app.** Set `GEMINI_API_KEY` in your server environment or a local `.env` file. You can set `ALPHA_AUDITOR_GEMINI_MODEL` to a model available to your account. Start the server and open [the workbench](http://127.0.0.1:8000).

```bash
python -m uvicorn app.server:app --port 8000
```

Gemini receives the hypothesis and preceding user turns. It selects one of five supported strategy templates, requests clarification, or returns no match. The local compiler builds the rule, and the researcher confirms the assets, parameters and costs before running it. Python performs the statistical tests, backtest and report calculations. The browser loads Plotly from a CDN. Uncached tickers require a Yahoo Finance connection.

## Research design

- Signals use observed bars and execute at the next configured open or close. Pairs trades use signed, unit-gross weights. Daily P&L and trade records include modelled commission and slippage.
- The report separates formation and holdout performance. It records the specification, dataset hash, computational environment and random seeds. A run with no holdout exposure is labelled explicitly.
- Five stress tests examine dependence on winning trades, execution delays and costs, market regimes, random entry timing, and hedge or parameter stability. The random-entry comparison uses 500 seeded schedules matched to the observed trade durations, directions and leg weights.
- Daily-bar execution approximates real fills. The simulation omits market impact and borrow fees and uses 252 observations per year for annualisation. Exploring new parameters on a previously inspected holdout makes that analysis exploratory. The scorecard summarises historical evidence.

## Code and tests

| Path | Role |
| --- | --- |
| `engine/` | Specification schemas, signal generation, backtesting, econometrics, stress tests and exports |
| `app/server.py` and `frontend/` | FastAPI endpoints and browser workbench |
| `data/cache/` | Bundled adjusted daily bars. The snapshot manifest records each file's SHA-256 hash. |
| `adapter.py` | JSONL interface for the hackathon evaluation harness |
| `agentverse/` | Organiser-provided starter kit, fixtures and evaluation harness |
| `eval-runs/challenge*/` | Archived self-serve challenge results from the original hackathon submission |

```bash
python -m pytest -q
```

The project tests exercise next-bar fills, transaction costs, prefix invariance, split handling, simulated Gemini responses, API endpoints and notebook reconciliation. The archived challenge results describe the original submission. The automated workflow runs the project tests and replay check on subsequent pushes and pull requests.
