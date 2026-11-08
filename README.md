# Alpha Auditor — Quantitative Hypothesis Verification Workbench

Alpha Auditor is a strictly causal, deterministic quantitative research workbench that translates plain-English market intuitions into audited strategy specifications, out-of-sample backtests, 5-test robustness suites, and self-contained executable Jupyter Notebooks (`.ipynb`) and PDF reports.

## Key Capabilities & Causal Invariants

1. **Multi-Interpretation Hypothesis Compiler (`engine/hypothesis_parser.py`)**
   - Translates plain-English hypotheses into typed `StrategySpec` specifications.
   - Separates **User-Specified Parameters** from **Software-Inferred Defaults** and requires explicit human researcher confirmation before any backtest can execute.
   - Disambiguates co-movement intuitions (e.g., *"Do Pepsi and Apple move together?"*) into three distinct statistical lenses:
     1. **Return Co-Movement** (daily Pearson correlation $\rho(r_A, r_B)$ & beta),
     2. **Long-Run Level Equilibrium** (Engle-Granger two-step cointegration ADF $p$-value & Ornstein-Uhlenbeck half-life $t_{1/2} = -\ln(2)/\ln(1+\lambda)$), and
     3. **Active Spread-Reversion Trading Rule P&L** (net of fees/slippage on held-out out-of-sample data).

2. **Strictly Causal Signal & Execution Engine (`engine/backtester.py`)**
   - **Zero Look-Ahead Filter**: Signals at bar $t$ depend solely on data up to $t$ ($\mathcal{F}_t$). In `ols_log` pairs mode, the log-price OLS hedge ratio $(\hat{\alpha}_t, \hat{\beta}_t)$ is estimated via an **expanding causal window $[0..t]$** inside the Formation Period ($t < t_{\text{split}}$) and **frozen at $(\hat{\alpha}_{\text{split}}, \hat{\beta}_{\text{split}})$** throughout the Held-Out Out-of-Sample Period ($t \ge t_{\text{split}}$).
   - **Unit Gross Exposure & Signed Hedge Weights**: Two-asset positions size legs with $w_A = \pm\frac{1}{1+|\hat{\beta}|}$ and $w_B = \mp\frac{\hat{\beta}}{1+|\hat{\beta}|}$ so $|w_A| + |w_B| = 1.0$ while preserving the sign of $\hat{\beta}$.
   - **Exact Compounded Trade & Daily Accounting**: Per-trade net return matches the compounded ratio of net equity across the trade lifecycle, including end-of-sample open-position liquidation on the final bar:
     $$C_0 \prod_{k=1}^{K} \left(1 + \frac{R_{\text{net},k}}{100}\right) = E_{\text{net},T-1}$$
   - **Running Peak Anchored at Initial Capital**: Running peak equity is $P_t = \max\left(C_0, \max_{0 \le s \le t} E_s\right)$ so immediate post-entry losses are never understated in Maximum Drawdown.
   - **Zero-Exposure & Halted Holdout Transparency**: If a circuit breaker halts trading during Formation before $t_{\text{split}}$, Out-of-Sample metrics explicitly set `is_zero_exposure = True` and `halted_prior_to_period = True`, and Out-of-Sample Scorecard dimensions report `NO_EVIDENCE` rather than penalizing or rewarding zero exposure.

3. **5-Test Robustness Suite (`engine/stress_tests.py`)**
   - **Test 1** removes the largest winning trades to measure dependence on outliers.
   - **Test 2** delays fills by one more bar and stresses transaction costs.
   - **Test 3** compares results across volatility and market regimes.
   - **Test 4** compares the signal with 500 simulated random-entry trade sequences when sufficient data and trades are available.
   - **Test 5** checks structural stability for pairs or parameter sensitivity for a single-asset rule.

4. **Executable v4 Jupyter Notebook (`.ipynb`) & PDF Exports (`engine/notebook_ipynb.py`, `engine/notebook_pdf.py`)**
   - Exports a notebook with embedded aligned market data. Run it in a fresh Python kernel with the project dependencies to recompute the audit and check the results against the saved report.

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Workbench

Step 1 sends the user's hypothesis and earlier user inputs to Gemini for classification. Gemini selects one of the five built-in baskets, requests more detail, or returns no match. Alpha Auditor's local compiler fills the selected template. The user reviews the rule and confirms before running it. Backtests, literature notes, stress tests and reports run locally.

Set `GEMINI_API_KEY` in the server environment or the project's Git-ignored `.env` file. Set `ALPHA_AUDITOR_GEMINI_MODEL` to select an accessible model (default `gemini-3.5-flash`).

```bash
uvicorn app.server:app --reload --port 8000
```
Open `http://127.0.0.1:8000` in your browser.

### 3. Deterministic CLI Replay & Verification
```bash
# Verify deterministic two-pass repeatability
python replay.py --verify-repeatability

# Replay a saved audit artifact and verify fresh-kernel .ipynb execution
python replay.py data/runs/audit-022cbf20a8.json --ipynb /tmp/audit.ipynb --execute-ipynb
```

### 4. Run Test Suite
```bash
pytest -q
```
