"""Econometric and statistical inference methods grounded in financial literature:
- Lo (2002) & Mertens (2002): Non-IID Sharpe Ratio standard errors under skewness & kurtosis
- Bailey & López de Prado (2012, 2014): Probabilistic Sharpe Ratio (PSR) & Deflated Sharpe Ratio (DSR)
- Politis & Romano (1994): Stationary Bootstrap for time-dependent financial series
- Newey & West (1987): Heteroskedasticity and Autocorrelation Consistent (HAC) factor regression
- Engle & Granger (1987): Two-step cointegration & Augmented Dickey-Fuller (MacKinnon p-values)
- Vasicek (1977) / Ornstein-Uhlenbeck: Exact discrete-time AR(1) half-life of mean reversion
- Lo & MacKinlay (1988): Heteroskedasticity-consistent Variance Ratio test for random walk vs mean reversion
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from engine.schemas import (
    BootstrapConfidenceInterval,
    FactorRegressionResult,
    PairsDiagnostics,
)

EULER_MASCHERONI = 0.5772156649015329
TRADING_DAYS_PER_YEAR = 252.0


def lo_mertens_sharpe_se_and_psr(
    daily_returns: np.ndarray,
    benchmark_sr_annual: float = 0.0,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> tuple[float, float, float, float, float]:
    """Compute annualized Sharpe Ratio, Mertens (2002) / Lo (2002) non-normal standard error,
    Probabilistic Sharpe Ratio PSR(SR*), sample skewness, and sample Pearson kurtosis.

    Returns:
        (annual_sr, annual_sr_se, psr_pct, skewness, pearson_kurtosis)
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 5:
        return 0.0, 1.0, 50.0, 0.0, 3.0

    std = float(np.std(r, ddof=1))
    if std <= 1e-12:
        return 0.0, 1.0, 50.0, 0.0, 3.0

    mean = float(np.mean(r))
    sr_daily = mean / std
    sr_annual = sr_daily * math.sqrt(periods_per_year)
    sr_bench_daily = benchmark_sr_annual / math.sqrt(periods_per_year)

    skew = float(stats.skew(r, bias=False)) if n >= 8 else 0.0
    # Pearson kurtosis (Normal = 3.0)
    kurt_pearson = float(stats.kurtosis(r, fisher=False, bias=False)) if n >= 8 else 3.0
    kurt_pearson = max(1.0, min(kurt_pearson, 50.0))
    skew = max(-10.0, min(skew, 10.0))

    # Mertens (2002) / Bailey & López de Prado (2012) variance of non-annualized SR:
    # Var(SR_hat) = [1 - gamma_3 * SR_hat + ((gamma_4 - 1) / 4) * SR_hat^2] / (T - 1)
    numerator = 1.0 - skew * sr_daily + ((kurt_pearson - 1.0) / 4.0) * (sr_daily**2)
    numerator = max(1e-6, numerator)
    se_daily = math.sqrt(numerator / max(1, n - 1))
    se_annual = se_daily * math.sqrt(periods_per_year)

    z_stat = (sr_daily - sr_bench_daily) / se_daily
    psr = float(stats.norm.cdf(z_stat)) * 100.0
    return (
        round(sr_annual, 4),
        round(se_annual, 4),
        round(psr, 2),
        round(skew, 4),
        round(kurt_pearson, 4),
    )


def deflated_sharpe_ratio_bailey_lopez(
    daily_returns: np.ndarray,
    trial_annual_sharpes: list[float],
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> tuple[float, float]:
    """Compute Bailey & López de Prado (2014) Expected Maximum Sharpe Ratio under the null
    across K trials and the resulting Deflated Sharpe Ratio (DSR as a percentage 0..100).

    Returns:
        (expected_max_sr_annual, dsr_pct)
    """
    valid_srs = [float(x) for x in trial_annual_sharpes if np.isfinite(x)]
    k = len(valid_srs)
    if k <= 1:
        _, _, psr_pct, _, _ = lo_mertens_sharpe_se_and_psr(
            daily_returns, benchmark_sr_annual=0.0, periods_per_year=periods_per_year
        )
        return 0.0, psr_pct

    std_trials_annual = float(np.std(valid_srs, ddof=1))
    if std_trials_annual <= 1e-9:
        std_trials_annual = 0.5

    # Extreme Value Theory (False Strategy Theorem, Bailey & López de Prado 2014):
    # E[max_k SR_k] = sigma_{SR} * [(1 - gamma) * Z^{-1}(1 - 1/K) + gamma * Z^{-1}(1 - 1/(K*e))]
    q1 = float(stats.norm.ppf(1.0 - 1.0 / k))
    q2 = float(stats.norm.ppf(1.0 - 1.0 / (k * math.e)))
    expected_max_sr_annual = std_trials_annual * (
        (1.0 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2
    )
    expected_max_sr_annual = max(0.0, expected_max_sr_annual)

    _, _, dsr_pct, _, _ = lo_mertens_sharpe_se_and_psr(
        daily_returns,
        benchmark_sr_annual=expected_max_sr_annual,
        periods_per_year=periods_per_year,
    )
    return round(expected_max_sr_annual, 4), round(dsr_pct, 2)


def stationary_bootstrap_cis(
    daily_net_returns: np.ndarray,
    daily_bench_returns: np.ndarray,
    trade_net_returns_pct: list[float],
    replications: int = 1000,
    seed: int = 20261107,
) -> list[BootstrapConfidenceInterval]:
    """Politis & Romano (1994) Stationary Bootstrap with geometrically distributed block lengths
    preserving autocorrelation and volatility clustering (GARCH effects).
    """
    r = np.asarray(daily_net_returns, dtype=np.float64)
    b_ret = np.asarray(daily_bench_returns, dtype=np.float64)
    n = len(r)
    if n < 10:
        return []

    block_len = max(5, int(math.ceil(n ** (1.0 / 3.0))))
    p_new_block = 1.0 / block_len
    rng = np.random.default_rng(seed)

    # Vectorized generation of stationary bootstrap index paths (replications x n)
    starts = rng.integers(0, n, size=(replications, n))
    new_block_mask = rng.random(size=(replications, n)) < p_new_block
    new_block_mask[:, 0] = True

    indices = np.empty((replications, n), dtype=np.int64)
    indices[:, 0] = starts[:, 0]
    for t in range(1, n):
        indices[:, t] = np.where(new_block_mask[:, t], starts[:, t], (indices[:, t - 1] + 1) % n)

    boot_r = r[indices]
    boot_b = b_ret[indices]

    # 1. Annualized Mean Return (%)
    point_ann_ret = float(np.mean(r) * TRADING_DAYS_PER_YEAR * 100.0)
    boot_ann_ret = np.mean(boot_r, axis=1) * TRADING_DAYS_PER_YEAR * 100.0

    # 2. Annualized Sharpe Ratio
    std_r = float(np.std(r, ddof=1))
    point_sr = float((np.mean(r) / std_r) * math.sqrt(TRADING_DAYS_PER_YEAR)) if std_r > 1e-12 else 0.0
    boot_stds = np.std(boot_r, axis=1, ddof=1)
    boot_means = np.mean(boot_r, axis=1)
    boot_sr = np.where(
        boot_stds > 1e-12,
        (boot_means / boot_stds) * math.sqrt(TRADING_DAYS_PER_YEAR),
        0.0,
    )

    # 3. Annualized Excess Return vs Benchmark (%)
    diff_r = r - b_ret
    point_excess = float(np.mean(diff_r) * TRADING_DAYS_PER_YEAR * 100.0)
    boot_excess = np.mean(boot_r - boot_b, axis=1) * TRADING_DAYS_PER_YEAR * 100.0

    results = [
        BootstrapConfidenceInterval(
            metric_name="Annualized Return (Mean)",
            point_estimate=round(point_ann_ret, 2),
            ci_lower_95=round(float(np.percentile(boot_ann_ret, 2.5)), 2),
            ci_upper_95=round(float(np.percentile(boot_ann_ret, 97.5)), 2),
            bootstrap_std_error=round(float(np.std(boot_ann_ret, ddof=1)), 2),
            block_length=block_len,
            replications=replications,
            unit="%",
        ),
        BootstrapConfidenceInterval(
            metric_name="Annualized Sharpe Ratio",
            point_estimate=round(point_sr, 3),
            ci_lower_95=round(float(np.percentile(boot_sr, 2.5)), 3),
            ci_upper_95=round(float(np.percentile(boot_sr, 97.5)), 3),
            bootstrap_std_error=round(float(np.std(boot_sr, ddof=1)), 3),
            block_length=block_len,
            replications=replications,
            unit="ratio",
        ),
        BootstrapConfidenceInterval(
            metric_name="Annualized Excess Return vs. Benchmark",
            point_estimate=round(point_excess, 2),
            ci_lower_95=round(float(np.percentile(boot_excess, 2.5)), 2),
            ci_upper_95=round(float(np.percentile(boot_excess, 97.5)), 2),
            bootstrap_std_error=round(float(np.std(boot_excess, ddof=1)), 2),
            block_length=block_len,
            replications=replications,
            unit="%",
        ),
    ]

    # 4. Per-Trade Net Return Bootstrap (if >= 4 trades)
    if len(trade_net_returns_pct) >= 4:
        tr = np.asarray(trade_net_returns_pct, dtype=np.float64)
        k_tr = len(tr)
        tr_idx = rng.integers(0, k_tr, size=(replications, k_tr))
        boot_tr_means = np.mean(tr[tr_idx], axis=1)
        results.append(
            BootstrapConfidenceInterval(
                metric_name="Mean Net Return per Trade",
                point_estimate=round(float(np.mean(tr)), 3),
                ci_lower_95=round(float(np.percentile(boot_tr_means, 2.5)), 3),
                ci_upper_95=round(float(np.percentile(boot_tr_means, 97.5)), 3),
                bootstrap_std_error=round(float(np.std(boot_tr_means, ddof=1)), 3),
                block_length=1,
                replications=replications,
                unit="%",
            )
        )

    return results


def newey_west_factor_regression(
    daily_net_returns: np.ndarray,
    daily_bench_returns: np.ndarray,
) -> FactorRegressionResult:
    """Newey & West (1987) HAC OLS regression r_{p,t} = alpha + beta * r_{b,t} + eps_t."""
    y = np.asarray(daily_net_returns, dtype=np.float64)
    x = np.asarray(daily_bench_returns, dtype=np.float64)
    n = len(y)
    if n < 15 or np.std(y) <= 1e-12 or np.std(x) <= 1e-12:
        return FactorRegressionResult(
            annualized_alpha_pct=0.0,
            alpha_t_stat_hac=0.0,
            alpha_p_value_hac=1.0,
            beta=0.0,
            beta_t_stat_hac=0.0,
            r_squared=0.0,
            information_ratio=0.0,
            newey_west_lags=1,
        )

    # Newey-West (1994) plug-in lag selection rule: floor(4 * (T / 100)^(2/9))
    max_lags = max(1, int(math.floor(4.0 * ((n / 100.0) ** (2.0 / 9.0)))))
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})

    alpha_daily = float(model.params[0])
    beta = float(model.params[1])
    alpha_t = float(model.tvalues[0]) if np.isfinite(model.tvalues[0]) else 0.0
    alpha_p = float(model.pvalues[0]) if np.isfinite(model.pvalues[0]) else 1.0
    beta_t = float(model.tvalues[1]) if np.isfinite(model.tvalues[1]) else 0.0
    r2 = float(max(0.0, model.rsquared)) if np.isfinite(model.rsquared) else 0.0

    active_ret = y - x
    active_std = float(np.std(active_ret, ddof=1))
    ir = (
        float((np.mean(active_ret) / active_std) * math.sqrt(TRADING_DAYS_PER_YEAR))
        if active_std > 1e-12
        else 0.0
    )

    return FactorRegressionResult(
        annualized_alpha_pct=round(alpha_daily * TRADING_DAYS_PER_YEAR * 100.0, 2),
        alpha_t_stat_hac=round(alpha_t, 3),
        alpha_p_value_hac=round(alpha_p, 4),
        beta=round(beta, 3),
        beta_t_stat_hac=round(beta_t, 3),
        r_squared=round(r2, 4),
        information_ratio=round(ir, 3),
        newey_west_lags=max_lags,
    )


def lo_mackinlay_variance_ratio(series: np.ndarray, q: int = 5) -> tuple[float, float]:
    """Lo & MacKinlay (1988) heteroskedasticity-consistent Variance Ratio test VR(q)
    on a level/spread series S_t.
    VR(q) < 1 indicates mean-reverting negative serial correlation;
    VR(q) = 1 under a martingale/random walk.
    Returns (vr_q, two_sided_p_value).
    """
    s = np.asarray(series, dtype=np.float64)
    n = len(s)
    if n < 4 * q + 5:
        return 1.0, 1.0

    dy = np.diff(s)
    nq = len(dy)
    mu = float(np.mean(dy))
    var_1 = float(np.sum((dy - mu) ** 2)) / (nq - 1)
    if var_1 <= 1e-15:
        return 1.0, 1.0

    dy_q = s[q:] - s[:-q]
    m = q * (nq - q + 1) * (1.0 - q / nq)
    var_q = float(np.sum((dy_q - q * mu) ** 2)) / m
    vr = var_q / var_1

    # Heteroskedasticity-robust asymptotic variance theta(q) (Lo & MacKinlay 1988, eq. 19)
    centered_sq = (dy - mu) ** 2
    denom = (float(np.sum(centered_sq))) ** 2
    theta = 0.0
    for j in range(1, q):
        delta_j = float(np.sum(centered_sq[j:] * centered_sq[:-j])) / denom
        weight = (2.0 * (q - j) / q) ** 2
        theta += weight * delta_j

    z_star = (vr - 1.0) / math.sqrt(max(1e-12, theta))
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_star))))
    return round(float(vr), 4), round(p_val, 4)


def generalized_hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """Estimate the Hurst exponent H via variogram scaling Var(S_{t+tau} - S_t) ~ tau^{2H}.
    H < 0.5 indicates mean reversion; H = 0.5 Brownian motion; H > 0.5 trend persistence.
    """
    s = np.asarray(series, dtype=np.float64)
    if len(s) < max_lag * 2:
        return 0.5
    lags = np.arange(2, max_lag + 1)
    tau_vars = []
    for lag in lags:
        diffs = s[lag:] - s[:-lag]
        v = float(np.var(diffs, ddof=1))
        tau_vars.append(max(v, 1e-15))
    poly = np.polyfit(np.log(lags), np.log(tau_vars), 1)
    h = float(poly[0] / 2.0)
    return round(max(0.01, min(0.99, h)), 3)


def compute_pairs_econometrics(
    dates_form: list[str],
    close_a_form: np.ndarray,
    close_b_form: np.ndarray,
) -> tuple[PairsDiagnostics, float, float]:
    """Estimate Formation-period OLS cointegration relationship:
        ln(P_{A,t}) = alpha + beta * ln(P_{B,t}) + S_t
    and run Engle-Granger, ADF, Ornstein-Uhlenbeck half-life, Hurst exponent, and
    Lo-MacKinlay Variance Ratio diagnostics strictly on the formation split.
    """
    log_a = np.log(np.asarray(close_a_form, dtype=np.float64))
    log_b = np.log(np.asarray(close_b_form, dtype=np.float64))

    X = sm.add_constant(log_b)
    ols_res = sm.OLS(log_a, X).fit()
    alpha = float(ols_res.params[0])
    beta = float(ols_res.params[1])
    r2 = float(max(0.0, ols_res.rsquared)) if np.isfinite(ols_res.rsquared) else 0.0

    spread_form = log_a - (alpha + beta * log_b)

    # Return correlation & price-level correlation
    ret_a = np.diff(log_a)
    ret_b = np.diff(log_b)
    ret_corr = float(np.corrcoef(ret_a, ret_b)[0, 1]) if len(ret_a) > 2 else 0.0
    lvl_corr = float(np.corrcoef(close_a_form, close_b_form)[0, 1]) if len(close_a_form) > 2 else 0.0

    # Augmented Dickey-Fuller test on the formation spread S_t
    adf_res = adfuller(spread_form, autolag="AIC")
    adf_stat = float(adf_res[0])
    adf_p = float(adf_res[1])
    adf_crit = {k: round(float(v), 3) for k, v in adf_res[4].items()}

    # Engle-Granger two-step cointegration test (MacKinnon critical values accounting for estimated beta)
    try:
        eg_stat, eg_p, _ = coint(log_a, log_b, trend="c", autolag="AIC")
        eg_stat_f = float(eg_stat)
        eg_p_f = float(eg_p)
    except Exception:
        eg_stat_f, eg_p_f = adf_stat, adf_p

    # Ornstein-Uhlenbeck discrete AR(1): dS_t = c + lambda * S_{t-1} + eta_t
    s_lag = spread_form[:-1]
    ds = np.diff(spread_form)
    ou_ols = sm.OLS(ds, sm.add_constant(s_lag)).fit()
    ou_lambda = float(ou_ols.params[1])
    if -0.999 < ou_lambda < -1e-6:
        half_life: float | None = round(-math.log(2.0) / math.log(1.0 + ou_lambda), 1)
    else:
        half_life = None

    hurst = generalized_hurst_exponent(spread_form)
    vr_q5, vr_p = lo_mackinlay_variance_ratio(spread_form, q=5)

    if eg_p_f < 0.05:
        status_note = (
            f"Formation spread rejects a unit root under the Engle-Granger test at the 5% level "
            f"(p = {eg_p_f:.4f}, ADF p = {adf_p:.4f}, ADF t = {adf_stat:.2f}, return correlation = {ret_corr:.2f})."
        )
    elif adf_p < 0.05:
        status_note = (
            f"Formation residual spread rejects a univariate unit root at the 5% level "
            f"(ADF p = {adf_p:.4f}, t = {adf_stat:.2f}; two-step Engle-Granger p = {eg_p_f:.4f}, return correlation = {ret_corr:.2f}). "
            "While the formation spread mean-reverts in-sample, out-of-sample structural drift remains possible."
        )
    elif eg_p_f < 0.10 or adf_p < 0.10:
        status_note = (
            f"Formation spread shows marginal stationarity evidence at the 10% level "
            f"(Engle-Granger p = {eg_p_f:.4f}, ADF p = {adf_p:.4f}, ADF t = {adf_stat:.2f}, return correlation = {ret_corr:.2f})."
        )
    else:
        status_note = (
            f"Formation spread does NOT reject the unit-root null at standard significance levels "
            f"(Engle-Granger p = {eg_p_f:.4f}, ADF p = {adf_p:.4f}, return correlation = {ret_corr:.2f}). "
            "High return correlation does not establish cointegration; out-of-sample divergence risk is elevated."
        )

    diag = PairsDiagnostics(
        formation_start=str(dates_form[0]),
        formation_end=str(dates_form[-1]),
        ols_alpha=round(alpha, 5),
        ols_hedge_ratio_beta=round(beta, 4),
        ols_r_squared=round(r2, 4),
        return_correlation=round(ret_corr, 4),
        level_correlation=round(lvl_corr, 4),
        adf_statistic=round(adf_stat, 3),
        adf_p_value=round(adf_p, 4),
        adf_critical_values=adf_crit,
        engle_granger_stat=round(eg_stat_f, 3),
        engle_granger_p_value=round(eg_p_f, 4),
        ou_lambda=round(ou_lambda, 5),
        half_life_days=half_life,
        hurst_exponent=hurst,
        variance_ratio_q5=vr_q5,
        variance_ratio_p_value=vr_p,
        cointegration_status_note=status_note,
    )
    return diag, alpha, beta


def exact_sign_test_p_value(wins: int, losses: int) -> float | None:
    """Two-sided exact binomial sign test of wins vs losses at p = 0.5."""
    n = wins + losses
    if n <= 0:
        return None
    val = float(stats.binomtest(wins, n, p=0.5, alternative="two-sided").pvalue)
    return round(val, 4)
