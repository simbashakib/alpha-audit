/**
 * Alpha Auditor — 6-Step Quantitative Hypothesis & Econometric Workbench
 */

const state = {
  catalog: null,
  currentSpec: null,
  initialParsedSpec: null,
  currentReport: null,
  baselineReport: null,
  thresholdMode: 'user_specified',
  lastSteeredCards: [],
  chatHistory: [],
  holdoutTouchesByAssetPair: {},
  activeStressIdx: 0,
};

function escapeHtml(raw) {
  if (raw === null || raw === undefined) return '';
  return String(raw)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatMathHtml(raw) {
  if (!raw) return '';
  let s = escapeHtml(raw);
  s = s
    .replace(/&lt;=/g, '≤')
    .replace(/&gt;=/g, '≥')
    .replace(/!=/g, '≠')
    .replace(/\+-/g, '±')
    .replace(/\\times/g, '×')
    .replace(/\\sqrt\{([^}]+)\}/g, '√($1)')
    .replace(/√252/g, '√252')
    .replace(/\\ln/g, 'ln')
    .replace(/\\Phi/g, 'Φ')
    .replace(/\\alpha/g, 'α')
    .replace(/\\beta/g, 'β')
    .replace(/\\gamma/g, 'γ')
    .replace(/\\lambda/g, 'λ')
    .replace(/\\sigma/g, 'σ')
    .replace(/\\theta/g, 'θ')
    .replace(/\\mu/g, 'μ')
    .replace(/\\rho/g, 'ρ')
    .replace(/\\tau/g, 'τ')
    .replace(/([A-Za-zα-ωΑ-ΩΦ]+)_\{([^}]+)\}/g, '$1<sub>$2</sub>')
    .replace(/([A-Za-zα-ωΑ-ΩΦ]+)_([0-9a-zA-Z,+\-]+)/g, '$1<sub>$2</sub>')
    .replace(/\^\{([^}]+)\}/g, '<sup>$1</sup>')
    .replace(/\^([0-9]+)/g, '<sup>$1</sup>');
  return s;
}

function safePlotlyNewPlot(containerId, traces, layout, config) {
  const el = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
  if (!el) return;
  if (window.Plotly && typeof window.Plotly.newPlot === 'function') {
    try {
      const defaultHeight = el.classList.contains('chart-box-sm') ? 310 : 360;
      const mergedLayout = {
        autosize: true,
        height: (layout && layout.height) || defaultHeight,
        ...layout,
      };
      window.Plotly.newPlot(el, traces, mergedLayout, config);
      return;
    } catch (err) {
      console.warn(`Plotly render warning on ${containerId}:`, err);
    }
  }
  el.innerHTML = `
    <div style="padding:18px; border:1px dashed var(--border-subtle); border-radius:6px; font-family:var(--font-mono); font-size:12px; color:var(--text-secondary);">
      Interactive chart canvas unavailable offline — consult the Accessible Numeric Summary Table and KPI Strip for exact values.
    </div>
  `;
}

function numOr(raw, fallback) {
  if (raw === '' || raw === null || raw === undefined) return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function intOr(raw, fallback) {
  if (raw === '' || raw === null || raw === undefined) return fallback;
  const parsed = parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function getAssetPairKey(spec) {
  const a = (spec.asset_a || 'SPY').toUpperCase();
  const b = spec.strategy_family === 'pairs_trading' && spec.asset_b ? `:${spec.asset_b.toUpperCase()}` : '';
  return `${a}${b}`;
}

function isDarkMode() {
  return document.documentElement.classList.contains('dark');
}

function getPlotlyLayout() {
  const dark = isDarkMode();
  return {
    autosize: true,
    paper_bgcolor: dark ? '#141210' : '#ffffff',
    plot_bgcolor: dark ? '#0c0a09' : '#fafaf9',
    font: {
      family: 'JetBrains Mono, monospace',
      color: dark ? '#a8a29e' : '#57534e',
      size: 11,
    },
    margin: { t: 28, r: 20, b: 38, l: 56 },
    xaxis: {
      gridcolor: dark ? '#292524' : '#e7e5e4',
      zerolinecolor: dark ? '#44403c' : '#d6d3d1',
      linecolor: dark ? '#292524' : '#e7e5e4',
    },
    yaxis: {
      gridcolor: dark ? '#292524' : '#e7e5e4',
      zerolinecolor: dark ? '#44403c' : '#d6d3d1',
      linecolor: dark ? '#292524' : '#e7e5e4',
    },
    legend: {
      orientation: 'h',
      yanchor: 'bottom',
      y: 1.02,
      xanchor: 'left',
      x: 0,
      bgcolor: dark ? 'rgba(20,18,16,0.82)' : 'rgba(255,255,255,0.88)',
    },
  };
}

function applyTheme(theme) {
  const dark = theme === 'dark';
  document.documentElement.classList.toggle('dark', dark);
  const labelEl = document.getElementById('theme-toggle-label');
  if (labelEl) labelEl.textContent = dark ? 'Light mode' : 'Dark mode';
  try {
    localStorage.setItem('alpha_auditor_theme', dark ? 'dark' : 'light');
  } catch (_) {}
  if (state.currentReport) {
    renderEquityAndDrawdownCharts(state.currentReport);
    renderSignalTrajectoryChart(state.currentReport);
    renderEconometricsTab(state.currentReport);
    renderStressTestsStep(state.currentReport);
  }
}

function initThemeToggle() {
  let saved = 'light';
  try {
    saved = localStorage.getItem('alpha_auditor_theme') || 'light';
  } catch (_) {}
  applyTheme(saved);

  const btn = document.getElementById('btn-theme-toggle');
  if (btn) {
    btn.addEventListener('click', () => {
      applyTheme(isDarkMode() ? 'light' : 'dark');
    });
  }
}

const PLOTLY_CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
};

function fmtPct(val, digits = 2, showPlus = true) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return '—';
  const n = Number(val);
  const sign = showPlus && n > 0 ? '+' : '';
  return `${sign}${n.toFixed(digits)}%`;
}

function fmtPp(val, digits = 2) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return '—';
  const n = Number(val);
  const sign = n > 0 ? '+' : '';
  return `${sign}${n.toFixed(digits)} pp`;
}

function fmtNum(val, digits = 2) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return '—';
  return Number(val).toFixed(digits);
}

function colorClass(val) {
  const n = Number(val);
  if (Number.isNaN(n) || n === 0) return '';
  return n > 0 ? 'val-pos' : 'val-neg';
}

function returnColorClassVsBenchmark(retPct, benchPct) {
  const r = Number(retPct);
  const b = Number(benchPct);
  if (Number.isNaN(r) || r === 0) return '';
  if (r < 0) return 'val-neg';
  if (!Number.isNaN(b) && r < b) return '';
  return 'val-pos';
}

function resizeVisibleCharts() {
  if (!window.Plotly) return;
  document.querySelectorAll('.view-panel.active .chart-box, .view-panel.active .chart-box-sm').forEach((el) => {
    if (el.offsetParent !== null && el.data) {
      try {
        window.Plotly.Plots.resize(el);
      } catch (_) {}
    }
  });
}

function switchTab(targetId) {
  document.querySelectorAll('.nav-tab').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.target === targetId);
  });
  document.querySelectorAll('.view-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === targetId);
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
  resizeVisibleCharts();
  setTimeout(() => {
    window.dispatchEvent(new Event('resize'));
    resizeVisibleCharts();
  }, 60);
}

function switchSubTab(groupName, subtargetId) {
  const resolvedTarget = subtargetId || groupName;
  const targetEl = document.getElementById(resolvedTarget);
  const resolvedGroup =
    targetEl?.dataset?.group || groupName;
  document
    .querySelectorAll(`.subtab-bar[data-group="${resolvedGroup}"] .subtab-btn`)
    .forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.subtarget === resolvedTarget);
    });
  document
    .querySelectorAll(`.subtab-panel[data-group="${resolvedGroup}"]`)
    .forEach((panel) => {
      panel.classList.toggle('active', panel.id === resolvedTarget);
    });
  resizeVisibleCharts();
  setTimeout(() => {
    window.dispatchEvent(new Event('resize'));
    resizeVisibleCharts();
  }, 60);
}

function renderPreAuditEmptyStates() {
  const emptyChartNotice = (label) => `
    <div style="height:100%; display:flex; align-items:center; justify-content:center; padding:24px; border:1px dashed var(--border-subtle); border-radius:var(--radius-md); font-family:var(--font-sans); font-size:13px; color:var(--text-muted); text-align:center;">
      <div>
        <strong style="color:var(--text-secondary); display:block; margin-bottom:6px;">${escapeHtml(label)}</strong>
        Confirm the specification and click <strong>Confirm &amp; Run Audit</strong> in Step 2 (Configuration) to compute historical performance and econometric diagnostics.
      </div>
    </div>
  `;

  const kpiStrip = document.getElementById('kpi-strip-container');
  if (kpiStrip && !kpiStrip.innerHTML.trim()) {
    kpiStrip.innerHTML = `
      <div class="kpi-card" style="grid-column: 1 / -1; text-align:center; padding:18px 24px; color:var(--text-secondary); font-family:var(--font-sans); font-size:13.5px;">
        No audit has been executed yet. Configure your hypothesis in <strong>1. Hypothesis</strong> and click <strong>Confirm &amp; Run Audit</strong> in <strong>2. Configuration</strong> to populate Out-of-Sample KPIs, charts, and econometric tests.
      </div>
    `;
  }

  for (const [id, label] of [
    ['chart-equity', 'Cumulative Portfolio Equity Curve'],
    ['chart-drawdown', 'Underwater Drawdown Trajectory'],
    ['chart-signal', 'Signal Trajectory & Triggered Trades'],
    ['chart-sensitivity-heatmap', '25-Trial Parameter Sensitivity Surface'],
    ['chart-cost-sensitivity', 'Transaction Cost Sensitivity Curve'],
  ]) {
    const el = document.getElementById(id);
    if (el && !el.innerHTML.trim()) {
      el.innerHTML = emptyChartNotice(label);
    }
  }

  for (const [sel, cols] of [
    ['#bracket-comparison-table tbody', 9],
    ['#split-comparison-table tbody', 6],
    ['#sensitivity-grid-table tbody', 8],
    ['#bootstrap-table tbody', 6],
    ['#factor-reg-table tbody', 4],
    ['#subperiod-table tbody', 7],
    ['#trades-blotter-table tbody', 16],
  ]) {
    const tbody = document.querySelector(sel);
    if (tbody && !tbody.innerHTML.trim()) {
      tbody.innerHTML = `<tr><td colspan="${cols}" style="text-align:center; color:var(--text-muted); padding:18px; font-family:var(--font-sans);">Run an audit in Step 2 (Configuration) to populate this table.</td></tr>`;
    }
  }

  const execSum = document.getElementById('scorecard-exec-summary');
  if (execSum && !execSum.textContent.trim()) {
    execSum.textContent = 'Run an audit in Step 2 (Configuration) to generate the empirical hypothesis verdict, 6-dimension scorecard, and step-by-step calculation ledger.';
  }
  const clarityStrip = document.getElementById('step6-clarity-strip');
  if (clarityStrip && !clarityStrip.innerHTML.trim()) {
    clarityStrip.innerHTML = `
      <div class="clarity-card" style="grid-column: 1 / -1; text-align:center; color:var(--text-secondary); font-size:13px;">
        Awaiting historical audit execution in Step 2 (Configuration).
      </div>
    `;
  }
  const dimGrid = document.getElementById('scorecard-dimensions-grid');
  if (dimGrid && !dimGrid.innerHTML.trim()) {
    dimGrid.innerHTML = `
      <div class="dimension-card" style="grid-column: 1 / -1; text-align:center; color:var(--text-muted); font-size:13px;">
        Empirical verification checks will appear here once an audit is executed in Step 2.
      </div>
    `;
  }
  const calcSteps = document.getElementById('calculation-steps-container');
  if (calcSteps && !calcSteps.innerHTML.trim()) {
    calcSteps.innerHTML = `
      <div class="dimension-card" style="text-align:center; color:var(--text-muted); font-size:13px;">
        Step-by-step mathematical derivations and substituted inputs will appear here after running an audit in Step 2 (Configuration).
      </div>
    `;
  }
  const nbPreview = document.getElementById('jupyter-notebook-preview');
  if (nbPreview && !nbPreview.innerHTML.trim()) {
    nbPreview.innerHTML = `
      <div class="dimension-card" style="text-align:center; color:var(--text-muted); font-size:13px;">
        Confirm &amp; Run Audit in Step 2 (Configuration) to generate the executable Jupyter Calculation Notebook preview.
      </div>
    `;
  }
  const provGrid = document.getElementById('provenance-grid');
  if (provGrid && !provGrid.innerHTML.trim()) {
    provGrid.innerHTML = `
      <div class="kpi-card" style="grid-column: 1 / -1; text-align:center; color:var(--text-muted); font-size:13px;">
        Cryptographic SHA-256 dataset and specification provenance will appear here once an audit is executed in Step 2.
      </div>
    `;
  }
}

async function initWorkbench() {
  initThemeToggle();

  document.querySelectorAll('.nav-tab').forEach((btn) => {
    btn.addEventListener('click', () => switchTab(btn.dataset.target));
  });

  document.querySelectorAll('.subtab-bar[data-group]').forEach((bar) => {
    const group = bar.dataset.group;
    bar.querySelectorAll('.subtab-btn').forEach((btn) => {
      btn.addEventListener('click', () => switchSubTab(group, btn.dataset.subtarget));
    });
  });

  const res = await fetch('/api/catalog');
  state.catalog = await res.json();

  populateCatalogUI();
  bindSpecFormEvents();
  state.initialParsedSpec = readSpecFromForm();
  state.currentSpec = JSON.parse(JSON.stringify(state.initialParsedSpec));
  syncStep2SummaryFromForm();
  renderPreAuditEmptyStates();
  loadChallengeEvidence();

  // Optional URL query hooks for reproducible browser inspection (?theme=dark&q=...&autorun=1&step=4)
  try {
    const params = new URLSearchParams(window.location.search);
    const themeParam = params.get('theme');
    if (themeParam === 'dark' || themeParam === 'light') {
      applyTheme(themeParam);
    }
    const qParam = params.get('q');
    if (qParam) {
      const inputEl = document.getElementById('hypothesis-input');
      if (inputEl) inputEl.value = qParam;
      const compiled = await compileHypothesis(qParam, true);
      if (compiled && params.get('autorun') === '1') {
        const chk = document.getElementById('chk-human-confirm');
        if (chk) chk.checked = true;
        await executeAuditFromForm();
      }
    }
    const stepParam = params.get('step');
    if (stepParam) {
      switchTab(`view-step-${stepParam}`);
    }
    const subtabParam = params.get('subtab');
    if (subtabParam && subtabParam.includes(':')) {
      const [grp, subId] = subtabParam.split(':');
      switchSubTab(grp, subId);
    }
  } catch (_) {}
}

function populateCatalogUI() {
  const { tickers, presets } = state.catalog;

  const dl = document.getElementById('catalog-tickers-list');
  if (dl) {
    dl.innerHTML = (tickers || [])
      .map((t) => `<option value="${t.ticker}">${t.name} (${t.category})</option>`)
      .join('');
  }

  const presetBox = document.getElementById('preset-container');
  if (presetBox && Array.isArray(presets) && presets.length > 0) {
    presetBox.innerHTML = `
      <span class="preset-bar-label">Example Hypotheses:</span>
      ${presets
        .map(
          (p, idx) => `
          <button type="button" class="preset-chip" data-preset-idx="${idx}" title="${escapeHtml(p.text)}">
            ${escapeHtml(p.label)}
          </button>
        `
        )
        .join('')}
    `;
    presetBox.querySelectorAll('.preset-chip').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const idx = parseInt(btn.dataset.presetIdx, 10);
        const preset = presets[idx];
        if (!preset) return;
        const inputEl = document.getElementById('hypothesis-input');
        if (inputEl) inputEl.value = preset.text;
        await compileHypothesis(preset.text, true);
      });
    });
  }

  updateTemplateDropdown('single_asset', 'return_threshold', 'cum_return_drop');
}

function updateTemplateDropdown(family, selectedTemplateId, selectedVariantId) {
  const templates = state.catalog.templates.filter((t) => t.family === family);
  const tplSelect = document.getElementById('spec-template');
  tplSelect.innerHTML = templates
    .map((t) => `<option value="${t.template_id}">${t.name}</option>`)
    .join('');

  const targetTpl =
    templates.find((t) => t.template_id === selectedTemplateId) || templates[0];
  if (targetTpl) {
    tplSelect.value = targetTpl.template_id;
    updateVariantDropdown(targetTpl.template_id, selectedVariantId);
  }

  const groupB = document.getElementById('group-asset-b');
  groupB.style.display = family === 'pairs_trading' ? 'flex' : 'none';
}

function updateThresholdFieldLabels(templateId, variantId) {
  const lblEntry = document.getElementById('lbl-entry-thresh');
  const lblExit = document.getElementById('lbl-exit-thresh');
  const lblLookback = document.getElementById('lbl-lookback');
  const lblSecWin = document.getElementById('lbl-secondary-win');
  if (!lblEntry || !lblExit || !lblLookback || !lblSecWin) return;

  if (templateId === 'return_threshold') {
    lblEntry.textContent =
      variantId === 'vol_scaled_shock'
        ? 'Shock Threshold (σ Units)'
        : 'Return Threshold (% Drop / Surge)';
    lblExit.textContent = 'Early Exit Target (% / 0 = Hold Only)';
    lblLookback.textContent = 'Return Measurement Window (Days)';
    lblSecWin.textContent = 'Volatility Reference Window (Days)';
  } else if (templateId === 'consecutive_streak') {
    lblEntry.textContent = 'Streak Trigger Threshold (Days)';
    lblExit.textContent = 'Exit Threshold (0 = Fixed Hold)';
    lblLookback.textContent = 'Reference Window (Days)';
    lblSecWin.textContent = 'Secondary Window (Days)';
  } else if (templateId === 'mean_reversion_oscillator') {
    lblEntry.textContent =
      variantId === 'rsi_extremes'
        ? 'RSI Oversold Entry Level (0–50)'
        : 'Entry Z-Score / Band Width (σ)';
    lblExit.textContent =
      variantId === 'rsi_extremes'
        ? 'RSI Mean-Reversion Exit Level (e.g. 50)'
        : 'Exit Z-Score Target (σ)';
    lblLookback.textContent =
      variantId === 'rsi_extremes'
        ? 'Wilder RSI Window (Days)'
        : 'Rolling Mean / Bollinger Window (Days)';
    lblSecWin.textContent = 'Secondary Reference Window (Days)';
  } else if (templateId === 'trend_breakout') {
    lblEntry.textContent = 'Breakout / MA Spread Buffer (%)';
    lblExit.textContent = 'Exit Buffer (%)';
    lblLookback.textContent =
      variantId === 'donchian_breakout'
        ? 'Donchian Entry Channel (Days)'
        : 'Fast Moving Average Window (Days)';
    lblSecWin.textContent =
      variantId === 'donchian_breakout'
        ? 'Donchian Exit Channel (Days)'
        : 'Slow Moving Average Window (Days)';
  } else if (templateId === 'pairs_cointegration_reversion') {
    lblEntry.textContent = 'Spread Entry Z-Score (|Z| ≥ k σ)';
    lblExit.textContent = 'Spread Mean-Reversion Exit (|Z| ≤ exit σ)';
    lblLookback.textContent = 'Rolling Spread Z-Score Window (Days)';
    lblSecWin.textContent = 'Cointegration Formation Window (Days)';
  }
}

function updateVariantDropdown(templateId, selectedVariantId) {
  const tpl = state.catalog.templates.find((t) => t.template_id === templateId);
  if (!tpl) return;
  const varSelect = document.getElementById('spec-variant');
  varSelect.innerHTML = tpl.variants
    .map((v) => `<option value="${v.id}">${v.label}</option>`)
    .join('');

  const targetVar =
    tpl.variants.find((v) => v.id === selectedVariantId) || tpl.variants[0];
  if (targetVar) {
    varSelect.value = targetVar.id;
    updateFormulaBanner(tpl, targetVar);
    updateThresholdFieldLabels(tpl.template_id, targetVar.id);
  }

  const liveVarSelect = document.getElementById('live-variant');
  if (liveVarSelect) {
    liveVarSelect.innerHTML = varSelect.innerHTML;
    liveVarSelect.value = varSelect.value;
  }
}

function updateFormulaBanner(tpl, variantObj) {
  const banner = document.getElementById('active-formula-banner');
  if (!banner) return;
  banner.innerHTML = `<strong>${escapeHtml(tpl.name)}:</strong> <code>${formatMathHtml(variantObj.formula)}</code> &nbsp;|&nbsp; <span style="color:var(--text-secondary);">Ref: ${escapeHtml(tpl.literature_ref)}</span>`;
}

function setThresholdMode(mode) {
  state.thresholdMode = mode;
  document.getElementById('mode-card-specific').classList.toggle('active', mode === 'user_specified');
  document.getElementById('mode-card-bracket').classList.toggle('active', mode === 'auto_standard_bracket');
  const chkBrackets = document.getElementById('chk-show-brackets');
  if (chkBrackets) {
    chkBrackets.checked = mode === 'auto_standard_bracket';
  }
  resetConfirmationGate();
}

function bindSpecFormEvents() {
  const sendHypothesis = async () => {
    const inputEl = document.getElementById('hypothesis-input');
    const txt = inputEl.value.trim();
    if (!txt) return;
    await compileHypothesis(txt, true);
  };

  document.getElementById('btn-compile-hypothesis').addEventListener('click', sendHypothesis);

  const hypInput = document.getElementById('hypothesis-input');
  hypInput.addEventListener('keydown', async (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      await sendHypothesis();
    }
  });

  document.getElementById('btn-goto-step-2').addEventListener('click', () => {
    switchTab('view-step-2');
  });

  const btnPdf = document.getElementById('btn-download-notebook-pdf');
  if (btnPdf) {
    btnPdf.addEventListener('click', () => downloadNotebookPDF());
  }

  const btnIpynb = document.getElementById('btn-download-notebook-ipynb');
  if (btnIpynb) {
    btnIpynb.addEventListener('click', () => downloadNotebookIpynb());
  }

  const chkBrackets = document.getElementById('chk-show-brackets');
  if (chkBrackets) {
    chkBrackets.addEventListener('change', () => {
      if (state.currentReport) {
        renderEquityAndDrawdownCharts(state.currentReport);
      }
    });
  }

  document.getElementById('mode-card-specific').addEventListener('click', () => {
    setThresholdMode('user_specified');
  });
  document.getElementById('mode-card-bracket').addEventListener('click', () => {
    setThresholdMode('auto_standard_bracket');
  });

  const famSelect = document.getElementById('spec-family');
  famSelect.addEventListener('change', () => {
    const fam = famSelect.value;
    updateTemplateDropdown(
      fam,
      fam === 'pairs_trading' ? 'pairs_cointegration_reversion' : 'return_threshold'
    );
    resetConfirmationGate();
  });

  const tplSelect = document.getElementById('spec-template');
  tplSelect.addEventListener('change', () => {
    updateVariantDropdown(tplSelect.value);
    resetConfirmationGate();
  });

  const varSelect = document.getElementById('spec-variant');
  varSelect.addEventListener('change', () => {
    const tpl = state.catalog.templates.find((t) => t.template_id === tplSelect.value);
    const vObj = tpl?.variants.find((v) => v.id === varSelect.value);
    if (tpl && vObj) {
      updateFormulaBanner(tpl, vObj);
      updateThresholdFieldLabels(tpl.template_id, vObj.id);
    }
    resetConfirmationGate();
  });

  const streakInput = document.getElementById('spec-streak');
  const entryInput = document.getElementById('spec-entry-thresh');
  if (streakInput && entryInput) {
    streakInput.addEventListener('change', () => {
      if (tplSelect.value === 'consecutive_streak') {
        entryInput.value = streakInput.value;
      }
      resetConfirmationGate();
    });
    entryInput.addEventListener('change', () => {
      if (tplSelect.value === 'consecutive_streak') {
        streakInput.value = String(Math.max(1, Math.round(numOr(entryInput.value, 3))));
      }
      resetConfirmationGate();
    });
  }

  document.getElementById('btn-validate-a').addEventListener('click', () =>
    validateTickerInput('spec-asset-a', 'status-asset-a')
  );
  document.getElementById('btn-validate-b').addEventListener('click', () =>
    validateTickerInput('spec-asset-b', 'status-asset-b')
  );
  document.getElementById('btn-validate-bench').addEventListener('click', () =>
    validateTickerInput('spec-benchmark', 'status-benchmark')
  );

  const formInputs = document.querySelectorAll('#view-step-2 input:not(#chk-human-confirm), #view-step-2 select');
  formInputs.forEach((el) => {
    el.addEventListener('change', () => resetConfirmationGate());
  });

  const chkConfirm = document.getElementById('chk-human-confirm');
  const btnRun = document.getElementById('btn-run-audit');
  chkConfirm.addEventListener('change', () => {
    btnRun.disabled = !chkConfirm.checked;
  });

  btnRun.addEventListener('click', async () => {
    await executeAuditFromForm();
  });

  document.getElementById('btn-recompute-live').addEventListener('click', async () => {
    await executeLiveExplorerRecompute();
  });

  for (const [inId, lblId] of [
    ['live-entry-thresh', 'val-live-entry'],
    ['live-exit-thresh', 'val-live-exit'],
    ['live-lookback', 'val-live-lookback'],
    ['live-holding', 'val-live-hold'],
    ['live-cost-bps', 'val-live-cost'],
  ]) {
    const inEl = document.getElementById(inId);
    const lblEl = document.getElementById(lblId);
    if (inEl && lblEl) {
      inEl.addEventListener('input', () => {
        lblEl.textContent = inEl.value;
      });
    }
  }

  document.getElementById('filter-trade-split').addEventListener('change', () => {
    if (state.currentReport) renderTradesBlotter(state.currentReport.trades);
  });

  document.getElementById('btn-export-json').addEventListener('click', () => {
    if (!state.currentReport) return;
    const blob = new Blob([JSON.stringify(state.currentReport, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${state.currentReport.run_id}-reproducible-audit.json`;
    a.click();
  });

  document.getElementById('btn-export-trades-csv').addEventListener('click', () => {
    if (!state.currentReport) return;
    exportTradesToCSV(state.currentReport.trades, state.currentReport.run_id);
  });
}

function updateModifiedSinceLastRunBanner() {
  const banner = document.getElementById('spec-modified-banner');
  if (!banner) return;
  const refSpec = state.baselineReport?.spec || state.initialParsedSpec || state.currentSpec;
  if (!refSpec) {
    banner.style.display = 'none';
    return;
  }
  const cur = readSpecFromForm();
  const diffs = [];
  if (cur.asset_a !== refSpec.asset_a) diffs.push(`Asset A (${refSpec.asset_a} → ${cur.asset_a})`);
  if ((cur.asset_b || null) !== (refSpec.asset_b || null)) diffs.push(`Asset B (${refSpec.asset_b || 'None'} → ${cur.asset_b || 'None'})`);
  if ((cur.benchmark_ticker || 'SPY') !== (refSpec.benchmark_ticker || 'SPY')) {
    diffs.push(`Benchmark (${refSpec.benchmark_ticker || 'SPY'} → ${cur.benchmark_ticker || 'SPY'})`);
  }
  if (cur.strategy_family !== refSpec.strategy_family) diffs.push(`Family (${refSpec.strategy_family} → ${cur.strategy_family})`);
  if (cur.template_id !== refSpec.template_id) diffs.push(`Template (${refSpec.template_id} → ${cur.template_id})`);
  if (cur.signal_variant !== refSpec.signal_variant) diffs.push(`Variant (${refSpec.signal_variant} → ${cur.signal_variant})`);
  if (cur.parameters.entry_threshold !== refSpec.parameters.entry_threshold) {
    diffs.push(`Entry Threshold (${refSpec.parameters.entry_threshold} → ${cur.parameters.entry_threshold})`);
  }
  if (cur.parameters.exit_threshold !== refSpec.parameters.exit_threshold) {
    diffs.push(`Exit Threshold (${refSpec.parameters.exit_threshold} → ${cur.parameters.exit_threshold})`);
  }
  if (cur.parameters.lookback_window !== refSpec.parameters.lookback_window) {
    diffs.push(`Lookback (${refSpec.parameters.lookback_window}d → ${cur.parameters.lookback_window}d)`);
  }
  if (cur.parameters.secondary_window !== refSpec.parameters.secondary_window) {
    diffs.push(`Secondary Window (${refSpec.parameters.secondary_window}d → ${cur.parameters.secondary_window}d)`);
  }
  if (cur.parameters.streak_length !== refSpec.parameters.streak_length) {
    diffs.push(`Streak (${refSpec.parameters.streak_length}d → ${cur.parameters.streak_length}d)`);
  }
  if (cur.parameters.holding_period !== refSpec.parameters.holding_period) {
    diffs.push(`Hold (${refSpec.parameters.holding_period}d → ${cur.parameters.holding_period}d)`);
  }
  if (cur.parameters.exit_mode !== refSpec.parameters.exit_mode) {
    diffs.push(`Exit Mode (${refSpec.parameters.exit_mode} → ${cur.parameters.exit_mode})`);
  }
  if (cur.parameters.direction !== refSpec.parameters.direction) {
    diffs.push(`Direction (${refSpec.parameters.direction} → ${cur.parameters.direction})`);
  }
  if (cur.parameters.stop_z_threshold !== (refSpec.parameters.stop_z_threshold ?? 4.0)) {
    diffs.push(`Stop Z (${refSpec.parameters.stop_z_threshold ?? 4.0} → ${cur.parameters.stop_z_threshold})`);
  }
  if (cur.start_date !== refSpec.start_date || cur.end_date !== refSpec.end_date) {
    diffs.push(`Date Range (${refSpec.start_date}..${refSpec.end_date} → ${cur.start_date}..${cur.end_date})`);
  }
  if (cur.execution.train_split_ratio !== refSpec.execution.train_split_ratio) {
    diffs.push(`Split (${Math.round(refSpec.execution.train_split_ratio * 100)}% → ${Math.round(cur.execution.train_split_ratio * 100)}%)`);
  }
  if (cur.execution.execution_timing !== refSpec.execution.execution_timing) {
    diffs.push(`Fill Timing (${refSpec.execution.execution_timing} → ${cur.execution.execution_timing})`);
  }
  if (cur.execution.commission_bps !== refSpec.execution.commission_bps || cur.execution.slippage_bps !== refSpec.execution.slippage_bps) {
    diffs.push(`Friction (${refSpec.execution.commission_bps + refSpec.execution.slippage_bps} → ${cur.execution.commission_bps + cur.execution.slippage_bps} bps)`);
  }
  if (cur.execution.initial_capital !== refSpec.execution.initial_capital) {
    diffs.push(`Capital ($${refSpec.execution.initial_capital} → $${cur.execution.initial_capital})`);
  }
  if (cur.risk_controls.max_drawdown_circuit_breaker_pct !== (refSpec.risk_controls.max_drawdown_circuit_breaker_pct ?? 25)) {
    diffs.push(`Max DD Breaker (${refSpec.risk_controls.max_drawdown_circuit_breaker_pct ?? 25}% → ${cur.risk_controls.max_drawdown_circuit_breaker_pct}%)`);
  }
  if (cur.risk_controls.per_trade_stop_loss_pct !== (refSpec.risk_controls.per_trade_stop_loss_pct ?? 15)) {
    diffs.push(`Per-Trade Stop (${refSpec.risk_controls.per_trade_stop_loss_pct ?? 15}% → ${cur.risk_controls.per_trade_stop_loss_pct}%)`);
  }
  if (cur.threshold_mode !== (refSpec.threshold_mode || 'user_specified')) {
    diffs.push(`Threshold Mode (${refSpec.threshold_mode || 'user_specified'} → ${cur.threshold_mode})`);
  }

  if (diffs.length > 0) {
    banner.style.display = 'block';
    const label = state.baselineReport ? 'Modified since last confirmed run' : 'Modified from initial parsed baseline';
    banner.innerHTML = `<strong style="color:var(--accent-amber);">${label}:</strong> ${diffs.map(escapeHtml).join(' · ')}`;
  } else {
    banner.style.display = 'none';
  }
}

function resetConfirmationGate() {
  const chk = document.getElementById('chk-human-confirm');
  const btn = document.getElementById('btn-run-audit');
  chk.checked = false;
  btn.disabled = true;
  syncStep2SummaryFromForm();
  updateModifiedSinceLastRunBanner();
}

async function validateTickerInput(inputId, statusId) {
  const raw = document.getElementById(inputId).value.trim();
  const statusEl = document.getElementById(statusId);
  if (!raw) return;
  statusEl.className = 'ticker-status';
  statusEl.textContent = `Querying dataset for ${raw}...`;

  try {
    const res = await fetch(`/api/ticker/validate?symbol=${encodeURIComponent(raw)}`);
    const data = await res.json();
    if (data.valid) {
      document.getElementById(inputId).value = data.resolved_ticker;
      statusEl.className = 'ticker-status ok';
      statusEl.textContent = `${data.resolved_ticker}: ${data.total_rows} bars (${data.first_date}..${data.last_date}) Last=$${data.latest_close}`;
      syncStep2SummaryFromForm();
      updateModifiedSinceLastRunBanner();
    } else {
      statusEl.className = 'ticker-status err';
      statusEl.textContent = `${data.error}`;
    }
  } catch (err) {
    statusEl.className = 'ticker-status err';
    statusEl.textContent = `Validation request failed`;
  }
}

function appendUserChatTurn(text) {
  const thread = document.getElementById('chat-thread');
  if (!thread) return;

  const userMsg = document.createElement('div');
  userMsg.className = 'chat-msg user';
  const safeText = escapeHtml(text);
  userMsg.innerHTML = `
    <div class="chat-bubble">
      <div class="chat-sender" style="color:var(--text-primary);">You</div>
      <div>${safeText}</div>
    </div>
  `;
  thread.appendChild(userMsg);
  thread.scrollTop = thread.scrollHeight;
}

async function compileHypothesis(text, fromUserAction = false) {
  const resolvedText =
    typeof text === 'string' && text.trim()
      ? text.trim()
      : document.getElementById('hypothesis-input')?.value?.trim() || '';
  if (!resolvedText) return;
  text = resolvedText;
  const sendBtn = document.getElementById('btn-compile-hypothesis');
  if (sendBtn && sendBtn.disabled) return;
  if (sendBtn) sendBtn.disabled = true;

  const input = document.getElementById('hypothesis-input');
  const status = document.getElementById('chat-status');
  status.textContent = 'Matching your hypothesis to a basket…';
  const badge = document.getElementById('compiler-status-badge');
  if (badge) {
    badge.textContent = 'Matching your hypothesis to a basket…';
    badge.className = 'panel-badge cyan';
  }

  let interp;
  try {
    const res = await fetch('/api/hypothesis/interpret', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        hypothesis_text: text,
        chat_history: state.chatHistory,
      }),
    });
    if (!res.ok) {
      const error = await res.json().catch(() => ({}));
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Gemini chat is unavailable. Please try again.');
    }
    interp = await res.json();
    if (interp.interpretation_mode !== 'live_gemini' || !interp.gemini_reply?.trim()) {
      throw new Error('Gemini did not return a reply. Please try again.');
    }
  } catch (err) {
    status.textContent = err instanceof TypeError ? 'Could not reach the chat server. Please try again.' : err.message;
    if (badge) {
      badge.textContent = 'Gemini unavailable';
      badge.className = 'panel-badge amber';
    }
    return false;
  } finally {
    if (sendBtn) sendBtn.disabled = false;
  }

  status.textContent = '';
  if (fromUserAction) appendUserChatTurn(text);
  if (input.value.trim() === text) input.value = '';
  state.chatHistory = interp.chat_history;
  const modelTag = document.getElementById('compiler-model-tag');
  if (modelTag) modelTag.textContent = `Basket classifier: ${interp.model_used}`;
  const thread = document.getElementById('chat-thread');
  const ambBox = document.getElementById('ambiguity-container');

  if (!interp.proposed_spec) {
    if (badge) {
      badge.textContent = interp.status === 'unsupported' ? 'No matching basket' : 'More detail needed';
      badge.className = 'panel-badge amber';
    }
    if (thread) {
      const replyEl = document.createElement('div');
      replyEl.className = 'chat-msg assistant dynamic-assistant';
      replyEl.innerHTML = `
        <div class="chat-bubble" style="border-color:rgba(244, 63, 94, 0.45);">
          <div class="chat-sender" style="color:var(--accent-rose);">Alpha Auditor</div>
          <div>${escapeHtml(interp.gemini_reply)}</div>
        </div>
      `;
      thread.appendChild(replyEl);
      thread.scrollTop = thread.scrollHeight;
    }
    return;
  }

  if (badge) {
    badge.textContent = 'Basket matched';
    badge.className = 'panel-badge emerald';
  }

  const basket = state.catalog.templates.find(t => t.template_id === interp.proposed_spec.template_id);
  const safeReply = escapeHtml(interp.gemini_reply);
  const formattedReply = safeReply
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.*?)`/g, '<code>$1</code>');

  const turnSteeredCards = Array.isArray(interp.steered_cards) ? interp.steered_cards : [];
  state.lastSteeredCards = turnSteeredCards;

  let inlineAmbiguitiesHtml = '';
  if (interp.ambiguities && interp.ambiguities.length > 0) {
    inlineAmbiguitiesHtml = `
      <div class="chat-inline-guidance">
        <div class="chat-suggestions-label" style="color:var(--accent-amber); margin-bottom:6px;">
          Optional parameters (software defaults pre-filled; click to override):
        </div>
        ${interp.ambiguities
          .map(
            (q) => `
            <div class="chat-guidance-row">
              <span class="chat-guidance-q">${escapeHtml(q.question)}</span>
              <div class="chip-row">
                ${(q.suggested_options || [])
                  .map(
                    (opt) => `
                    <button type="button" class="option-chip chat-opt-chip ${String(opt) === String(q.current_value) ? 'active' : ''}"
                            data-key="${escapeHtml(q.parameter_key)}" data-val="${escapeHtml(opt)}">
                      ${escapeHtml(opt)} ${escapeHtml(q.unit)}
                    </button>
                  `
                  )
                  .join('')}
              </div>
            </div>
          `
          )
          .join('')}
      </div>
    `;
  }

  let variationsHtml = '';
  if (turnSteeredCards.length > 0) {
    variationsHtml = `
      <div style="margin-top:14px;">
        <div class="chat-suggestions-label">Alternative analytical lenses &amp; specifications:</div>
        <div class="chip-row" style="margin-top:6px;">
          ${turnSteeredCards
            .map(
              (c, idx) => `
              <button type="button" class="option-chip steered-pill" data-idx="${idx}" title="${escapeHtml(c.hypothesis_text)}">
                ${escapeHtml(c.label)}
              </button>
            `
            )
            .join('')}
        </div>
      </div>
    `;
  }

  if (thread) {
    // Historical controls reference the shared configuration form. Only the current proposal may edit it.
    thread.querySelectorAll('.dynamic-assistant button').forEach((button) => {
      button.disabled = true;
      button.title = 'A newer proposal is available below.';
    });
    const replyEl = document.createElement('div');
    replyEl.className = 'chat-msg assistant dynamic-assistant';
    replyEl.innerHTML = `
      <div class="chat-bubble">
        <div class="chat-sender">Gemini · Matched basket</div>
        <strong>${escapeHtml(basket?.name || interp.proposed_spec.template_id)}</strong>
        <div class="chat-sender" style="margin-top:12px;">Alpha Auditor · Proposed rule</div>
        <div>${formattedReply}</div>
        ${inlineAmbiguitiesHtml}
        ${variationsHtml}
        <div class="chat-cta-row" style="display:flex; gap:12px; flex-wrap:wrap; margin-top:18px;">
          <button type="button" class="btn btn-emerald btn-chat-goto-step2">
            Continue to Configuration
          </button>
          <button type="button" class="btn btn-secondary btn-chat-auto-bracket">
            Test Standard Thresholds
          </button>
        </div>
      </div>
    `;
    thread.appendChild(replyEl);
    thread.scrollTop = thread.scrollHeight;

    replyEl.querySelectorAll('.chat-opt-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        applyAmbiguityChipToForm(chip.dataset.key, chip.dataset.val);
      });
    });

    replyEl.querySelectorAll('.steered-pill').forEach((pill) => {
      pill.addEventListener('click', () => {
        const idx = parseInt(pill.dataset.idx, 10);
        const chosen = turnSteeredCards[idx];
        if (chosen && chosen.proposed_spec) {
          document.querySelectorAll('#chat-thread .steered-pill').forEach((p) => p.classList.remove('active'));
          pill.classList.add('active');
          document.getElementById('hypothesis-input').value = chosen.hypothesis_text;
          state.baselineReport = null;
          state.initialParsedSpec = JSON.parse(JSON.stringify(chosen.proposed_spec));
          if (ambBox) ambBox.style.display = 'none';
          const notesList = document.getElementById('interpretation-notes-list');
          if (notesList) {
            notesList.innerHTML = [
              `Selected Steered Specification: ${chosen.label}`,
              `Hypothesis Formulation: ${chosen.hypothesis_text}`,
              `Analytical Rationale: ${chosen.rationale || 'Structured parametric lens selected from Step 1.'}`,
              `Causal Execution Safeguard: Signal evaluated strictly at bar t close and filled at bar t+1 (${chosen.proposed_spec.execution?.execution_timing || 'next_open'}).`,
            ]
              .map((n) => `<li>${escapeHtml(n)}</li>`)
              .join('');
          }
          populateSpecForm(chosen.proposed_spec);
          switchTab('view-step-2');
        }
      });
    });

    const btnGoto = replyEl.querySelector('.btn-chat-goto-step2');
    if (btnGoto) {
      btnGoto.addEventListener('click', () => switchTab('view-step-2'));
    }
    const btnBracket = replyEl.querySelector('.btn-chat-auto-bracket');
    if (btnBracket) {
      btnBracket.addEventListener('click', () => {
        setThresholdMode('auto_standard_bracket');
        switchTab('view-step-2');
      });
    }
  }

  const notesList = document.getElementById('interpretation-notes-list');
  notesList.innerHTML = (interp.interpretation_notes || [])
    .map((n) => `<li>${escapeHtml(n)}</li>`)
    .join('');

  if (interp.ambiguities && interp.ambiguities.length > 0) {
    ambBox.style.display = 'block';
    const ambList = document.getElementById('ambiguity-list');
    ambList.innerHTML = interp.ambiguities
      .map(
        (q) => `
        <div class="ambiguity-item" data-amb-key="${escapeHtml(q.parameter_key)}" data-amb-unit="${escapeHtml(q.unit)}">
          <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:12px; color:var(--text-primary);">
            <span><strong>${escapeHtml(q.label)}</strong></span>
            <span class="amb-current-tag" style="color:var(--accent-amber);">Default: ${escapeHtml(q.current_value)} ${escapeHtml(q.unit)}</span>
          </div>
          <div style="font-size:12px; color:var(--text-secondary); margin-top:2px;">${escapeHtml(q.question)}</div>
          <div class="chip-row">
            ${(q.suggested_options || [])
              .map(
                (opt) => `
                <button type="button" class="option-chip ${String(opt) === String(q.current_value) ? 'active' : ''}"
                        data-key="${escapeHtml(q.parameter_key)}" data-val="${escapeHtml(opt)}">
                  ${escapeHtml(opt)} ${escapeHtml(q.unit)}
                </button>
              `
              )
              .join('')}
          </div>
        </div>
      `
      )
      .join('');

    ambList.querySelectorAll('.option-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        applyAmbiguityChipToForm(chip.dataset.key, chip.dataset.val);
      });
    });
  } else {
    ambBox.style.display = 'none';
  }

  if (interp.proposed_spec) {
    state.baselineReport = null;
    state.initialParsedSpec = JSON.parse(JSON.stringify(interp.proposed_spec));
    populateSpecForm(interp.proposed_spec);
    return true;
  }
  return false;
}

function renderUserVsDefaultsBox(spec) {
  const box = document.getElementById('user-vs-defaults-container');
  if (!box || !spec) return;
  const userFields = Array.isArray(spec.user_specified_fields) ? spec.user_specified_fields : [];
  const rawDefs = spec.software_inferred_defaults || {};
  const defEntries = Array.isArray(rawDefs)
    ? rawDefs.map((d) => [String(d), ''])
    : Object.entries(rawDefs);

  if (userFields.length === 0 && defEntries.length === 0) {
    box.style.display = 'none';
    return;
  }
  box.style.display = 'block';
  box.innerHTML = `
    <div class="grid-2" style="gap:18px;">
      <div>
        <div class="section-kicker" style="color:var(--accent-emerald); margin-bottom:6px;">
          Extracted / Confirmed By User (${userFields.length})
        </div>
        <div style="font-family:var(--font-mono); font-size:12px; color:var(--text-primary); line-height:1.6;">
          ${
            userFields.length
              ? userFields
                  .map(
                    (f) =>
                      `<span class="subtle-tag" style="display:inline-block; margin:2px 4px 2px 0;">${escapeHtml(f)}</span>`
                  )
                  .join('')
              : 'None explicitly specified (using template defaults)'
          }
        </div>
      </div>
      <div>
        <div class="section-kicker" style="color:var(--accent-amber); margin-bottom:6px;">
          Software-Added Standard Defaults (${defEntries.length})
        </div>
        <div style="font-family:var(--font-mono); font-size:12px; color:var(--text-secondary); line-height:1.6;">
          ${
            defEntries.length
              ? defEntries
                  .map(
                    ([k, v]) =>
                      `<span class="subtle-tag" style="display:inline-block; margin:2px 4px 2px 0;">${escapeHtml(k)}${v ? `: ${escapeHtml(v)}` : ''}</span>`
                  )
                  .join('')
              : 'All parameters explicitly specified'
          }
        </div>
      </div>
    </div>
  `;
}

function syncStep2SummaryFromForm() {
  const tbody = document.querySelector('#understood-summary-table tbody');
  if (!tbody || !state.catalog) return;

  const cur = readSpecFromForm();
  const ref = state.initialParsedSpec;
  const userFields = new Set(Array.isArray(cur.user_specified_fields) ? cur.user_specified_fields : []);

  if (ref && state.currentSpec) {
    const manualDiffChecks = [
      ['asset_a', cur.asset_a !== ref.asset_a, ['asset_a']],
      ['asset_pair', (cur.asset_b || null) !== (ref.asset_b || null), ['asset_b', 'asset_pair']],
      ['template_id', cur.template_id !== ref.template_id, ['template_id', 'strategy_family']],
      ['signal_variant', cur.signal_variant !== ref.signal_variant, ['signal_variant']],
      [
        'parameters.entry_threshold',
        cur.parameters.entry_threshold !== ref.parameters.entry_threshold,
        ['entry_threshold', 'parameters.entry_threshold'],
      ],
      [
        'parameters.exit_threshold',
        cur.parameters.exit_threshold !== ref.parameters.exit_threshold,
        ['exit_threshold', 'parameters.exit_threshold'],
      ],
      [
        'parameters.lookback_window',
        cur.parameters.lookback_window !== ref.parameters.lookback_window,
        ['lookback_window', 'parameters.lookback_window'],
      ],
      [
        'parameters.secondary_window',
        cur.parameters.secondary_window !== ref.parameters.secondary_window,
        ['secondary_window', 'parameters.secondary_window'],
      ],
      [
        'parameters.streak_length',
        cur.parameters.streak_length !== ref.parameters.streak_length,
        ['streak_length', 'parameters.streak_length', 'entry_threshold'],
      ],
      [
        'parameters.holding_period',
        cur.parameters.holding_period !== ref.parameters.holding_period,
        ['holding_period', 'parameters.holding_period'],
      ],
      [
        'parameters.exit_mode',
        cur.parameters.exit_mode !== ref.parameters.exit_mode,
        ['exit_mode', 'parameters.exit_mode'],
      ],
      [
        'parameters.direction',
        cur.parameters.direction !== ref.parameters.direction,
        ['direction', 'parameters.direction'],
      ],
      [
        'execution.execution_timing',
        cur.execution.execution_timing !== ref.execution.execution_timing,
        ['execution_timing', 'execution.execution_timing'],
      ],
      [
        'execution.cost_bps',
        cur.execution.commission_bps + cur.execution.slippage_bps !==
          ref.execution.commission_bps + ref.execution.slippage_bps,
        ['commission_bps', 'slippage_bps', 'execution.cost_bps'],
      ],
      [
        'risk_controls.per_trade_stop_loss_pct',
        cur.risk_controls.per_trade_stop_loss_pct !== ref.risk_controls.per_trade_stop_loss_pct,
        ['per_trade_stop_loss_pct', 'risk_controls.per_trade_stop_loss_pct'],
      ],
      [
        'risk_controls.max_drawdown_circuit_breaker_pct',
        cur.risk_controls.max_drawdown_circuit_breaker_pct !==
          ref.risk_controls.max_drawdown_circuit_breaker_pct,
        ['max_drawdown_circuit_breaker_pct', 'risk_controls.max_drawdown_circuit_breaker_pct'],
      ],
    ];

    for (const [fieldKey, isChanged, defKeys] of manualDiffChecks) {
      if (isChanged) {
        userFields.add(fieldKey);
        for (const dk of defKeys) {
          delete cur.software_inferred_defaults[dk];
          if (
            state.currentSpec.software_inferred_defaults &&
            typeof state.currentSpec.software_inferred_defaults === 'object'
          ) {
            delete state.currentSpec.software_inferred_defaults[dk];
          }
        }
      }
    }
    cur.user_specified_fields = Array.from(userFields);
    state.currentSpec.user_specified_fields = Array.from(userFields);
  }

  // Keep ambiguity chips & tags synchronized with current form values
  const paramValueMap = {
    asset_a: cur.asset_a,
    asset_pair: cur.asset_b ? `${cur.asset_a} / ${cur.asset_b}` : cur.asset_a,
    signal_variant: cur.signal_variant,
    'parameters.entry_threshold': cur.parameters.entry_threshold,
    'parameters.exit_threshold': cur.parameters.exit_threshold,
    'parameters.lookback_window': cur.parameters.lookback_window,
    'parameters.secondary_window': cur.parameters.secondary_window,
    'parameters.streak_length': cur.parameters.streak_length,
    'parameters.holding_period': cur.parameters.holding_period,
  };
  Object.entries(paramValueMap).forEach(([pKey, pVal]) => {
    document.querySelectorAll(`.option-chip[data-key="${pKey}"]`).forEach((chip) => {
      chip.classList.toggle('active', String(chip.dataset.val) === String(pVal));
    });
    if (userFields.has(pKey)) {
      document.querySelectorAll(`.ambiguity-item[data-amb-key="${pKey}"]`).forEach((item) => {
        const tag = item.querySelector('.amb-current-tag');
        const unit = item.dataset.ambUnit || '';
        if (tag) {
          tag.textContent = `Selected: ${pVal} ${unit}`.trim();
          tag.style.color = 'var(--accent-emerald)';
        }
      });
    }
  });

  const tpl = state.catalog.templates.find((t) => t.template_id === cur.template_id);
  const vObj = tpl?.variants.find((v) => v.id === cur.signal_variant);
  const tplName = tpl ? tpl.name : cur.template_id;
  const varLabel = vObj ? vObj.label : cur.signal_variant;

  const assetStr =
    cur.strategy_family === 'pairs_trading' && cur.asset_b
      ? `${cur.asset_a} vs. ${cur.asset_b} (Relative-Value Spread · Benchmarked vs. ${cur.benchmark_ticker})`
      : `${cur.asset_a} (Single-Asset Directional · Benchmarked vs. ${cur.benchmark_ticker})`;
  const assetUserSpecified =
    userFields.has('asset_a') ||
    userFields.has('asset_pair') ||
    (ref && (cur.asset_a !== ref.asset_a || (cur.asset_b || null) !== (ref.asset_b || null)));

  const familyStr = `${tplName} — ${varLabel}`;
  const familyUserSpecified =
    userFields.has('template_id') ||
    userFields.has('signal_variant') ||
    (ref && (cur.template_id !== ref.template_id || cur.signal_variant !== ref.signal_variant));

  let entryRuleDesc = '';
  if (cur.threshold_mode === 'auto_standard_bracket') {
    entryRuleDesc = `Auto Standard Literature Bracket (Aggressive / Baseline / Conservative profiles around ${cur.parameters.entry_threshold}, Lookback = ${cur.parameters.lookback_window}d)`;
  } else if (cur.template_id === 'return_threshold') {
    const unit = cur.signal_variant === 'vol_scaled_shock' ? 'σ' : '%';
    const dirWord = cur.signal_variant === 'cum_return_surge' ? 'surge ≥ +' : 'drop ≤ -';
    entryRuleDesc = `Enter ${cur.parameters.direction.toUpperCase()} when ${cur.parameters.lookback_window}-day return ${dirWord}${cur.parameters.entry_threshold}${unit}`;
  } else if (cur.template_id === 'consecutive_streak') {
    const isUpStreak =
      cur.signal_variant === 'positive_streak' || cur.signal_variant === 'consecutive_up';
    entryRuleDesc = `Enter ${cur.parameters.direction.toUpperCase()} after ≥ ${cur.parameters.streak_length} consecutive ${isUpStreak ? 'up' : 'down'} trading days`;
  } else if (cur.template_id === 'mean_reversion_oscillator') {
    if (cur.signal_variant === 'rsi_extremes') {
      entryRuleDesc = `Enter ${cur.parameters.direction.toUpperCase()} when ${cur.parameters.lookback_window}-day Wilder RSI reaches ≤ ${cur.parameters.entry_threshold} (Exit target = ${cur.parameters.exit_threshold || 50})`;
    } else {
      entryRuleDesc = `Enter ${cur.parameters.direction.toUpperCase()} at ${cur.parameters.lookback_window}-day excursion |Z| ≥ ${cur.parameters.entry_threshold}σ (Exit target = ${cur.parameters.exit_threshold}σ)`;
    }
  } else if (cur.template_id === 'trend_breakout') {
    entryRuleDesc = `Enter ${cur.parameters.direction.toUpperCase()} on ${cur.parameters.lookback_window}d / ${cur.parameters.secondary_window}d ${varLabel} (Buffer = ${cur.parameters.entry_threshold}%)`;
  } else if (cur.template_id === 'pairs_cointegration_reversion') {
    entryRuleDesc = `Enter dollar-neutral spread when ${cur.parameters.lookback_window}-day Z-score |Z| ≥ ${cur.parameters.entry_threshold}σ (Exit |Z| ≤ ${cur.parameters.exit_threshold}σ, Stop |Z| ≥ ${cur.parameters.stop_z_threshold}σ)`;
  } else {
    entryRuleDesc = `Threshold = ${cur.parameters.entry_threshold} | Lookback = ${cur.parameters.lookback_window}d`;
  }

  const threshUserSpecified =
    userFields.has('parameters.entry_threshold') ||
    userFields.has('parameters.lookback_window') ||
    userFields.has('parameters.streak_length') ||
    (ref &&
      (cur.parameters.entry_threshold !== ref.parameters.entry_threshold ||
        cur.parameters.lookback_window !== ref.parameters.lookback_window ||
        cur.parameters.streak_length !== ref.parameters.streak_length));

  const exitModeLabels = {
    fixed_hold: 'Fixed Holding Horizon',
    threshold_reversion: 'Threshold Convergence',
    first_positive_close: 'First Profitable Close',
    signal_reversal: 'Mean-Reversion / Signal Exit',
    either: 'Whichever Comes First (Signal Reversal or Max Hold)',
  };
  const holdExitStr = `Hold up to ${cur.parameters.holding_period} trading day(s) (${exitModeLabels[cur.parameters.exit_mode] || cur.parameters.exit_mode}) · Stop-Loss = ${cur.risk_controls.per_trade_stop_loss_pct}%, Max DD Breaker = ${cur.risk_controls.max_drawdown_circuit_breaker_pct}%`;
  const holdUserSpecified =
    userFields.has('parameters.holding_period') ||
    userFields.has('risk_controls.per_trade_stop_loss_pct') ||
    userFields.has('risk_controls.max_drawdown_circuit_breaker_pct') ||
    (ref &&
      (cur.parameters.holding_period !== ref.parameters.holding_period ||
        cur.parameters.exit_mode !== ref.parameters.exit_mode ||
        cur.risk_controls.per_trade_stop_loss_pct !== ref.risk_controls.per_trade_stop_loss_pct ||
        cur.risk_controls.max_drawdown_circuit_breaker_pct !==
          ref.risk_controls.max_drawdown_circuit_breaker_pct));

  const trainPct = Math.round(cur.execution.train_split_ratio * 100);
  const execStr = `${cur.start_date} to ${cur.end_date} (${trainPct}% Formation / ${100 - trainPct}% Held-Out OOS) · Bar t+1 (${cur.execution.execution_timing}) · Friction = ${cur.execution.commission_bps + cur.execution.slippage_bps} bps/leg`;
  const execUserSpecified =
    userFields.has('execution.execution_timing') ||
    userFields.has('execution.cost_bps') ||
    (ref &&
      (cur.execution.commission_bps !== ref.execution.commission_bps ||
        cur.execution.slippage_bps !== ref.execution.slippage_bps ||
        cur.execution.train_split_ratio !== ref.execution.train_split_ratio ||
        cur.execution.execution_timing !== ref.execution.execution_timing ||
        cur.start_date !== ref.start_date ||
        cur.end_date !== ref.end_date));

  const rows = [
    ['Target Asset(s) & Benchmark', assetStr, assetUserSpecified ? 'User / Prompt' : 'Software Default'],
    ['Strategy Family & Signal', familyStr, familyUserSpecified ? 'User / Prompt' : 'Software Default'],
    [
      'Entry Threshold & Lookback',
      entryRuleDesc,
      cur.threshold_mode === 'auto_standard_bracket'
        ? 'Auto Literature Bracket'
        : threshUserSpecified
        ? 'User / Prompt'
        : 'Software Default',
    ],
    ['Holding Period, Exit & Risk', holdExitStr, holdUserSpecified ? 'User / Prompt' : 'Software Default'],
    ['Sample Split & Execution', execStr, execUserSpecified ? 'User / Prompt' : 'Software Default'],
  ];

  tbody.innerHTML = rows
    .map(([k, v, p]) => {
      const badgeCls =
        p === 'Software Default' ? 'amber' : p === 'Auto Literature Bracket' ? 'cyan' : 'emerald';
      return `
      <tr>
        <td style="width:26%; color:var(--accent-cyan);"><strong>${escapeHtml(k)}</strong></td>
        <td style="white-space:normal; font-family:var(--font-sans);">${escapeHtml(v)}</td>
        <td><span class="panel-badge ${badgeCls}">${escapeHtml(p)}</span></td>
      </tr>
    `;
    })
    .join('');

  renderUserVsDefaultsBox(cur);
  const repSpec = state.currentReport?.spec;
  if (
    !state.currentReport ||
    !state.baselineReport ||
    !repSpec ||
    cur.template_id !== repSpec.template_id ||
    cur.signal_variant !== repSpec.signal_variant ||
    cur.asset_a !== repSpec.asset_a ||
    (cur.asset_b || null) !== (repSpec.asset_b || null)
  ) {
    renderLiteraturePreviewFromSpec(cur);
  }
}

function renderUnderstoodSummaryFromSpec(spec) {
  syncStep2SummaryFromForm();
}

function applyAmbiguityChipToForm(paramKey, rawVal) {
  if (paramKey === 'asset_a') {
    document.getElementById('spec-asset-a').value = rawVal;
    validateTickerInput('spec-asset-a', 'status-asset-a');
  } else if (paramKey === 'asset_pair') {
    const parts = rawVal.split('/').map((s) => s.trim());
    if (parts.length === 2) {
      document.getElementById('spec-asset-a').value = parts[0];
      document.getElementById('spec-asset-b').value = parts[1];
      document.getElementById('spec-title').value = `${parts[0]}/${parts[1]} Cointegration & Pairs Spread Mean-Reversion`;
      validateTickerInput('spec-asset-a', 'status-asset-a');
      validateTickerInput('spec-asset-b', 'status-asset-b');
    }
  } else if (paramKey === 'signal_variant') {
    const tplId = document.getElementById('spec-template').value;
    updateVariantDropdown(tplId, rawVal);
  } else if (paramKey === 'parameters.entry_threshold') {
    document.getElementById('spec-entry-thresh').value = rawVal;
    if (document.getElementById('spec-template').value === 'consecutive_streak') {
      document.getElementById('spec-streak').value = String(Math.max(1, Math.round(numOr(rawVal, 3))));
    }
  } else if (paramKey === 'parameters.exit_threshold') {
    document.getElementById('spec-exit-thresh').value = rawVal;
  } else if (paramKey === 'parameters.lookback_window') {
    document.getElementById('spec-lookback').value = rawVal;
  } else if (paramKey === 'parameters.secondary_window') {
    document.getElementById('spec-secondary-window').value = rawVal;
  } else if (paramKey === 'parameters.streak_length') {
    document.getElementById('spec-streak').value = rawVal;
    document.getElementById('spec-entry-thresh').value = rawVal;
  } else if (paramKey === 'parameters.holding_period') {
    document.getElementById('spec-holding-period').value = rawVal;
  }

  // Synchronize active chip state across Step 1 inline chat chips and Step 2 ambiguity list
  document.querySelectorAll(`.option-chip[data-key="${paramKey}"]`).forEach((chip) => {
    chip.classList.toggle('active', String(chip.dataset.val) === String(rawVal));
  });
  document.querySelectorAll(`.ambiguity-item[data-amb-key="${paramKey}"]`).forEach((item) => {
    const tag = item.querySelector('.amb-current-tag');
    const unit = item.dataset.ambUnit || '';
    if (tag) {
      tag.textContent = `Selected: ${rawVal} ${unit}`.trim();
      tag.style.color = 'var(--accent-emerald)';
    }
  });

  // Record explicit user override in currentSpec provenance
  if (state.currentSpec) {
    if (!Array.isArray(state.currentSpec.user_specified_fields)) {
      state.currentSpec.user_specified_fields = [];
    }
    if (!state.currentSpec.user_specified_fields.includes(paramKey)) {
      state.currentSpec.user_specified_fields.push(paramKey);
    }
    const shortKey = paramKey.replace(/^parameters\./, '');
    if (
      state.currentSpec.software_inferred_defaults &&
      typeof state.currentSpec.software_inferred_defaults === 'object' &&
      !Array.isArray(state.currentSpec.software_inferred_defaults)
    ) {
      delete state.currentSpec.software_inferred_defaults[shortKey];
      delete state.currentSpec.software_inferred_defaults[paramKey];
    }
  }

  resetConfirmationGate();
}

function populateSpecForm(spec) {
  state.currentSpec = JSON.parse(JSON.stringify(spec));
  document.getElementById('spec-title').value = spec.title;
  document.getElementById('spec-family').value = spec.strategy_family;
  updateTemplateDropdown(spec.strategy_family, spec.template_id, spec.signal_variant);

  document.getElementById('spec-asset-a').value = spec.asset_a;
  if (spec.asset_b) {
    document.getElementById('spec-asset-b').value = spec.asset_b;
  }
  document.getElementById('spec-benchmark').value = spec.benchmark_ticker || 'SPY';

  const p = spec.parameters;
  document.getElementById('spec-entry-thresh').value = p.entry_threshold;
  document.getElementById('spec-exit-thresh').value = p.exit_threshold;
  document.getElementById('spec-lookback').value = p.lookback_window;
  document.getElementById('spec-secondary-window').value = p.secondary_window;
  document.getElementById('spec-streak').value = p.streak_length;
  document.getElementById('spec-holding-period').value = p.holding_period;
  document.getElementById('spec-exit-mode').value = p.exit_mode;
  document.getElementById('spec-direction').value = p.direction;
  document.getElementById('spec-stop-z').value = p.stop_z_threshold ?? 4.0;

  document.getElementById('spec-start-date').value = spec.start_date;
  document.getElementById('spec-end-date').value = spec.end_date;
  document.getElementById('spec-split-ratio').value = String(spec.execution.train_split_ratio);
  document.getElementById('spec-exec-timing').value = spec.execution.execution_timing;
  document.getElementById('spec-comm-bps').value = spec.execution.commission_bps;
  document.getElementById('spec-slip-bps').value = spec.execution.slippage_bps;
  document.getElementById('spec-capital').value = spec.execution.initial_capital;

  document.getElementById('spec-max-dd').value = spec.risk_controls.max_drawdown_circuit_breaker_pct ?? 25;
  document.getElementById('spec-stop-loss').value = spec.risk_controls.per_trade_stop_loss_pct ?? 15;

  setThresholdMode(spec.threshold_mode || 'user_specified');
  resetConfirmationGate();
}

function readSpecFromForm() {
  const family = document.getElementById('spec-family').value;
  const holdDays = intOr(document.getElementById('spec-holding-period').value, 5);
  const rawDefs = state.currentSpec?.software_inferred_defaults;
  const safeDefs =
    rawDefs && typeof rawDefs === 'object' && !Array.isArray(rawDefs) ? rawDefs : {};
  return {
    hypothesis_text:
      state.currentSpec?.hypothesis_text ||
      document.getElementById('spec-title').value.trim() ||
      'Historical market hypothesis evaluation',
    title:
      document.getElementById('spec-title').value.trim() ||
      'Historical Strategy Evaluation',
    strategy_family: family,
    template_id: document.getElementById('spec-template').value,
    signal_variant: document.getElementById('spec-variant').value,
    asset_a: document.getElementById('spec-asset-a').value.trim().toUpperCase(),
    asset_b:
      family === 'pairs_trading'
        ? document.getElementById('spec-asset-b').value.trim().toUpperCase()
        : null,
    benchmark_ticker: document.getElementById('spec-benchmark').value.trim().toUpperCase() || 'SPY',
    start_date: document.getElementById('spec-start-date').value,
    end_date: document.getElementById('spec-end-date').value,
    threshold_mode: state.thresholdMode,
    user_specified_fields: Array.isArray(state.currentSpec?.user_specified_fields)
      ? state.currentSpec.user_specified_fields
      : [],
    software_inferred_defaults: safeDefs,
    evaluation_stage: 'confirmatory',
    prior_holdout_touches: 0,
    parameters: {
      lookback_window: intOr(document.getElementById('spec-lookback').value, 20),
      secondary_window: intOr(document.getElementById('spec-secondary-window').value, 50),
      streak_length: intOr(document.getElementById('spec-streak').value, 3),
      entry_threshold: numOr(document.getElementById('spec-entry-thresh').value, 2.0),
      exit_threshold: numOr(document.getElementById('spec-exit-thresh').value, 0.0),
      stop_z_threshold: numOr(document.getElementById('spec-stop-z').value, 4.0),
      holding_period: holdDays,
      exit_mode: document.getElementById('spec-exit-mode').value,
      direction: document.getElementById('spec-direction').value,
    },
    execution: {
      initial_capital: numOr(document.getElementById('spec-capital').value, 100000),
      execution_timing: document.getElementById('spec-exec-timing').value,
      commission_bps: numOr(document.getElementById('spec-comm-bps').value, 0),
      slippage_bps: numOr(document.getElementById('spec-slip-bps').value, 0),
      train_split_ratio: numOr(document.getElementById('spec-split-ratio').value, 0.5),
    },
    risk_controls: {
      max_drawdown_circuit_breaker_pct: numOr(document.getElementById('spec-max-dd').value, 25.0),
      per_trade_stop_loss_pct: numOr(document.getElementById('spec-stop-loss').value, 15.0),
      max_holding_days: Math.max(30, holdDays * 2),
    },
    selected_diagnostics: [
      'stationary_bootstrap',
      'newey_west_alpha',
      'subperiod_regimes',
      'cost_sensitivity',
      'parameter_sensitivity_grid',
    ],
    human_confirmed: document.getElementById('chk-human-confirm').checked,
  };
}

function updateHoldoutTouchBadge(spec) {
  const badge = document.getElementById('holdout-touch-badge');
  if (!badge || !spec) return;
  const key = getAssetPairKey(spec);
  const touches = state.holdoutTouchesByAssetPair[key] || 1;
  badge.textContent =
    touches > 1
      ? `Holdout Touches (${key}): ${touches} — Exploratory Reuse`
      : `Holdout Touches (${key}): 1 (Confirmatory)`;
  badge.className = touches > 1 ? 'panel-badge amber' : 'subtle-tag';
}

async function executeAuditFromForm() {
  const errBanner = document.getElementById('run-error-banner');
  errBanner.style.display = 'none';
  const btn = document.getElementById('btn-run-audit');
  const origLabel = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'RUNNING BACKTEST, LITERATURE, STRESS SUITE & SCORECARD...';

  try {
    const spec = readSpecFromForm();
    const pairKey = getAssetPairKey(spec);
    const touches = (state.holdoutTouchesByAssetPair[pairKey] || 0) + 1;
    state.holdoutTouchesByAssetPair[pairKey] = touches;
    spec.prior_holdout_touches = touches - 1;
    spec.evaluation_stage = touches > 1 ? 'exploratory_tuner' : 'confirmatory';

    const res = await fetch('/api/audit/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(spec),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Audit execution failed');
    }
    const report = await res.json();
    state.currentSpec = report.spec;
    state.currentReport = report;
    state.baselineReport = report;
    state.activeStressIdx = 0;

    const liveBtn = document.getElementById('btn-recompute-live');
    if (liveBtn) liveBtn.disabled = false;

    updateHoldoutTouchBadge(report.spec);
    syncLiveExplorerFromSpec(report.spec);
    updateModifiedSinceLastRunBanner();
    renderFullAuditReport(report);
    // Navigate directly to Performance & Verdict view (with immediate top verdict banner)
    switchTab('view-step-4');
  } catch (err) {
    errBanner.textContent = `Execution Error: ${err.message}`;
    errBanner.style.display = 'block';
  } finally {
    btn.disabled = !document.getElementById('chk-human-confirm').checked;
    btn.textContent = origLabel;
  }
}

function syncLiveExplorerFromSpec(spec) {
  const tpl = state.catalog.templates.find((t) => t.template_id === spec.template_id);
  const liveVar = document.getElementById('live-variant');
  if (tpl) {
    liveVar.innerHTML = tpl.variants
      .map((v) => `<option value="${v.id}">${v.label}</option>`)
      .join('');
    liveVar.value = spec.signal_variant;
  }
  document.getElementById('live-entry-thresh').value = spec.parameters.entry_threshold;
  document.getElementById('val-live-entry').textContent = spec.parameters.entry_threshold;

  document.getElementById('live-exit-thresh').value = spec.parameters.exit_threshold;
  document.getElementById('val-live-exit').textContent = spec.parameters.exit_threshold;

  document.getElementById('live-lookback').value = spec.parameters.lookback_window;
  document.getElementById('val-live-lookback').textContent = spec.parameters.lookback_window;

  document.getElementById('live-holding').value = spec.parameters.holding_period;
  document.getElementById('val-live-hold').textContent = spec.parameters.holding_period;

  const totalCost = spec.execution.commission_bps + spec.execution.slippage_bps;
  document.getElementById('live-cost-bps').value = totalCost;
  document.getElementById('val-live-cost').textContent = totalCost;
}

async function executeLiveExplorerRecompute() {
  if (!state.currentSpec || !state.currentReport) return;
  const liveBtn = document.getElementById('btn-recompute-live');
  const statusEl = document.getElementById('live-sweep-status');
  if (liveBtn) liveBtn.disabled = true;
  statusEl.textContent = 'Recomputing deterministic backtest, stress tests & scorecard...';

  const modSpec = JSON.parse(JSON.stringify(state.currentSpec));
  modSpec.signal_variant = document.getElementById('live-variant').value;
  modSpec.parameters.entry_threshold = numOr(document.getElementById('live-entry-thresh').value, 2.0);
  modSpec.parameters.exit_threshold = numOr(document.getElementById('live-exit-thresh').value, 0.0);
  modSpec.parameters.lookback_window = intOr(document.getElementById('live-lookback').value, 20);
  modSpec.parameters.holding_period = intOr(document.getElementById('live-holding').value, 5);
  if (modSpec.template_id === 'consecutive_streak') {
    modSpec.parameters.streak_length = Math.max(1, Math.round(modSpec.parameters.entry_threshold));
  }
  const totalCost = numOr(document.getElementById('live-cost-bps').value, 0);
  modSpec.execution.commission_bps = totalCost / 2.0;
  modSpec.execution.slippage_bps = totalCost / 2.0;

  const pairKey = getAssetPairKey(modSpec);
  const touches = (state.holdoutTouchesByAssetPair[pairKey] || 1) + 1;
  state.holdoutTouchesByAssetPair[pairKey] = touches;
  modSpec.evaluation_stage = 'exploratory_tuner';
  modSpec.prior_holdout_touches = touches - 1;
  modSpec.human_confirmed = true;

  document.getElementById('val-live-entry').textContent = modSpec.parameters.entry_threshold;
  document.getElementById('val-live-exit').textContent = modSpec.parameters.exit_threshold;
  document.getElementById('val-live-lookback').textContent = modSpec.parameters.lookback_window;
  document.getElementById('val-live-hold').textContent = modSpec.parameters.holding_period;
  document.getElementById('val-live-cost').textContent = totalCost;

  // Keep Step 2 configuration form synchronized with exploratory parameter tuner changes
  document.getElementById('spec-variant').value = modSpec.signal_variant;
  updateThresholdFieldLabels(modSpec.template_id, modSpec.signal_variant);
  document.getElementById('spec-entry-thresh').value = modSpec.parameters.entry_threshold;
  document.getElementById('spec-exit-thresh').value = modSpec.parameters.exit_threshold;
  document.getElementById('spec-lookback').value = modSpec.parameters.lookback_window;
  if (modSpec.template_id === 'consecutive_streak') {
    document.getElementById('spec-streak').value = modSpec.parameters.streak_length;
  }
  document.getElementById('spec-holding-period').value = modSpec.parameters.holding_period;
  document.getElementById('spec-comm-bps').value = modSpec.execution.commission_bps;
  document.getElementById('spec-slip-bps').value = modSpec.execution.slippage_bps;

  try {
    const res = await fetch('/api/audit/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(modSpec),
    });
    if (!res.ok) throw new Error('Recompute failed');
    const report = await res.json();
    state.currentReport = report;
    state.currentSpec = report.spec;
    updateHoldoutTouchBadge(report.spec);
    syncStep2SummaryFromForm();
    updateModifiedSinceLastRunBanner();
    renderFullAuditReport(report);

    const baseOOS = state.baselineReport?.out_of_sample_metrics?.sharpe_ratio ?? 0;
    const newOOS = report.out_of_sample_metrics.sharpe_ratio;
    const diffNum = newOOS - baseOOS;
    const diff = diffNum.toFixed(3);
    statusEl.innerHTML = `Updated (Touch #${touches}): Held-Out Sharpe = <strong>${report.out_of_sample_metrics.is_zero_exposure ? 'N/A (0 trades)' : newOOS.toFixed(3)}</strong> (Δ vs initial: ${diffNum >= 0 ? '+' : ''}${diff}) | Trades = <strong>${report.full_sample_metrics.trade_count}</strong>`;
  } catch (err) {
    statusEl.textContent = `Recompute error: ${err.message}`;
  } finally {
    if (liveBtn) liveBtn.disabled = false;
  }
}

function renderFullAuditReport(report) {
  renderLiteratureStep(report);
  renderStep4VerdictBanner(report);
  renderKPIStrip(report);
  renderBracketComparison(report);
  renderSplitComparisonTable(report);
  renderEquityAndDrawdownCharts(report);
  renderSignalTrajectoryChart(report);
  renderEconometricsTab(report);
  renderStressTestsStep(report);
  renderScorecardAndCalculations(report);
  renderProvenanceAndBlotter(report);
}

/* =========================================================================
   STEP 3 RENDERER: LITERATURE & BACKTESTING RIGOR
   ========================================================================= */
const PREVIEW_STRATEGY_LITERATURE = {
  return_threshold: {
    mechanism:
      'Short-horizon return shock and overreaction strategies test whether sharp single-day or multi-day price drops/surges reflect temporary dealer inventory imbalances and panic liquidity demand (Lehmann, 1990; Jegadeesh, 1990) or information-driven momentum (Jegadeesh & Titman, 1993).',
    papers: [
      {
        title: 'Fads, Martingales, and Market Efficiency',
        authors: 'Lehmann, B. N.',
        year: 1990,
        journal: 'Quarterly Journal of Economics, 105(1), 1–28',
        core_finding:
          'Documents significant short-horizon return reversals at 1-to-5 day horizons driven by short-term liquidity demand and inventory shocks.',
        empirical_pitfall_warned:
          'Same-bar close fills and bid-ask bounce severely overstate short-horizon reversal Sharpe ratios.',
        how_alpha_auditor_addresses:
          'Enforces strict bar t+1 execution and tests net returns across 0 to 50 bps/leg transaction friction.',
      },
      {
        title: 'Evidence of Predictable Behavior of Security Returns',
        authors: 'Jegadeesh, N.',
        year: 1990,
        journal: 'Journal of Finance, 45(3), 881–898',
        core_finding:
          'Demonstrates negative first-order serial correlation at short horizons contrasted with intermediate-horizon continuation.',
        empirical_pitfall_warned:
          'Data-mining specific drop percentages in-sample frequently fails out-of-sample across volatility regimes.',
        how_alpha_auditor_addresses:
          'Evaluates Aggressive, Baseline, and Conservative literature threshold brackets plus a 25-cell 2D parameter grid.',
      },
    ],
  },
  consecutive_streak: {
    mechanism:
      'Multi-day directional streak strategies test whether k consecutive up or down closes induce short-term exhaustion and mean reversion (De Bondt & Thaler, 1985) or reflect persistent institutional order flow (Moskowitz, Ooi & Pedersen, 2012).',
    papers: [
      {
        title: 'Does the Stock Market Overreact?',
        authors: 'De Bondt, W. F. M., & Thaler, R.',
        year: 1985,
        journal: 'Journal of Finance, 40(3), 793–805',
        core_finding:
          'Establishes the behavioral overreaction hypothesis: sequences of directional returns are often followed by subsequent reversals.',
        empirical_pitfall_warned:
          'Small sample sizes (N < 15) at long streak lengths (k ≥ 5) produce noisy Sharpe estimates with wide standard errors.',
        how_alpha_auditor_addresses:
          'Reports Lo (2002) non-normal Sharpe standard errors, exact binomial sign-test p-values, and stationary bootstrap CIs.',
      },
      {
        title: 'Time Series Momentum',
        authors: 'Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H.',
        year: 2012,
        journal: 'Journal of Financial Economics, 104(2), 228–250',
        core_finding:
          'Documents strong time-series continuation across liquid instruments before longer-horizon reversals occur.',
        empirical_pitfall_warned:
          'Buying into a multi-day decline during a structural macro drawdown can suffer severe left-tail crash risk.',
        how_alpha_auditor_addresses:
          'Enforces per-trade stop-loss and portfolio max-drawdown circuit breakers and stress-tests regime sub-windows.',
      },
    ],
  },
  mean_reversion_oscillator: {
    mechanism:
      'Normalized oscillator strategies (Rolling Z-Score, Bollinger Band excursion, Wilder RSI) scale price deviations by local trailing volatility so entry thresholds adapt across quiet and turbulent market regimes (Brock, Lakonishok & LeBaron, 1992; Lo, Mamaysky & Wang, 2000).',
    papers: [
      {
        title: 'Simple Technical Trading Rules and the Stochastic Properties of Stock Returns',
        authors: 'Brock, W., Lakonishok, J., & LeBaron, B.',
        year: 1992,
        journal: 'Journal of Finance, 47(5), 1731–1764',
        core_finding:
          'Evaluates oscillator and band rules via bootstrap distributions against null return-generating processes.',
        empirical_pitfall_warned:
          'Standard normal t-tests ignore autocorrelation and conditional heteroskedasticity in oscillator returns.',
        how_alpha_auditor_addresses:
          'Uses Politis-Romano (1994) stationary block bootstrap and Newey-West (1987) HAC standard errors.',
      },
      {
        title: 'Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation',
        authors: 'Lo, A. W., Mamaysky, H., & Wang, J.',
        year: 2000,
        journal: 'Journal of Finance, 55(4), 1705–1765',
        core_finding:
          'Formalizes nonparametric conditioning and statistical inference for technical oscillator indicators.',
        empirical_pitfall_warned:
          'Fine-tuning lookback windows and RSI/Z levels creates severe multiple-testing selection bias.',
        how_alpha_auditor_addresses:
          'Applies the Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR) across K=25 parameter grid trials.',
      },
    ],
  },
  trend_breakout: {
    mechanism:
      'Trend-following and channel breakout rules (Moving Average Crossover, Donchian Channel Breakout) seek to capture intermediate-term price momentum and convex positive skewness during sustained macro trends (Jegadeesh & Titman, 1993; Hurst, Ooi & Pedersen, 2017).',
    papers: [
      {
        title: 'Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency',
        authors: 'Jegadeesh, N., & Titman, S.',
        year: 1993,
        journal: 'Journal of Finance, 48(1), 65–91',
        core_finding:
          'Demonstrates persistent intermediate-horizon return continuation across equities.',
        empirical_pitfall_warned:
          'Trend rules suffer frequent false breakouts ("whipsaws") in mean-reverting or range-bound regimes.',
        how_alpha_auditor_addresses:
          'Decomposes performance across 5 macro sub-regimes and stress-tests t+2 execution latency and slippage.',
      },
      {
        title: 'A Century of Evidence on Trend-Following Investing',
        authors: 'Hurst, B., Ooi, Y. H., & Pedersen, L. H.',
        year: 2017,
        journal: 'Journal of Portfolio Management, 44(1), 15–29',
        core_finding:
          'Shows that time-series trend rules exhibit lower win rates (~35–45%) compensated by positive payoff skewness.',
        empirical_pitfall_warned:
          'Evaluating trend strategies solely by win rate or without benchmark beta decomposition conflates market drift with timing skill.',
        how_alpha_auditor_addresses:
          'Decomposes returns into systematic benchmark Beta (β) vs. Newey-West HAC Jensen Alpha (α).',
      },
    ],
  },
  pairs_cointegration_reversion: {
    mechanism:
      'Statistical arbitrage pairs trading tests whether two economically linked assets share a stationary cointegrating equilibrium ln(P_A) - α - β ln(P_B) (Engle & Granger, 1987; Gatev, Goetzmann & Rouwenhorst, 2006), entering dollar-neutral long/short positions when the spread Z-score diverges.',
    papers: [
      {
        title: 'Co-Integration and Error Correction: Representation, Estimation, and Testing',
        authors: 'Engle, R. F., & Granger, C. W. J.',
        year: 1987,
        journal: 'Econometrica, 55(2), 251–276',
        core_finding:
          'Introduces the two-step cointegration test and proves that cointegrated series admit an error-correction representation.',
        empirical_pitfall_warned:
          'High daily return correlation does NOT imply price-level cointegration; non-stationary spreads diverge permanently.',
        how_alpha_auditor_addresses:
          'Separately tests daily return correlation, Engle-Granger / ADF spread stationarity, Ornstein-Uhlenbeck half-life, and trading P&L.',
      },
      {
        title: 'Pairs Trading: Performance of a Relative-Value Arbitrage Rule',
        authors: 'Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G.',
        year: 2006,
        journal: 'Review of Financial Studies, 19(3), 797–827',
        core_finding:
          'Benchmarks formation-period spread estimation and out-of-sample 2σ trading rules across U.S. equities.',
        empirical_pitfall_warned:
          'Estimating the hedge ratio β̂ on the full sample leaks future relative price drift into early trades.',
        how_alpha_auditor_addresses:
          'Estimates OLS hedge ratio β̂ strictly on the pre-holdout Formation window and freezes it for Out-of-Sample trading.',
      },
    ],
  },
};

const PREVIEW_METHODOLOGY_PAPERS = [
  {
    title: 'The Statistics of Sharpe Ratios',
    authors: 'Lo, A. W.',
    year: 2002,
    journal: 'Financial Analysts Journal, 58(4), 36–52',
    core_finding:
      'Derives asymptotic standard errors for Sharpe ratios accounting for non-normality, skewness, and serial correlation.',
    empirical_pitfall_warned: 'Assuming i.i.d. normal returns understates Sharpe ratio estimation error.',
    how_alpha_auditor_addresses: 'Computes Lo (2002) / Mertens skewness-and-kurtosis-adjusted Sharpe standard errors.',
  },
  {
    title: 'The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality',
    authors: 'Bailey, D. H., & López de Prado, M.',
    year: 2014,
    journal: 'Journal of Portfolio Management, 40(5), 94–107',
    core_finding:
      'Penalizes observed Sharpe ratios for the number of independent parameter trials K and return non-normality.',
    empirical_pitfall_warned: 'Selecting the best parameter combination from a grid inflates false discoveries.',
    how_alpha_auditor_addresses: 'Evaluates a K=25 parameter grid and reports the Deflated Sharpe Ratio (DSR) haircut.',
  },
];

function renderLiteraturePreviewFromSpec(spec) {
  if (!spec) return;
  const tplLit =
    PREVIEW_STRATEGY_LITERATURE[spec.template_id] || PREVIEW_STRATEGY_LITERATURE.return_threshold;
  const mechEl = document.getElementById('lit-mechanism-summary');
  if (mechEl) {
    mechEl.textContent = tplLit.mechanism;
  }

  const synthEl = document.getElementById('lit-gemini-synthesis');
  if (synthEl) {
    const assetDesc =
      spec.strategy_family === 'pairs_trading' && spec.asset_b
        ? `${spec.asset_a} vs. ${spec.asset_b}`
        : spec.asset_a;
    synthEl.style.display = 'block';
    synthEl.innerHTML = `
      <div style="font-family:var(--font-mono); font-size:10.5px; color:var(--accent-cyan); text-transform:uppercase; margin-bottom:4px;">
        PRE-AUDIT LITERATURE &amp; SAFEGUARD BRIEFING (${escapeHtml(assetDesc)} · ${escapeHtml(spec.signal_variant)})
      </div>
      Configured specification maps to <strong>${escapeHtml(spec.template_id)}</strong> (${escapeHtml(spec.signal_variant)}). Confirm and run the audit in Step 2 to generate the empirical post-backtest synthesis comparing observed Formation vs. Held-Out Out-of-Sample metrics against this literature baseline.
    `;
  }

  const renderPapers = (papers) =>
    (papers || [])
      .map(
        (p) => `
      <div class="lit-paper-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:8px; margin-bottom:5px;">
          <div style="font-weight:700; color:var(--text-primary); font-size:13px;">${escapeHtml(p.title)}</div>
          <span class="panel-badge cyan" style="white-space:nowrap;">${escapeHtml(p.authors)} (${escapeHtml(p.year)})</span>
        </div>
        <div style="font-family:var(--font-mono); font-size:11px; color:var(--accent-cyan); margin-bottom:6px;">
          ${escapeHtml(p.journal)}
        </div>
        <div style="font-size:12px; color:var(--text-primary); margin-bottom:5px;">
          <strong>Core Finding:</strong> ${escapeHtml(p.core_finding)}
        </div>
        <div style="font-size:12px; color:var(--accent-amber); margin-bottom:5px;">
          <strong>Documented Empirical Pitfall:</strong> ${escapeHtml(p.empirical_pitfall_warned)}
        </div>
        <div style="font-size:12px; color:var(--accent-emerald);">
          <strong>How Alpha Auditor Addresses It:</strong> ${escapeHtml(p.how_alpha_auditor_addresses)}
        </div>
      </div>
    `
      )
      .join('');

  const stratEl = document.getElementById('lit-strategy-papers');
  if (stratEl) stratEl.innerHTML = renderPapers(tplLit.papers);
  const methEl = document.getElementById('lit-methodology-papers');
  if (methEl) methEl.innerHTML = renderPapers(PREVIEW_METHODOLOGY_PAPERS);

  const rigorTbody = document.querySelector('#lit-rigor-table tbody');
  if (rigorTbody) {
    const trainPct = Math.round((spec.execution?.train_split_ratio || 0.5) * 100);
    const totalFriction = (spec.execution?.commission_bps || 0) + (spec.execution?.slippage_bps || 0);
    const checklist = [
      {
        safeguard: 'Strict Causal Signal & Execution Lag (t → t+1)',
        status: 'ENFORCED',
        detail: `Signals are computed exclusively from bar t and prior closes; trade fills execute at bar t+1 (${spec.execution?.execution_timing || 'next_open'}).`,
      },
      {
        safeguard: 'Chronological In-Sample / Out-of-Sample Split',
        status: 'ENFORCED',
        detail: `${trainPct}% Formation / ${100 - trainPct}% Held-Out Out-of-Sample partition (${spec.start_date} to ${spec.end_date}) with zero look-ahead leakage.`,
      },
      {
        safeguard: 'Transaction Cost & Slippage Deduction',
        status: 'ENFORCED',
        detail: `Deducts ${totalFriction} bps/leg (${spec.execution?.commission_bps || 0} bps commission + ${spec.execution?.slippage_bps || 0} bps slippage) plus 0–50 bps break-even stress sweep.`,
      },
      {
        safeguard: 'Multiple-Testing & Non-Normal Sharpe Inference',
        status: 'ENFORCED',
        detail: 'Reports Lo (2002) Sharpe SE, Politis-Romano (1994) stationary bootstrap CIs, Newey-West HAC alpha, and Bailey-López de Prado (2014) DSR.',
      },
      {
        safeguard: 'Portfolio Drawdown & Per-Trade Risk Circuit Breakers',
        status: 'ENFORCED',
        detail: `Per-trade stop-loss at -${spec.risk_controls?.per_trade_stop_loss_pct ?? 15}% and portfolio max-drawdown halt at -${spec.risk_controls?.max_drawdown_circuit_breaker_pct ?? 25}%.`,
      },
    ];
    rigorTbody.innerHTML = checklist
      .map(
        (r) => `
        <tr>
          <td><strong>${escapeHtml(r.safeguard)}</strong></td>
          <td><span class="panel-badge emerald">${escapeHtml(r.status)}</span></td>
          <td style="white-space:normal; font-family:var(--font-sans); color:var(--text-secondary);">${escapeHtml(r.detail)}</td>
        </tr>
      `
      )
      .join('');
  }
}

function commentarySource(report) {
  if (report.commentary_model) return `Gemini · ${escapeHtml(report.commentary_model)}`;
  if (report.commentary_error) return `Local explanation · ${escapeHtml(report.commentary_error)}`;
  return 'Local explanation';
}

function renderLiteratureStep(report) {
  const lit = report.literature_review;
  if (!lit) return;

  document.getElementById('lit-mechanism-summary').textContent = lit.theoretical_mechanism_summary;
  const synthEl = document.getElementById('lit-gemini-synthesis');
  if (synthEl) {
    synthEl.style.display = 'block';
    synthEl.innerHTML = `
      <div style="font-family:var(--font-mono); font-size:10.5px; color:var(--accent-cyan); text-transform:uppercase; margin-bottom:4px;">
        Literature &amp; empirical bias · ${commentarySource(report)}
      </div>
      ${escapeHtml(lit.gemini_literature_synthesis)}
    `;
  }

  const renderPapers = (papers) =>
    (papers || [])
      .map(
        (p) => `
      <div class="lit-paper-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:8px; margin-bottom:5px;">
          <div style="font-weight:700; color:var(--text-primary); font-size:13px;">${escapeHtml(p.title)}</div>
          <span class="panel-badge cyan" style="white-space:nowrap;">${escapeHtml(p.authors)} (${escapeHtml(p.year)})</span>
        </div>
        <div style="font-family:var(--font-mono); font-size:11px; color:var(--accent-cyan); margin-bottom:6px;">
          ${escapeHtml(p.journal)}
        </div>
        <div style="font-size:12px; color:var(--text-primary); margin-bottom:5px;">
          <strong>Core Finding:</strong> ${escapeHtml(p.core_finding)}
        </div>
        <div style="font-size:12px; color:var(--accent-amber); margin-bottom:5px;">
          <strong>Documented Empirical Pitfall:</strong> ${escapeHtml(p.empirical_pitfall_warned)}
        </div>
        <div style="font-size:12px; color:var(--accent-emerald);">
          <strong>How Alpha Auditor Addresses It:</strong> ${escapeHtml(p.how_alpha_auditor_addresses)}
        </div>
      </div>
    `
      )
      .join('');

  document.getElementById('lit-strategy-papers').innerHTML = renderPapers(lit.strategy_papers);
  document.getElementById('lit-methodology-papers').innerHTML = renderPapers(lit.methodology_papers);

  document.querySelector('#lit-rigor-table tbody').innerHTML = (lit.rigor_checklist || [])
    .map(
      (r) => `
      <tr>
        <td><strong>${escapeHtml(r.safeguard)}</strong></td>
        <td><span class="panel-badge emerald">${escapeHtml(r.status)}</span></td>
        <td style="white-space:normal; font-family:var(--font-sans); color:var(--text-secondary);">${escapeHtml(r.detail)}</td>
      </tr>
    `
    )
    .join('');
}

/* =========================================================================
   STEP 4 RENDERER: PERFORMANCE, MULTI-THRESHOLD BRACKET & ECONOMETRICS
   ========================================================================= */
function renderStep4VerdictBanner(report) {
  const banner = document.getElementById('step4-verdict-banner');
  if (!banner) return;
  const sc = report.scorecard;
  const oos = report.out_of_sample_metrics;
  if (!sc) {
    banner.style.display = 'none';
    return;
  }

  const verdictLabel = sc.overall_verdict_label || sc.historical_verdict_label;
  const badgeCls =
    verdictLabel === 'SUPPORTED BY HISTORICAL DATA'
      ? 'emerald'
      : verdictLabel === 'CONTRADICTED BY HISTORICAL DATA' || verdictLabel.includes('HALTED')
      ? 'badge-vulnerable'
      : 'amber';

  const evalCount = sc.evaluated_checks_count ?? sc.total_checks_count;
  const breakdown = sc.hypothesis_comparison_breakdown || [];
  let lensHtml = '';
  if (breakdown.length > 0) {
    lensHtml = `
      <div class="hypothesis-comparison-grid" style="margin-top:14px;">
        ${breakdown
          .map((b) => {
            const title = b.lens_title || b.aspect || '';
            const status = b.status || b.verdict || 'EVALUATED';
            const metric = b.primary_metric || b.evidence || '';
            const interp = b.interpretation || b.explanation || '';
            const cls =
              status === 'SUPPORTED' || status.includes('POSITIVE') || status === 'COINTEGRATED (p < 0.05)'
                ? 'emerald'
                : status === 'CONTRADICTED' || status.includes('NEGATIVE') || status.includes('NON-STATIONARY')
                ? 'badge-vulnerable'
                : status === 'NO_EVIDENCE'
                ? 'badge-neutral'
                : 'amber';
            return `
          <div class="hypothesis-lens-card">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:4px;">
              <strong style="font-size:12.5px; color:var(--text-primary);">${escapeHtml(title)}</strong>
              <span class="panel-badge ${cls}">${escapeHtml(b.verdict || status)}</span>
            </div>
            <div style="font-family:var(--font-mono); font-size:12px; color:var(--accent-cyan); margin-bottom:4px;">${escapeHtml(metric)}</div>
            <div style="font-size:12px; color:var(--text-secondary);">${escapeHtml(interp)}</div>
          </div>
        `;
          })
          .join('')}
      </div>
    `;
  }

  banner.style.display = 'block';
  banner.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap;">
      <div style="flex:1; min-width:280px;">
        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:6px;">
          <span class="panel-badge ${badgeCls}">${escapeHtml(verdictLabel)}</span>
          <span style="font-family:var(--font-mono); font-size:12px; color:var(--text-secondary);">
            ${sc.supported_checks_count} / ${evalCount} Evaluated Checks Supported · Stage: ${escapeHtml((report.spec.evaluation_stage || 'confirmatory').replace(/_/g, ' '))}
          </span>
          ${oos.is_zero_exposure ? `<span class="panel-badge badge-neutral">${escapeHtml(oos.status_note || '0 Held-Out Trades')}</span>` : ''}
        </div>
        <div style="font-size:14px; font-weight:600; color:var(--text-primary); margin-bottom:4px;">
          "${escapeHtml(sc.hypothesis_restatement)}"
        </div>
        <div style="font-size:13.5px; color:var(--text-secondary); line-height:1.55;">
          ${escapeHtml(sc.executive_summary)}
        </div>
      </div>
      <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
        <button type="button" class="btn btn-secondary" onclick="switchTab('view-step-6')">
          Full Scorecard &amp; Proofs
        </button>
        <button type="button" class="btn btn-secondary" onclick="downloadNotebookIpynb()">
          Download .ipynb
        </button>
        <button type="button" class="btn btn-emerald" onclick="downloadNotebookPDF()">
          Download PDF
        </button>
      </div>
    </div>
    ${lensHtml}
  `;
}

function renderBracketComparison(report) {
  const profiles = report.bracket_profiles || [];
  const badge = document.getElementById('bracket-mode-badge');
  if (badge) {
    badge.style.display = 'inline-flex';
    badge.textContent =
      report.spec.threshold_mode === 'auto_standard_bracket'
        ? 'AUTO-BRACKET MODE ACTIVE (3 STANDARD LITERATURE PROFILES)'
        : '3 STANDARD LITERATURE PROFILES COMPUTED FOR REFERENCE';
  }

  const tbody = document.querySelector('#bracket-comparison-table tbody');
  tbody.innerHTML = profiles
    .map((p) => {
      const oosZero = p.out_of_sample_metrics.is_zero_exposure;
      const oosSrStr = oosZero ? 'N/A (0 tr)' : fmtNum(p.out_of_sample_metrics.sharpe_ratio, 3);
      return `
      <tr>
        <td><strong>${escapeHtml(p.profile_label)}</strong></td>
        <td style="white-space:normal; font-family:var(--font-sans); color:var(--text-secondary);">${escapeHtml(p.threshold_description)}</td>
        <td class="num">${p.full_sample_metrics.trade_count}</td>
        <td class="num ${colorClass(p.in_sample_metrics.sharpe_ratio)}">${fmtNum(p.in_sample_metrics.sharpe_ratio, 3)}</td>
        <td class="num ${oosZero ? '' : colorClass(p.out_of_sample_metrics.sharpe_ratio)}" style="font-weight:700;">${oosSrStr}</td>
        <td class="num ${colorClass(p.full_sample_metrics.sharpe_ratio)}">${fmtNum(p.full_sample_metrics.sharpe_ratio, 3)}</td>
        <td class="num ${returnColorClassVsBenchmark(p.full_sample_metrics.total_net_return_pct, p.full_sample_metrics.benchmark_return_pct)}">${fmtPct(p.full_sample_metrics.total_net_return_pct)}</td>
        <td class="num val-neg">-${fmtNum(p.full_sample_metrics.max_drawdown_pct)}%</td>
        <td class="num">${fmtNum(p.full_sample_metrics.win_rate_pct, 1)}%</td>
      </tr>
    `;
    })
    .join('');
}

function renderKPIStrip(report) {
  const isM = report.in_sample_metrics;
  const oos = report.out_of_sample_metrics;
  const full = report.full_sample_metrics;
  const fac = report.factor_regression;

  const oosZero = Boolean(oos.is_zero_exposure);
  const oosRetCls = oosZero ? '' : returnColorClassVsBenchmark(oos.total_net_return_pct, oos.benchmark_return_pct);
  const oosSharpeDisplay = oosZero ? 'N/A (0 tr)' : fmtNum(oos.sharpe_ratio, 3);
  const oosSharpeSub = oosZero
    ? oos.halted_prior_to_period
      ? `Halted in IS (${escapeHtml(oos.circuit_breaker_date || '')}) · IS SR = ${fmtNum(isM.sharpe_ratio, 2)}`
      : `Zero OOS exposure · IS SR = ${fmtNum(isM.sharpe_ratio, 2)}`
    : `SE = ±${fmtNum(oos.sharpe_se_lo2002, 3)} (N=${oos.trade_count} tr) · IS SR = ${fmtNum(isM.sharpe_ratio, 2)}`;

  const oosDdVal = oosZero ? '0.00%' : `${fmtNum(oos.max_drawdown_pct, 2)}%`;
  const oosDdCls = oos.max_drawdown_pct > 0 ? 'val-neg' : '';

  const html = `
    <div class="kpi-card kpi-primary-oos">
      <div class="kpi-label">HELD-OUT OOS NET RETURN</div>
      <div class="kpi-value ${oosRetCls}">${fmtPct(oos.total_net_return_pct)}</div>
      <div class="kpi-sub">vs Bench ${fmtPct(oos.benchmark_return_pct)} (${fmtPp(oos.excess_return_pct)}) · N=${oos.trade_count} tr</div>
    </div>
    <div class="kpi-card kpi-primary-oos">
      <div class="kpi-label">HELD-OUT SHARPE (LO 2002)</div>
      <div class="kpi-value ${oosZero ? '' : colorClass(oos.sharpe_ratio)}">${oosSharpeDisplay}</div>
      <div class="kpi-sub">${oosSharpeSub}</div>
    </div>
    <div class="kpi-card kpi-primary-oos">
      <div class="kpi-label">MAX DRAWDOWN LOSS (OOS / FULL)</div>
      <div class="kpi-value ${oosDdCls}">${oosDdVal}</div>
      <div class="kpi-sub">Full Loss: ${fmtNum(full.max_drawdown_pct, 2)}% · Bench: ${fmtNum(full.benchmark_max_drawdown_pct, 2)}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">NEWEY-WEST HAC ANNUAL α</div>
      <div class="kpi-value ${colorClass(fac.annualized_alpha_pct)}">${fmtPct(fac.annualized_alpha_pct)}</div>
      <div class="kpi-sub">t_HAC = ${fmtNum(fac.alpha_t_stat_hac, 2)} (p = ${fmtNum(fac.alpha_p_value_hac, 4)}) | β = ${fmtNum(fac.beta, 2)}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">FULL-SAMPLE PSR(SR &gt; 0)</div>
      <div class="kpi-value">${full.is_zero_exposure ? 'N/A' : `${fmtNum(full.probabilistic_sharpe_psr_pct, 1)}%`}</div>
      <div class="kpi-sub">Skew = ${fmtNum(full.skewness_daily, 2)} | Kurt = ${fmtNum(full.kurtosis_daily, 2)} (${full.trading_days}d)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">SAMPLE SIZE N &amp; WIN RATE</div>
      <div class="kpi-value">${full.trade_count} <span style="font-size:13px; color:var(--text-secondary);">(${fmtNum(full.win_rate_pct, 1)}%)</span></div>
      <div class="kpi-sub">Formation: ${isM.trade_count} tr | Holdout OOS: ${oos.trade_count} tr</div>
    </div>
  `;
  document.getElementById('kpi-strip-container').innerHTML = html;
}

function renderSplitComparisonTable(report) {
  const isM = report.in_sample_metrics;
  const oosM = report.out_of_sample_metrics;
  const fullM = report.full_sample_metrics;
  const oosZero = Boolean(oosM.is_zero_exposure);

  document.getElementById('split-date-badge').textContent =
    `IN-SAMPLE (${isM.start_date}..${isM.end_date}) | HELD-OUT OOS (${oosM.start_date}..${oosM.end_date})`;

  const rows = [
    ['Trading Days / Sample Window', isM.trading_days, oosM.trading_days, fullM.trading_days, fullM.trading_days, 'Chronological non-overlapping partition'],
    ['Completed Historical Trades (N)', isM.trade_count, oosZero ? `0 (${oosM.status_note || 'Zero Exposure'})` : oosM.trade_count, fullM.trade_count, '1 (Buy & Hold)', 'Sample size N for trade-level inference'],
    ['Market Exposure (% of Days Active)', `${isM.exposure_pct}%`, `${oosM.exposure_pct}%`, `${fullM.exposure_pct}%`, '100.0%', 'Fraction of trading days with open position'],
    ['Cumulative Gross Return (%)', fmtPct(isM.total_gross_return_pct), fmtPct(oosM.total_gross_return_pct), fmtPct(fullM.total_gross_return_pct), '—', 'Pre-cost return at bar t+1 fills'],
    ['Cumulative Net Return (%)', fmtPct(isM.total_net_return_pct), fmtPct(oosM.total_net_return_pct), fmtPct(fullM.total_net_return_pct), fmtPct(fullM.benchmark_return_pct), 'After explicit commission + slippage'],
    ['Excess Return vs Benchmark (pp)', fmtPp(isM.excess_return_pct), fmtPp(oosM.excess_return_pct), fmtPp(fullM.excess_return_pct), '0.00 pp', 'Active outperformance in percentage points'],
    ['Compound Annual Growth Rate (CAGR %)', fmtPct(isM.cagr_net_pct), fmtPct(oosM.cagr_net_pct), fmtPct(fullM.cagr_net_pct), fmtPct(fullM.benchmark_cagr_pct), 'Geometric annualized return'],
    ['Annualized Volatility (%)', `${fmtNum(isM.annualized_volatility_pct)}%`, `${fmtNum(oosM.annualized_volatility_pct)}%`, `${fmtNum(fullM.annualized_volatility_pct)}%`, `${fmtNum(fullM.benchmark_volatility_pct)}%`, 'Daily mark-to-market return SD × √252'],
    ['Annualized Sharpe Ratio', fmtNum(isM.sharpe_ratio, 3), oosZero ? 'N/A (0 trades)' : fmtNum(oosM.sharpe_ratio, 3), fmtNum(fullM.sharpe_ratio, 3), fmtNum(fullM.benchmark_sharpe_ratio, 3), 'Risk-adjusted return per unit volatility'],
    ['Sharpe Standard Error (Lo, 2002)', `±${fmtNum(isM.sharpe_se_lo2002, 3)}`, oosZero ? 'N/A' : `±${fmtNum(oosM.sharpe_se_lo2002, 3)}`, `±${fmtNum(fullM.sharpe_se_lo2002, 3)}`, '—', 'Adjusted for sample skewness & fat tails'],
    ['Probabilistic Sharpe Ratio PSR(0)', `${fmtNum(isM.probabilistic_sharpe_psr_pct, 1)}%`, oosZero ? 'N/A (0 trades)' : `${fmtNum(oosM.probabilistic_sharpe_psr_pct, 1)}%`, `${fmtNum(fullM.probabilistic_sharpe_psr_pct, 1)}%`, '—', 'Bailey & López de Prado (2012) probability SR > 0'],
    ['Sortino Ratio (Downside Deviation)', fmtNum(isM.sortino_ratio, 3), oosZero ? 'N/A' : fmtNum(oosM.sortino_ratio, 3), fmtNum(fullM.sortino_ratio, 3), '—', 'Return scaled by downside semi-deviation'],
    ['Maximum Drawdown (Peak-to-Trough Loss)', `${fmtNum(isM.max_drawdown_pct)}%`, `${fmtNum(oosM.max_drawdown_pct)}%`, `${fmtNum(fullM.max_drawdown_pct)}%`, `${fmtNum(fullM.benchmark_max_drawdown_pct)}%`, 'Peak-to-trough daily equity decline (anchored at C_0)'],
    ['Circuit-Breaker Status', isM.circuit_breaker_triggered ? `HALTED (${isM.circuit_breaker_date})` : 'Active', oosM.halted_prior_to_period ? `HALTED IN IS (${oosM.circuit_breaker_date})` : oosM.circuit_breaker_triggered ? `HALTED (${oosM.circuit_breaker_date})` : 'Active', fullM.circuit_breaker_triggered ? `HALTED (${fullM.circuit_breaker_date})` : 'Active', 'N/A', 'Portfolio drawdown limit enforcement'],
    ['Trade Win Rate (%)', `${fmtNum(isM.win_rate_pct, 1)}%`, oosZero ? 'N/A' : `${fmtNum(oosM.win_rate_pct, 1)}%`, `${fmtNum(fullM.win_rate_pct, 1)}%`, '—', 'Percentage of trades with Net Return > 0'],
    ['Mean / Median Trade Net Return (%)', `${fmtPct(isM.mean_trade_net_pct)} / ${fmtPct(isM.median_trade_net_pct)}`, oosZero ? 'N/A' : `${fmtPct(oosM.mean_trade_net_pct)} / ${fmtPct(oosM.median_trade_net_pct)}`, `${fmtPct(fullM.mean_trade_net_pct)} / ${fmtPct(fullM.median_trade_net_pct)}`, '—', 'Compare mean vs median for outlier skew'],
    ['Two-Sided t-Statistic (p-Value)', `${fmtNum(isM.t_stat_mean_return, 2)} (p=${fmtNum(isM.p_value_two_sided, 3)})`, oosZero ? 'N/A (0 trades)' : `${fmtNum(oosM.t_stat_mean_return, 2)} (p=${fmtNum(oosM.p_value_two_sided, 3)})`, `${fmtNum(fullM.t_stat_mean_return, 2)} (p=${fmtNum(fullM.p_value_two_sided, 3)})`, '—', 'Null hypothesis: mean daily net return = 0'],
    ['Exact Binomial Sign-Test p-Value', isM.sign_test_p_value ?? '—', oosZero ? '—' : (oosM.sign_test_p_value ?? '—'), fullM.sign_test_p_value ?? '—', '—', 'Non-parametric test of win count vs 50% coin flip'],
  ];

  const tbody = document.querySelector('#split-comparison-table tbody');
  tbody.innerHTML = rows
    .map(
      (r) => `
      <tr>
        <td><strong>${r[0]}</strong></td>
        <td class="num">${r[1]}</td>
        <td class="num" style="background:rgba(14,165,233,0.05); font-weight:600;">${r[2]}</td>
        <td class="num">${r[3]}</td>
        <td class="num" style="color:var(--text-secondary);">${r[4]}</td>
        <td style="color:var(--text-muted); font-family:var(--font-sans);">${r[5]}</td>
      </tr>
    `
    )
    .join('');
}

function renderEquityAndDrawdownCharts(report) {
  const ts = report.timeseries;
  const splitDate = report.split_date;
  const fullM = report.full_sample_metrics;
  const oosM = report.out_of_sample_metrics;
  const isM = report.in_sample_metrics;
  const baseLayout = getPlotlyLayout();

  // Populate 1-sentence chart takeaway banner & accessible numeric summary table
  const summaryBanner = document.getElementById('equity-chart-summary');
  if (summaryBanner) {
    summaryBanner.style.display = 'flex';
    const cbNote = fullM.circuit_breaker_triggered
      ? ` Circuit breaker tripped on <strong>${fullM.circuit_breaker_date}</strong> at ${fmtNum(fullM.max_drawdown_pct)}% drawdown, halting subsequent entries.`
      : '';
    summaryBanner.innerHTML = `
      <div>
        <strong>Chart Takeaway:</strong> Over the held-out out-of-sample window (${oosM.start_date} to ${oosM.end_date}), the confirmed specification returned <strong>${fmtPct(oosM.total_net_return_pct)}</strong> (${oosM.trade_count} trades) vs. <strong>${fmtPct(oosM.benchmark_return_pct)}</strong> for Buy &amp; Hold (${fmtPp(oosM.excess_return_pct)}).${cbNote}
      </div>
    `;
  }

  const accWrap = document.getElementById('accessible-chart-table-wrap');
  if (accWrap) {
    const endEq = ts.equity_net && ts.equity_net.length ? ts.equity_net[ts.equity_net.length - 1] : report.spec.execution.initial_capital;
    const endBench = ts.equity_benchmark && ts.equity_benchmark.length ? ts.equity_benchmark[ts.equity_benchmark.length - 1] : report.spec.execution.initial_capital;
    accWrap.innerHTML = `
      <table class="quant-table">
        <thead>
          <tr>
            <th>Partition</th>
            <th>Date Range</th>
            <th class="num">Trades (N)</th>
            <th class="num">Strategy Net Return</th>
            <th class="num">Benchmark Return</th>
            <th class="num">Excess (pp)</th>
            <th class="num">Sharpe Ratio</th>
            <th class="num">Max Drawdown</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Formation (In-Sample)</strong></td>
            <td>${isM.start_date}..${isM.end_date}</td>
            <td class="num">${isM.trade_count}</td>
            <td class="num">${fmtPct(isM.total_net_return_pct)}</td>
            <td class="num">${fmtPct(isM.benchmark_return_pct)}</td>
            <td class="num">${fmtPp(isM.excess_return_pct)}</td>
            <td class="num">${fmtNum(isM.sharpe_ratio, 3)}</td>
            <td class="num">-${fmtNum(isM.max_drawdown_pct)}%</td>
          </tr>
          <tr>
            <td><strong>Held-Out Out-of-Sample</strong></td>
            <td>${oosM.start_date}..${oosM.end_date}</td>
            <td class="num">${oosM.trade_count}</td>
            <td class="num">${fmtPct(oosM.total_net_return_pct)}</td>
            <td class="num">${fmtPct(oosM.benchmark_return_pct)}</td>
            <td class="num">${fmtPp(oosM.excess_return_pct)}</td>
            <td class="num">${oosM.is_zero_exposure ? 'N/A (0 trades)' : fmtNum(oosM.sharpe_ratio, 3)}</td>
            <td class="num">-${fmtNum(oosM.max_drawdown_pct)}%</td>
          </tr>
          <tr>
            <td><strong>Full Sample (Terminal Equity: $${Number(endEq).toLocaleString()} vs $${Number(endBench).toLocaleString()})</strong></td>
            <td>${fullM.start_date}..${fullM.end_date}</td>
            <td class="num">${fullM.trade_count}</td>
            <td class="num">${fmtPct(fullM.total_net_return_pct)}</td>
            <td class="num">${fmtPct(fullM.benchmark_return_pct)}</td>
            <td class="num">${fmtPp(fullM.excess_return_pct)}</td>
            <td class="num">${fmtNum(fullM.sharpe_ratio, 3)}</td>
            <td class="num">-${fmtNum(fullM.max_drawdown_pct)}%</td>
          </tr>
        </tbody>
      </table>
    `;
  }

  const tracesEq = [
    {
      x: ts.dates,
      y: ts.equity_net,
      name: 'Confirmed Spec Net',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#059669', width: 2.4 },
    },
    {
      x: ts.dates,
      y: ts.equity_benchmark,
      name: `Buy & Hold (${report.spec.benchmark_ticker})`,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#78716c', width: 1.5 },
    },
  ];

  const showBrackets = Boolean(document.getElementById('chk-show-brackets')?.checked);
  if (showBrackets) {
    const bracketColors = {
      aggressive: '#d97706',
      baseline: '#0d9488',
      conservative: '#6366f1',
    };
    for (const prof of report.bracket_profiles || []) {
      if (prof.equity_net && prof.equity_net.length === ts.dates.length) {
        tracesEq.push({
          x: ts.dates,
          y: prof.equity_net,
          name: prof.profile_label,
          type: 'scatter',
          mode: 'lines',
          line: {
            color: bracketColors[prof.profile_id] || '#0d9488',
            width: 1.4,
            dash: 'dot',
          },
        });
      }
    }
  }

  const shapesEq = [
    {
      type: 'line',
      x0: splitDate,
      x1: splitDate,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: '#059669', width: 1.8, dash: 'dashdot' },
    },
  ];
  const annotationsEq = [
    {
      x: splitDate,
      y: 0.96,
      yref: 'paper',
      text: '← Formation | Held-Out OOS →',
      showarrow: false,
      font: { color: '#059669', size: 10 },
      bgcolor: isDarkMode() ? 'rgba(20,18,16,0.85)' : 'rgba(255,255,255,0.9)',
    },
  ];

  if (fullM.circuit_breaker_triggered && fullM.circuit_breaker_date) {
    shapesEq.push({
      type: 'line',
      x0: fullM.circuit_breaker_date,
      x1: fullM.circuit_breaker_date,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: '#e11d48', width: 1.6, dash: 'dot' },
    });
    annotationsEq.push({
      x: fullM.circuit_breaker_date,
      y: 0.12,
      yref: 'paper',
      text: `Circuit Breaker Halt (${fullM.circuit_breaker_date})`,
      showarrow: false,
      font: { color: '#e11d48', size: 10 },
      bgcolor: isDarkMode() ? 'rgba(20,18,16,0.85)' : 'rgba(255,255,255,0.9)',
    });
  }

  const layoutEq = {
    ...baseLayout,
    yaxis: { ...baseLayout.yaxis, title: 'Portfolio Equity ($)' },
    shapes: shapesEq,
    annotations: annotationsEq,
  };

  safePlotlyNewPlot('chart-equity', tracesEq, layoutEq, PLOTLY_CONFIG);

  const maxDDLimit = report.spec.risk_controls.max_drawdown_circuit_breaker_pct;
  const shapesDD = [
    {
      type: 'line',
      x0: splitDate,
      x1: splitDate,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: '#059669', width: 1.5, dash: 'dashdot' },
    },
  ];
  if (maxDDLimit) {
    shapesDD.push({
      type: 'line',
      x0: ts.dates[0],
      x1: ts.dates[ts.dates.length - 1],
      y0: -maxDDLimit,
      y1: -maxDDLimit,
      line: { color: '#e11d48', width: 1.5, dash: 'dot' },
    });
  }
  if (fullM.circuit_breaker_triggered && fullM.circuit_breaker_date) {
    shapesDD.push({
      type: 'line',
      x0: fullM.circuit_breaker_date,
      x1: fullM.circuit_breaker_date,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: '#e11d48', width: 1.5, dash: 'dot' },
    });
  }

  const tracesDD = [
    {
      x: ts.dates,
      y: ts.drawdown_net_pct,
      name: 'Strategy Drawdown (%)',
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      fillcolor: 'rgba(225, 29, 72, 0.14)',
      line: { color: '#e11d48', width: 1.6 },
    },
  ];

  const layoutDD = {
    ...baseLayout,
    yaxis: { ...baseLayout.yaxis, title: 'Underwater Drawdown (%)' },
    shapes: shapesDD,
  };

  safePlotlyNewPlot('chart-drawdown', tracesDD, layoutDD, PLOTLY_CONFIG);
}

function renderSignalTrajectoryChart(report) {
  const ts = report.timeseries;
  const spec = report.spec;
  const baseLayout = getPlotlyLayout();
  const entryT = Math.abs(spec.parameters.entry_threshold);
  const exitT = Math.abs(spec.parameters.exit_threshold);
  const dir = spec.parameters.direction || 'long';

  const entryDates = report.trades.map((t) => t.signal_date);
  const entryVals = report.trades.map((t) => t.signal_value);

  const tracesSig = [
    {
      x: ts.dates,
      y: ts.signal_values,
      name: `Signal Series (${spec.signal_variant})`,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#0d9488', width: 1.5 },
    },
    {
      x: entryDates,
      y: entryVals,
      name: 'Triggered Trade Signals (t)',
      type: 'scatter',
      mode: 'markers',
      marker: { color: '#059669', size: 7, symbol: 'diamond' },
    },
  ];

  const shapes = [
    {
      type: 'line',
      x0: report.split_date,
      x1: report.split_date,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: '#059669', width: 1.5, dash: 'dashdot' },
    },
  ];

  let yAxisTitle = 'Signal Value';
  let ruleDesc = '';

  if (spec.signal_variant === 'rsi_extremes') {
    yAxisTitle = 'Wilder RSI (0–100 Index)';
    const rawRsi = entryT > 0 ? entryT : 30.0;
    const rsiLow = rawRsi > 50 ? 100.0 - rawRsi : rawRsi;
    const rsiHigh = 100.0 - rsiLow;
    const rsiExit = spec.parameters.exit_threshold > 0 ? spec.parameters.exit_threshold : 50.0;
    if (dir === 'long' || dir === 'both') {
      shapes.push({
        type: 'line',
        x0: ts.dates[0],
        x1: ts.dates[ts.dates.length - 1],
        y0: rsiLow,
        y1: rsiLow,
        line: { color: '#059669', width: 1.3, dash: 'dash' },
      });
    }
    if (dir === 'short' || dir === 'both') {
      shapes.push({
        type: 'line',
        x0: ts.dates[0],
        x1: ts.dates[ts.dates.length - 1],
        y0: rsiHigh,
        y1: rsiHigh,
        line: { color: '#d97706', width: 1.3, dash: 'dash' },
      });
    }
    shapes.push({
      type: 'line',
      x0: ts.dates[0],
      x1: ts.dates[ts.dates.length - 1],
      y0: rsiExit,
      y1: rsiExit,
      line: { color: '#78716c', width: 1.0, dash: 'dot' },
    });
    ruleDesc = `Wilder ${spec.parameters.lookback_window}d RSI (${dir === 'short' ? `short ≥ ${rsiHigh}` : dir === 'both' ? `long ≤ ${rsiLow}, short ≥ ${rsiHigh}` : `long ≤ ${rsiLow}`}, exit ${rsiExit})`;
  } else if (spec.template_id === 'consecutive_streak') {
    const streakLen = Math.max(1, spec.parameters.streak_length || Math.round(entryT));
    yAxisTitle = 'Consecutive Run Length (Days)';
    shapes.push({
      type: 'line',
      x0: ts.dates[0],
      x1: ts.dates[ts.dates.length - 1],
      y0: streakLen,
      y1: streakLen,
      line: { color: '#059669', width: 1.3, dash: 'dash' },
    });
    ruleDesc = `${spec.signal_variant.replace(/_/g, ' ')} trigger at ≥ ${streakLen} consecutive trading days`;
  } else if (spec.strategy_family === 'pairs_trading' || spec.template_id === 'mean_reversion_oscillator') {
    yAxisTitle = 'Z-Score Excursion (σ)';
    const showLower = spec.strategy_family === 'pairs_trading' || dir === 'long' || dir === 'both';
    const showUpper = spec.strategy_family === 'pairs_trading' || dir === 'short' || dir === 'both';
    if (showLower) {
      shapes.push({
        type: 'line',
        x0: ts.dates[0],
        x1: ts.dates[ts.dates.length - 1],
        y0: -entryT,
        y1: -entryT,
        line: { color: '#0d9488', width: 1.2, dash: 'dash' },
      });
    }
    if (showUpper) {
      shapes.push({
        type: 'line',
        x0: ts.dates[0],
        x1: ts.dates[ts.dates.length - 1],
        y0: entryT,
        y1: entryT,
        line: { color: '#d97706', width: 1.2, dash: 'dash' },
      });
    }
    shapes.push({
      type: 'line',
      x0: ts.dates[0],
      x1: ts.dates[ts.dates.length - 1],
      y0: exitT,
      y1: exitT,
      line: { color: '#78716c', width: 1, dash: 'dot' },
    });
    ruleDesc = `${spec.signal_variant} (${spec.parameters.lookback_window}d) entry at ${showLower && showUpper ? `±${entryT}σ` : showLower ? `≤ -${entryT}σ` : `≥ +${entryT}σ`}, exit at ${exitT}σ`;
  } else if (spec.template_id === 'return_threshold') {
    yAxisTitle = spec.signal_variant === 'vol_scaled_shock' ? 'Volatility-Scaled Shock (σ)' : 'Cumulative Return (%)';
    const lineY = spec.signal_variant === 'cum_return_surge' ? entryT : -entryT;
    const unitSym = spec.signal_variant === 'vol_scaled_shock' ? 'σ' : '%';
    shapes.push({
      type: 'line',
      x0: ts.dates[0],
      x1: ts.dates[ts.dates.length - 1],
      y0: lineY,
      y1: lineY,
      line: { color: '#0d9488', width: 1.3, dash: 'dash' },
    });
    ruleDesc = `${spec.parameters.lookback_window}d ${spec.signal_variant} trigger at ${lineY > 0 ? `+${lineY}` : lineY}${unitSym}`;
  } else if (spec.template_id === 'trend_breakout') {
    yAxisTitle = spec.signal_variant === 'donchian_breakout' ? 'Channel Excursion (%)' : 'Moving-Average Spread (%)';
    const threshVal = spec.signal_variant === 'donchian_breakout' ? 0 : entryT;
    shapes.push({
      type: 'line',
      x0: ts.dates[0],
      x1: ts.dates[ts.dates.length - 1],
      y0: threshVal,
      y1: threshVal,
      line: { color: '#059669', width: 1.3, dash: 'dash' },
    });
    ruleDesc = `${spec.signal_variant} (${spec.parameters.lookback_window}d/${spec.parameters.secondary_window}d) trigger at ${threshVal}%`;
  }

  const sigSummaryEl = document.getElementById('signal-chart-summary');
  if (sigSummaryEl) {
    sigSummaryEl.style.display = 'flex';
    sigSummaryEl.innerHTML = `
      <div>
        <strong>Signal Overlay Rule:</strong> ${escapeHtml(ruleDesc)} — triggered <strong>${report.in_sample_metrics.trade_count}</strong> Formation trades and <strong>${report.out_of_sample_metrics.trade_count}</strong> Out-of-Sample trades (diamonds mark causal bar <em>t</em> signal closes, filled at <em>t+1</em> ${escapeHtml(spec.execution.execution_timing)}).
      </div>
    `;
  }

  safePlotlyNewPlot(
    'chart-signal',
    tracesSig,
    {
      ...baseLayout,
      yaxis: { ...baseLayout.yaxis, title: yAxisTitle },
      shapes,
    },
    PLOTLY_CONFIG
  );
}

function renderEconometricsTab(report) {
  const pairsPanel = document.getElementById('pairs-econometrics-panel');
  if (report.pairs_diagnostics) {
    pairsPanel.style.display = 'block';
    const pd = report.pairs_diagnostics;
    document.getElementById('pairs-coint-banner').textContent = pd.cointegration_status_note;

    const critStr = Object.entries(pd.adf_critical_values || {})
      .map(([k, v]) => `${k}: ${v}`)
      .join(', ');

    const rows = [
      ['Formation Estimation Window', `${pd.formation_start} to ${pd.formation_end}`, 'Strictly pre-holdout sample used to fit β̂'],
      ['OLS Cointegrating Hedge Ratio (β̂)', fmtNum(pd.ols_hedge_ratio_beta, 4), 'ln(P_A) = α̂ + β̂ ln(P_B) + S_t'],
      ['OLS Intercept (α̂) & R²', `α̂ = ${fmtNum(pd.ols_alpha, 4)} (R² = ${fmtNum(pd.ols_r_squared, 4)})`, 'Goodness of log-price linear fit'],
      ['Daily Log-Return Correlation (ρ)', fmtNum(pd.return_correlation, 4), 'Short-horizon co-movement (does not imply cointegration)'],
      ['Engle-Granger Two-Step t-Statistic (p-Value)', `${fmtNum(pd.engle_granger_stat, 3)} (p = ${fmtNum(pd.engle_granger_p_value, 4)})`, 'MacKinnon (2010) critical values (reject unit root if p < 0.05)'],
      ['Augmented Dickey-Fuller (ADF) Statistic', `${fmtNum(pd.adf_statistic, 3)} (p = ${fmtNum(pd.adf_p_value, 4)})`, `Critical values: ${critStr}`],
      ['Ornstein-Uhlenbeck Half-Life of Mean Reversion (τ_1/2)', pd.half_life_days ? `${pd.half_life_days} trading days` : '∞ (Non-mean-reverting)', 'Discrete AR(1) exact half-life -ln(2)/ln(1+λ̂)'],
      ['Generalized Hurst Exponent (H)', fmtNum(pd.hurst_exponent, 3), 'H < 0.50 mean-reverting | H = 0.50 Brownian | H > 0.50 trending'],
      ['Lo-MacKinlay (1988) Variance Ratio VR(q=5)', `${fmtNum(pd.variance_ratio_q5, 3)} (p = ${fmtNum(pd.variance_ratio_p_value, 4)})`, 'VR(q) < 1 indicates negative serial correlation'],
    ];

    document.querySelector('#pairs-diag-table tbody').innerHTML = rows
      .map(
        (r) => `
        <tr>
          <td><strong>${r[0]}</strong></td>
          <td class="num">${r[1]}</td>
          <td style="color:var(--text-secondary); font-family:var(--font-sans);">${r[2]}</td>
        </tr>
      `
      )
      .join('');

    const ts = report.timeseries;
    if (ts.rebased_a && ts.rebased_b) {
      const baseLayout = getPlotlyLayout();
      safePlotlyNewPlot(
        'chart-pairs-rebased',
        [
          {
            x: ts.dates,
            y: ts.rebased_a,
            name: `${report.spec.asset_a} (Rebased = 100)`,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#059669', width: 1.8 },
          },
          {
            x: ts.dates,
            y: ts.rebased_b,
            name: `${report.spec.asset_b} (Rebased = 100)`,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#d97706', width: 1.8 },
          },
        ],
        {
          ...baseLayout,
          yaxis: { ...baseLayout.yaxis, title: 'Normalized Price Index (100)' },
        },
        PLOTLY_CONFIG
      );
    }
  } else {
    pairsPanel.style.display = 'none';
  }

  const surf = report.parameter_surface;
  if (surf) {
    const baseLayout = getPlotlyLayout();
    document.getElementById('dsr-badge').textContent =
      `K=${surf.trials_count_k} TRIALS | EXPECTED NULL MAX SR = ${fmtNum(surf.expected_max_sr_under_null, 2)} | CONFIRMED SPEC DSR = ${fmtNum(surf.selected_spec_deflated_sharpe_psr_pct, 1)}%`;
    document.getElementById('dsr-methodology-banner').textContent = surf.methodology_note;
    document.getElementById('th-param-x').textContent = surf.param_x_label;
    document.getElementById('th-param-y').textContent = surf.param_y_label;

    const xVals = surf.param_x_values;
    const yVals = surf.param_y_values;
    const zMatrix = yVals.map((yv) =>
      xVals.map((xv) => {
        const cell = surf.cells.find(
          (c) => Math.abs(c.param_x_value - xv) < 1e-4 && Math.abs(c.param_y_value - yv) < 1e-4
        );
        return cell ? cell.full_sharpe : 0;
      })
    );

    safePlotlyNewPlot(
      'chart-sensitivity-heatmap',
      [
        {
          x: xVals.map(String),
          y: yVals.map(String),
          z: zMatrix,
          type: 'heatmap',
          colorscale: 'RdBu',
          reversescale: true,
          zmid: 0,
          colorbar: { title: 'Sharpe' },
        },
      ],
      {
        ...baseLayout,
        xaxis: { ...baseLayout.xaxis, title: surf.param_x_label },
        yaxis: { ...baseLayout.yaxis, title: surf.param_y_label },
      },
      PLOTLY_CONFIG
    );

    document.querySelector('#sensitivity-grid-table tbody').innerHTML = surf.cells
      .map(
        (c) => `
        <tr>
          <td class="num">${c.param_x_value}</td>
          <td class="num">${c.param_y_value}</td>
          <td class="num">${c.trade_count}</td>
          <td class="num ${colorClass(c.in_sample_sharpe)}">${fmtNum(c.in_sample_sharpe, 3)}</td>
          <td class="num ${colorClass(c.out_of_sample_sharpe)}" style="font-weight:600;">${fmtNum(c.out_of_sample_sharpe, 3)}</td>
          <td class="num ${colorClass(c.full_sharpe)}">${fmtNum(c.full_sharpe, 3)}</td>
          <td class="num ${colorClass(c.full_net_return_pct)}">${fmtPct(c.full_net_return_pct)}</td>
          <td class="num val-neg">-${fmtNum(c.max_drawdown_pct)}%</td>
        </tr>
      `
      )
      .join('');
  }

  document.querySelector('#bootstrap-table tbody').innerHTML = (report.bootstrap_cis || [])
    .map(
      (b) => `
      <tr>
        <td><strong>${escapeHtml(b.metric_name)}</strong></td>
        <td class="num">${b.point_estimate}${b.unit === '%' ? '%' : ''}</td>
        <td class="num">[${b.ci_lower_95}${b.unit === '%' ? '%' : ''}</td>
        <td class="num">${b.ci_upper_95}${b.unit === '%' ? '%' : ''}]</td>
        <td class="num">±${b.bootstrap_std_error}</td>
        <td class="num">${b.block_length}d (${b.replications} reps)</td>
      </tr>
    `
    )
    .join('');

  const fr = report.factor_regression;
  document.querySelector('#factor-reg-table tbody').innerHTML = `
    <tr>
      <td><strong>Annualized Jensen's Alpha (α)</strong></td>
      <td class="num ${colorClass(fr.annualized_alpha_pct)}">${fmtPct(fr.annualized_alpha_pct)}</td>
      <td class="num">t = ${fmtNum(fr.alpha_t_stat_hac, 3)} (p = ${fmtNum(fr.alpha_p_value_hac, 4)})</td>
      <td style="font-family:var(--font-sans); color:var(--text-secondary);">Benchmark-unexplained excess return (Newey-West lags = ${fr.newey_west_lags})</td>
    </tr>
    <tr>
      <td><strong>Systematic Benchmark Beta (β)</strong></td>
      <td class="num">${fmtNum(fr.beta, 3)}</td>
      <td class="num">t = ${fmtNum(fr.beta_t_stat_hac, 3)}</td>
      <td style="font-family:var(--font-sans); color:var(--text-secondary);">Sensitivity to ${escapeHtml(report.spec.benchmark_ticker)} daily returns</td>
    </tr>
    <tr>
      <td><strong>Regression R² &amp; Information Ratio</strong></td>
      <td class="num">R² = ${fmtNum(fr.r_squared, 4)}</td>
      <td class="num">IR = ${fmtNum(fr.information_ratio, 3)}</td>
      <td style="font-family:var(--font-sans); color:var(--text-secondary);">Active return divided by tracking error</td>
    </tr>
  `;

  const cs = report.cost_sensitivity || [];
  document.getElementById('breakeven-badge').textContent =
    report.break_even_cost_bps !== null
      ? `ESTIMATED BREAK-EVEN COST = ${report.break_even_cost_bps} BPS / LEG`
      : 'NEGATIVE GROSS RETURN';

  const baseLayoutCost = getPlotlyLayout();
  safePlotlyNewPlot(
    'chart-cost-sensitivity',
    [
      {
        x: cs.map((r) => `${r.total_cost_bps_per_leg} bps`),
        y: cs.map((r) => r.in_sample_net_return_pct),
        name: 'In-Sample Net Return (%)',
        type: 'bar',
        marker: { color: '#78716c' },
      },
      {
        x: cs.map((r) => `${r.total_cost_bps_per_leg} bps`),
        y: cs.map((r) => r.out_of_sample_net_return_pct),
        name: 'Held-Out OOS Net Return (%)',
        type: 'bar',
        marker: { color: '#059669' },
      },
    ],
    {
      ...baseLayoutCost,
      barmode: 'group',
      yaxis: { ...baseLayoutCost.yaxis, title: 'Cumulative Net Return (%)' },
    },
    PLOTLY_CONFIG
  );

  document.querySelector('#subperiod-table tbody').innerHTML = (report.subperiods || [])
    .map(
      (s) => `
      <tr>
        <td><strong>${s.label}</strong> <span style="color:var(--text-muted);">(${s.start_date}..${s.end_date})</span></td>
        <td class="num">${s.trade_count}</td>
        <td class="num ${colorClass(s.strategy_net_return_pct)}">${fmtPct(s.strategy_net_return_pct)}</td>
        <td class="num">${fmtPct(s.benchmark_return_pct)}</td>
        <td class="num ${colorClass(s.excess_return_pct)}">${fmtPct(s.excess_return_pct)}</td>
        <td class="num ${colorClass(s.sharpe_ratio)}">${fmtNum(s.sharpe_ratio, 3)}</td>
        <td class="num val-neg">-${fmtNum(s.max_drawdown_pct)}%</td>
      </tr>
    `
    )
    .join('');
}

/* =========================================================================
   STEP 5 RENDERER: ADVERSARIAL STRESS TESTS (TABBED 1 BY 1)
   ========================================================================= */
function renderStressTestsStep(report) {
  const suite = report.stress_test_suite;
  if (!suite) return;

  const summaryBadge = document.getElementById('stress-summary-badge');
  if (summaryBadge) {
    summaryBadge.style.display = 'inline-flex';
    summaryBadge.textContent =
      `${suite.resilient_count}/5 RESILIENT · ${5 - suite.resilient_count - suite.vulnerability_count}/5 MODERATE · ${suite.vulnerability_count}/5 HIGH VULNERABILITY`;
  }

  document.getElementById('stress-gemini-overview').innerHTML = `
    <div style="font-family:var(--font-mono); font-size:10.5px; color:var(--accent-cyan); text-transform:uppercase; margin-bottom:4px;">
      Stress test overview · ${commentarySource(report)}
    </div>
    ${escapeHtml(suite.gemini_overview)}
  `;

  const tests = suite.tests || [];
  const activeIdx =
    typeof state.activeStressIdx === 'number' &&
    state.activeStressIdx >= 0 &&
    state.activeStressIdx < tests.length
      ? state.activeStressIdx
      : 0;
  const tabsNav = document.getElementById('stress-tabs-nav');
  const container = document.getElementById('stress-tests-container');

  if (tabsNav) {
    tabsNav.style.display = 'inline-flex';
    tabsNav.innerHTML = tests
      .map((t, idx) => {
        const cleanTitle = t.title.replace(/\s*\([^)]*\)/g, '').trim();
        return `
        <button type="button" class="subtab-btn ${idx === activeIdx ? 'active' : ''}" data-stress-idx="${idx}">
          Test ${idx + 1}: ${escapeHtml(cleanTitle)}
        </button>
      `;
      })
      .join('');
  }

  container.innerHTML = tests
    .map((t, idx) => {
      const badgeCls =
        t.outcome_badge === 'RESILIENT'
          ? 'badge-resilient'
          : t.outcome_badge === 'MODERATE_SENSITIVITY'
          ? 'badge-moderate'
          : 'badge-vulnerable';

      const rows = t.breakdown_rows || [];
      let tableHtml = '';
      if (rows.length > 0) {
        const cols = Object.keys(rows[0]);
        tableHtml = `
          <table class="quant-table">
            <thead>
              <tr>${cols.map((c) => `<th>${escapeHtml(c.replace(/_/g, ' ').toUpperCase())}</th>`).join('')}</tr>
            </thead>
            <tbody>
              ${rows
                .map(
                  (r) =>
                    `<tr>${cols
                      .map((c, ci) => `<td class="${ci > 0 ? 'num' : ''}">${escapeHtml(r[c])}</td>`)
                      .join('')}</tr>`
                )
                .join('')}
            </tbody>
          </table>
        `;
      }

      return `
        <div class="stress-card subtab-panel ${idx === activeIdx ? 'active' : ''}" id="stress-panel-${idx}">
          <div class="panel-header">
            <div class="panel-title">STRESS TEST 0${idx + 1} — ${escapeHtml(t.title)}</div>
            <span class="panel-badge ${badgeCls}">${escapeHtml(t.outcome_badge.replace(/_/g, ' '))}</span>
          </div>
          <div class="panel-body">
            <div class="grid-2" style="margin-bottom:12px;">
              <div>
                <div style="font-size:12px; color:var(--accent-amber); margin-bottom:6px;">
                  <strong>Vulnerability Probed:</strong> ${escapeHtml(t.vulnerability_probed)}
                </div>
                <div style="font-size:12.5px; color:var(--text-secondary); margin-bottom:8px;">
                  <strong>Risk this test probes:</strong> ${escapeHtml(t.gemini_rationale)}
                </div>
                <div class="formula-banner" style="font-family:var(--font-sans); color:var(--text-primary);">
                  <strong>Empirical Finding:</strong> ${escapeHtml(t.key_finding)}
                </div>
                <div class="quant-table-wrap">${tableHtml}</div>
              </div>
              <div>
                <div id="stress-chart-${idx}" class="chart-box-sm"></div>
              </div>
            </div>
          </div>
        </div>
      `;
    })
    .join('');

  const renderSingleStressChart = (idx) => {
    const t = tests[idx];
    if (!t) return;
    const cp = t.chart_payload || {};
    if (cp.labels && cp.labels.length > 0) {
      const baseLayout = getPlotlyLayout();
      const colors = cp.values.map((v) => (v >= 0 ? '#059669' : '#e11d48'));
      safePlotlyNewPlot(
        `stress-chart-${idx}`,
        [
          {
            x: cp.labels,
            y: cp.values,
            type: 'bar',
            marker: { color: colors },
          },
        ],
        {
          ...baseLayout,
          yaxis: { ...baseLayout.yaxis, title: cp.y_label || 'Value' },
        },
        PLOTLY_CONFIG
      );
    }
  };

  if (tabsNav) {
    tabsNav.querySelectorAll('.subtab-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.stressIdx, 10);
        state.activeStressIdx = idx;
        tabsNav.querySelectorAll('.subtab-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        container.querySelectorAll('.stress-card').forEach((card, ci) => {
          card.classList.toggle('active', ci === idx);
        });
        renderSingleStressChart(idx);
        setTimeout(() => {
          window.dispatchEvent(new Event('resize'));
          resizeVisibleCharts();
        }, 50);
      });
    });
  }

  state.activeStressIdx = activeIdx;
  renderSingleStressChart(activeIdx);
}

/* =========================================================================
   STEP 6 RENDERER: EMPIRICAL HYPOTHESIS SCORECARD & JUPYTER NOTEBOOK PDF
   ========================================================================= */
function renderScorecardAndCalculations(report) {
  const sc = report.scorecard;
  if (!sc) return;

  const verdictLabel = sc.overall_verdict_label || sc.historical_verdict_label;
  const badge = document.getElementById('scorecard-verdict-badge');
  if (badge) {
    badge.style.display = 'inline-flex';
    badge.textContent = verdictLabel;
    badge.className =
      verdictLabel === 'SUPPORTED BY HISTORICAL DATA'
        ? 'panel-badge emerald'
        : verdictLabel === 'CONTRADICTED BY HISTORICAL DATA' || verdictLabel.includes('HALTED')
        ? 'panel-badge badge-vulnerable'
        : 'panel-badge amber';
  }

  const evalCount = sc.evaluated_checks_count ?? sc.total_checks_count;
  const totalTouches = (report.spec?.prior_holdout_touches ?? 0) + 1;
  const stageLabel = (report.spec?.evaluation_stage || 'confirmatory').replace(/_/g, ' ');
  document.getElementById('scorecard-hypothesis-title').textContent = `"${sc.hypothesis_restatement}"`;
  document.getElementById('scorecard-checks-counter').textContent =
    `${sc.supported_checks_count} / ${evalCount} Evaluated Checks Supported · Holdout Touches: ${totalTouches} (${stageLabel})`;
  document.getElementById('scorecard-exec-summary').textContent = sc.executive_summary;

  document.getElementById('scorecard-takeaways').innerHTML = (sc.key_takeaways || [])
    .map((k) => `<li>${escapeHtml(k)}</li>`)
    .join('');

  // Render 3-Lens Hypothesis Comparison Breakdown if present (Return Co-Movement vs Cointegration vs Trading Rule P&L)
  const compContainer = document.getElementById('hypothesis-comparison-container');
  if (compContainer) {
    const breakdown = sc.hypothesis_comparison_breakdown || [];
    if (breakdown.length > 0) {
      compContainer.style.display = 'block';
      compContainer.innerHTML = `
        <div class="section-kicker" style="margin-bottom:10px;">
          Multi-Question Decomposition (Co-Movement vs. Cointegration vs. Trading Rule P&amp;L)
        </div>
        <div class="hypothesis-comparison-grid">
          ${breakdown
            .map((b) => {
              const title = b.lens_title || b.aspect || '';
              const status = b.status || b.verdict || '';
              const question = b.question_answered || '';
              const metric = b.primary_metric || b.evidence || '';
              const interp = b.interpretation || b.explanation || '';
              return `
            <div class="hypothesis-lens-card">
              <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:6px;">
                <strong style="font-size:13px; color:var(--text-primary);">${escapeHtml(title)}</strong>
                <span class="panel-badge ${status === 'SUPPORTED' ? 'emerald' : status === 'CONTRADICTED' ? 'badge-vulnerable' : 'amber'}">${escapeHtml(status)}</span>
              </div>
              ${question ? `<div style="font-size:12px; color:var(--text-muted); margin-bottom:6px;">${escapeHtml(question)}</div>` : ''}
              <div style="font-family:var(--font-mono); font-size:12px; color:var(--accent-cyan); margin-bottom:6px;">${escapeHtml(metric)}</div>
              <div style="font-size:12px; color:var(--text-secondary); line-height:1.5;">${escapeHtml(interp)}</div>
            </div>
          `;
            })
            .join('')}
        </div>
      `;
    } else {
      compContainer.style.display = 'none';
    }
  }

  // Render the 4-Card At-a-Glance Plain-English Clarity Strip
  const clarityStrip = document.getElementById('step6-clarity-strip');
  if (clarityStrip) {
    const spec = report.spec;
    const fullM = report.full_sample_metrics;
    const oosM = report.out_of_sample_metrics;
    const fac = report.factor_regression;
    const assetsLabel = spec.asset_b ? `${spec.asset_a} vs. ${spec.asset_b}` : spec.asset_a;
    const beCost = report.break_even_cost_bps !== null && report.break_even_cost_bps !== undefined ? `${fmtNum(report.break_even_cost_bps, 1)} bps/leg` : '0.0 bps/leg';
    const oosSummaryStr = oosM.is_zero_exposure
      ? `${fmtPct(oosM.total_net_return_pct)} (${oosM.status_note || '0 held-out trades'})`
      : `${fmtPct(oosM.total_net_return_pct)} (Sharpe ${fmtNum(oosM.sharpe_ratio, 2)}, ${oosM.trade_count} trades)`;

    clarityStrip.innerHTML = `
      <div class="clarity-card">
        <div class="clarity-card-kicker">1. WHAT WAS TESTED</div>
        <div class="clarity-card-main">${escapeHtml(assetsLabel)} (${escapeHtml(spec.signal_variant)})</div>
        <div class="clarity-card-sub">
          <strong>${fullM.trade_count} trades</strong> across ${fullM.trading_days} trading days (${fullM.start_date} to ${fullM.end_date})
        </div>
      </div>
      <div class="clarity-card">
        <div class="clarity-card-kicker">2. PERFORMANCE VS. BENCHMARK</div>
        <div class="clarity-card-main ${returnColorClassVsBenchmark(fullM.total_net_return_pct, fullM.benchmark_return_pct)}">
          ${fmtPct(fullM.total_net_return_pct)} Net (Sharpe ${fmtNum(fullM.sharpe_ratio, 2)})
        </div>
        <div class="clarity-card-sub">
          vs. ${escapeHtml(spec.benchmark_ticker)} Buy &amp; Hold: <strong>${fmtPct(fullM.benchmark_return_pct)}</strong> (Excess ${fmtPp(fullM.excess_return_pct)})
        </div>
      </div>
      <div class="clarity-card">
        <div class="clarity-card-kicker">3. RISK &amp; HOLDOUT PERSISTENCE</div>
        <div class="clarity-card-main val-neg">
          Max DD -${fmtNum(fullM.max_drawdown_pct)}% | Win ${fmtNum(fullM.win_rate_pct, 1)}%
        </div>
        <div class="clarity-card-sub">
          Held-Out OOS: <strong>${escapeHtml(oosSummaryStr)}</strong>
        </div>
      </div>
      <div class="clarity-card">
        <div class="clarity-card-kicker">4. WHY THIS VERDICT</div>
        <div class="clarity-card-main" style="color:var(--accent-cyan);">
          ${sc.supported_checks_count} of ${evalCount} Evaluated Checks Supported
        </div>
        <div class="clarity-card-sub">
          HAC Alpha: <strong>${fmtPct(fac.annualized_alpha_pct)}</strong> (p=${fmtNum(fac.alpha_p_value_hac, 3)}) | Break-Even: <strong>${beCost}</strong>
        </div>
      </div>
    `;
  }

  document.getElementById('scorecard-dimensions-grid').innerHTML = (sc.dimensions || [])
    .map((d) => {
      const cls =
        d.status === 'SUPPORTED'
          ? 'badge-resilient'
          : d.status === 'MIXED'
          ? 'badge-moderate'
          : d.status === 'NO_EVIDENCE' || d.status === 'NEUTRAL'
          ? 'badge-neutral'
          : 'badge-vulnerable';
      return `
        <div class="dimension-card">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:6px;">
            <span style="font-family:var(--font-sans); font-weight:600; font-size:13px; color:var(--text-primary);">${escapeHtml(d.name)}</span>
            <span class="panel-badge ${cls}">${escapeHtml(d.status.replace(/_/g, ' '))}</span>
          </div>
          <div style="font-size:12px; color:var(--text-muted); margin-bottom:6px;">${escapeHtml(d.question)}</div>
          <div style="font-family:var(--font-mono); font-size:12px; color:var(--accent-cyan); margin-bottom:6px;">
            ${escapeHtml(d.evidence_metric)}
          </div>
          <div style="font-size:12px; color:var(--text-secondary);">${escapeHtml(d.explanation)}</div>
        </div>
      `;
    })
    .join('');

  renderJupyterNotebookPreview(report);

  // Handoff Section 7: Expandable "How is this calculated?" mathematical derivation cards
  const calcBox = document.getElementById('calculation-steps-container');
  if (calcBox) {
    const steps = sc.calculation_steps || [];
    if (steps.length > 0) {
      calcBox.innerHTML = `
        <div class="section-kicker" style="margin-bottom:8px;">
          STEP-BY-STEP MATHEMATICAL DERIVATIONS (EXPAND FOR FORMULA &amp; SUBSTITUTED INPUTS)
        </div>
        <div style="display:flex; flex-direction:column; gap:8px;">
          ${steps
            .map(
              (step, idx) => `
            <details class="calc-step-details" ${idx === 0 ? 'open' : ''} style="border:1px solid var(--border-subtle); border-radius:var(--radius-sm); background:var(--bg-panel-elevated); padding:10px 14px;">
              <summary style="cursor:pointer; display:flex; justify-content:space-between; align-items:center; gap:12px; font-weight:600; font-size:13px; color:var(--text-primary); list-style:none;">
                <span>${escapeHtml(step.title)} <span style="font-weight:400; font-size:11.5px; color:var(--text-muted);">— How is this calculated?</span></span>
                <span style="font-family:var(--font-mono); font-size:12px; color:var(--accent-emerald);">${escapeHtml(step.output_value)}</span>
              </summary>
              <div style="margin-top:10px; padding-top:10px; border-top:1px solid var(--border-subtle); font-size:12.5px; display:flex; flex-direction:column; gap:6px;">
                <div>
                  <span style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted); text-transform:uppercase;">Mathematical Definition (Daily Frequency, 252 Bars/Yr):</span>
                  <div class="formula-banner" style="margin-top:4px; padding:8px 10px; font-size:12px;">${formatMathHtml(step.formula)}</div>
                </div>
                <div>
                  <span style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted); text-transform:uppercase;">Substituted Values From This Run:</span>
                  <div style="font-family:var(--font-mono); font-size:12px; color:var(--text-primary); margin-top:2px;">${formatMathHtml(step.substituted_inputs)}</div>
                </div>
                <div>
                  <span style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted); text-transform:uppercase;">Plain-Language Interpretation:</span>
                  <div style="color:var(--text-secondary); margin-top:2px; line-height:1.5;">${escapeHtml(step.interpretation)}</div>
                </div>
              </div>
            </details>
          `
            )
            .join('')}
        </div>
      `;
    } else {
      calcBox.innerHTML = '';
    }
  }

  document.getElementById('methodology-notes-list').innerHTML = (report.methodology_notes || [])
    .map((m) => `<li>${escapeHtml(m)}</li>`)
    .join('');
  document.getElementById('limitations-list').innerHTML = (report.limitations || [])
    .map((l) => `<li>${escapeHtml(l)}</li>`)
    .join('');
}

function renderJupyterNotebookPreview(report) {
  const nbBox = document.getElementById('jupyter-notebook-preview');
  if (!nbBox) return;

  const spec = report.spec;
  const isM = report.in_sample_metrics;
  const oosM = report.out_of_sample_metrics;
  const fullM = report.full_sample_metrics;
  const fac = report.factor_regression;
  const sc = report.scorecard;
  const st = report.stress_test_suite;
  const prov = report.data_provenance || {};
  const alignedSha = prov.aligned_slice_sha256 || report.replay_manifest?.aligned_slice_sha256 || prov.asset_a?.sha256 || '';
  const engineVerStr =
    typeof report.engine_version === 'object' && report.engine_version !== null
      ? `${report.engine_version.engine_release || '2.0.0'} (${report.engine_version.code_sha256 || ''})`
      : report.engine_version || '2.0.0';

  const calcStepsOut = (sc?.calculation_steps || [])
    .map(
      (step) =>
        `${escapeHtml(step.title)}\n` +
        `  Equation : ${escapeHtml(step.formula)}\n` +
        `  Inputs   : ${escapeHtml(step.substituted_inputs)}\n` +
        `  Result   : ${escapeHtml(step.output_value)}\n` +
        `  Meaning  : ${escapeHtml(step.interpretation)}`
    )
    .join('\n' + '-'.repeat(88) + '\n');

  const stressOut = (st?.tests || [])
    .map(
      (t, i) =>
        `Test 0${i + 1} [${escapeHtml(t.outcome_badge.padEnd(20, ' '))}] ${escapeHtml(t.title)}\n` +
        `  Baseline: ${escapeHtml(t.baseline_value)} -> Stressed: ${escapeHtml(t.stressed_value)}`
    )
    .join('\n');

  const oosSrPreview = oosM.is_zero_exposure
    ? `N/A (0 trades, ${escapeHtml(oosM.status_note || 'Zero Exposure')})`
    : `${fmtNum(oosM.sharpe_ratio, 3)} (±${fmtNum(oosM.sharpe_se_lo2002, 3)})`;

  nbBox.innerHTML = `
    <div class="nb-cell">
      <div class="nb-prompt-in">In [1]:</div>
      <div class="nb-code-box"># 1. Load Audited StrategySpec &amp; Verify Cryptographic Provenance
from pathlib import Path
from engine.backtester import load_audit_run, run_full_audit

report = load_audit_run(Path("data/runs/${escapeHtml(report.run_id)}.json"))
spec = report.spec
print(f"Run ID: {report.run_id} | Engine: {report.engine_version.get('engine_release', '2.0.0')} | Market As-Of: {report.market_as_of_date}")</div>
    </div>
    <div class="nb-cell">
      <div class="nb-prompt-out">Out [1]:</div>
      <div class="nb-out-box">Run ID: ${escapeHtml(report.run_id)} | Engine: ${escapeHtml(engineVerStr)} | Market As-Of: ${escapeHtml(report.market_as_of_date || fullM.end_date)}
Spec SHA-256: ${escapeHtml(report.spec_sha256)} | Aligned Slice SHA-256: ${escapeHtml(alignedSha)}
Chronological Split: Formation (${isM.start_date}..${isM.end_date}, ${isM.trading_days}d) | Held-Out OOS (${oosM.start_date}..${oosM.end_date}, ${oosM.trading_days}d)</div>
    </div>
    <div class="nb-cell">
      <div class="nb-prompt-in">In [2]:</div>
      <div class="nb-code-box"># 2. Re-Execute Strictly Causal Walk-Forward Backtest &amp; Compounded Trade Reconciliation
replayed = run_full_audit(spec)
assert replayed.run_id == report.run_id
assert abs(spec.execution.initial_capital * np.prod([1 + t.net_return_pct / 100 for t in replayed.trades]) - replayed.timeseries["equity_net"][-1]) &lt; 5e-2</div>
    </div>
    <div class="nb-cell">
      <div class="nb-prompt-out">Out [2]:</div>
      <div class="nb-out-box">In-Sample Formation : Trades=${isM.trade_count} | Net=${fmtPct(isM.total_net_return_pct)} | Sharpe=${fmtNum(isM.sharpe_ratio, 3)} (±${fmtNum(isM.sharpe_se_lo2002, 3)}) | MaxDD=-${fmtNum(isM.max_drawdown_pct)}%
Held-Out OOS        : Trades=${oosM.trade_count} | Net=${fmtPct(oosM.total_net_return_pct)} | Sharpe=${oosSrPreview} | MaxDD=-${fmtNum(oosM.max_drawdown_pct)}%
Full Sample         : Trades=${fullM.trade_count} | Net=${fmtPct(fullM.total_net_return_pct)} | Sharpe=${fmtNum(fullM.sharpe_ratio, 3)} (PSR=${fmtNum(fullM.probabilistic_sharpe_psr_pct, 1)}%) | Alpha=${fmtPct(fac.annualized_alpha_pct)} (t=${fmtNum(fac.alpha_t_stat_hac, 2)})</div>
    </div>
    <div class="nb-cell">
      <div class="nb-prompt-in">In [3]:</div>
      <div class="nb-code-box"># 3. Step-by-Step Mathematical Estimators &amp; Substituted Run Inputs
from engine.econometrics import lo_mertens_sharpe_se_and_psr, deflated_sharpe_ratio_bailey_lopez, newey_west_factor_regression, stationary_bootstrap_cis

sr, se_lo2002, psr, skew, kurt = lo_mertens_sharpe_se_and_psr(daily_net_returns)
exp_max_sr_null, dsr = deflated_sharpe_ratio_bailey_lopez(daily_net_returns, k25_sharpes)
hac_reg = newey_west_factor_regression(daily_net_returns, daily_benchmark_returns)</div>
    </div>
    <div class="nb-cell">
      <div class="nb-prompt-out">Out [3]:</div>
      <div class="nb-out-box">${calcStepsOut}</div>
    </div>
    <div class="nb-cell">
      <div class="nb-prompt-in">In [4]:</div>
      <div class="nb-code-box"># 4. 5-Test Robustness Suite Summary (Walk-Forward, Cost/Latency Shock, 25-Cell Grid, Non-Overlapping MC Null, Stationarity/DSR)
for idx, test in enumerate(replayed.stress_test_suite.tests, 1):
    print(f"Test 0{idx} [{test.outcome_badge}] {test.title}: {test.baseline_value} -&gt; {test.stressed_value}")</div>
    </div>
    <div class="nb-cell">
      <div class="nb-prompt-out">Out [4]:</div>
      <div class="nb-out-box">${stressOut}</div>
    </div>
  `;
}

async function downloadNotebookPDF() {
  if (!state.currentReport) {
    alert('Please confirm your StrategySpec and run a research audit in Step 2 first before downloading the Jupyter Notebook PDF report.');
    return;
  }

  try {
    let res;
    if (state.currentReport && state.currentReport.run_id) {
      res = await fetch(`/api/audit/${encodeURIComponent(state.currentReport.run_id)}/pdf`);
    }
    if (!res || !res.ok) {
      res = await fetch('/api/audit/pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.currentReport),
      });
    }
    if (!res.ok) {
      throw new Error('Failed to generate Jupyter Notebook PDF');
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const runId = state.currentReport.run_id || 'alpha-audit';
    a.download = `${runId}-jupyter-notebook-report.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(`PDF Download Error: ${err.message}`);
  }
}

async function downloadNotebookIpynb() {
  if (!state.currentReport) {
    alert('Please confirm your StrategySpec and run a research audit in Step 2 first before downloading the executable Calculation Notebook (.ipynb).');
    return;
  }

  try {
    let res;
    if (state.currentReport && state.currentReport.run_id) {
      res = await fetch(`/api/audit/${encodeURIComponent(state.currentReport.run_id)}/ipynb`);
    }
    if (!res || !res.ok) {
      res = await fetch('/api/audit/ipynb', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.currentReport),
      });
    }
    if (!res.ok) {
      throw new Error('Failed to generate executable Jupyter Notebook (.ipynb)');
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const runId = state.currentReport.run_id || 'alpha-audit';
    a.download = `${runId}-calculation-notebook.ipynb`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(`Notebook (.ipynb) Download Error: ${err.message}`);
  }
}

function renderProvenanceAndBlotter(report) {
  const prov = report.data_provenance || {};
  const aMeta = prov.asset_a || {};
  const bMeta = prov.asset_b;
  const alignedSha = prov.aligned_slice_sha256 || report.replay_manifest?.aligned_slice_sha256 || aMeta.sha256 || '';
  const engineRelease =
    typeof report.engine_version === 'object' && report.engine_version !== null
      ? report.engine_version.engine_release || '2.0.0'
      : report.engine_version || '2.0.0';

  document.getElementById('provenance-grid').innerHTML = `
    <div class="kpi-card">
      <div class="kpi-label">AUDIT RUN ID, ENGINE &amp; SPEC SHA-256</div>
      <div style="font-family:var(--font-mono); font-size:12px; color:var(--text-primary); word-break:break-all;">
        <strong>Run:</strong> ${escapeHtml(report.run_id)} (Schema v${escapeHtml(report.schema_version || '2.0')}, Engine v${escapeHtml(engineRelease)})<br/>
        <strong>Spec Hash:</strong> ${escapeHtml((report.spec_sha256 || '').slice(0, 24))}...<br/>
        <strong>Market As-Of:</strong> ${escapeHtml(report.market_as_of_date || report.full_sample_metrics.end_date)} (Deterministic)
      </div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">DATASET PROVENANCE (${escapeHtml(aMeta.ticker || report.spec.asset_a)})</div>
      <div style="font-family:var(--font-mono); font-size:12px; color:var(--text-primary); word-break:break-all;">
        <strong>Source:</strong> ${escapeHtml(aMeta.source || 'Historical Daily OHLCV')}<br/>
        <strong>Aligned Slice SHA-256:</strong> ${escapeHtml(alignedSha.slice(0, 24))}...<br/>
        ${bMeta ? `<strong>Asset B (${escapeHtml(bMeta.ticker)}) SHA-256:</strong> ${escapeHtml((bMeta.sha256 || '').slice(0, 20))}...` : `<strong>Aligned Days:</strong> ${prov.aligned_trading_days} bars`}
      </div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">EXECUTION &amp; RISK ENFORCEMENT</div>
      <div style="font-family:var(--font-mono); font-size:12px; color:var(--text-primary);">
        <strong>Fill Rule:</strong> Bar t+1 (${escapeHtml(report.spec.execution.execution_timing)})<br/>
        <strong>Friction:</strong> ${report.spec.execution.commission_bps} bps comm + ${report.spec.execution.slippage_bps} bps slip / leg<br/>
        <strong>Circuit Breaker:</strong> ${report.full_sample_metrics.circuit_breaker_triggered ? `TRIPPED (${escapeHtml(report.full_sample_metrics.circuit_breaker_date)})` : 'ARMED (Not Tripped)'}
      </div>
    </div>
  `;

  renderTradesBlotter(report.trades);
}

function renderTradesBlotter(trades) {
  const splitFilter = document.getElementById('filter-trade-split').value;
  const filtered =
    splitFilter === 'all' ? trades : trades.filter((t) => t.split === splitFilter);

  const tbody = document.querySelector('#trades-blotter-table tbody');
  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="16" style="text-align:center; color:var(--text-muted); padding:20px;">No historical trades triggered in this partition under the current threshold specification.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered
    .map(
      (t) => `
      <tr>
        <td>${t.trade_id}</td>
        <td><span class="panel-badge ${t.split === 'out_of_sample' ? 'emerald' : 'cyan'}">${t.split === 'out_of_sample' ? 'HELD-OUT OOS' : 'IN-SAMPLE'}${t.crossed_split ? ' (CROSSED)' : ''}</span></td>
        <td>${escapeHtml(t.direction.toUpperCase())}</td>
        <td>${escapeHtml(t.signal_date)}</td>
        <td class="num">${fmtNum(t.signal_value, 3)}</td>
        <td>${escapeHtml(t.entry_date)}</td>
        <td class="num">$${fmtNum(t.entry_price_a, 2)}</td>
        <td class="num">${t.entry_price_b ? `$${fmtNum(t.entry_price_b, 2)}` : '—'}</td>
        <td>${escapeHtml(t.exit_date)}</td>
        <td class="num">$${fmtNum(t.exit_price_a, 2)}</td>
        <td class="num">${t.exit_price_b ? `$${fmtNum(t.exit_price_b, 2)}` : '—'}</td>
        <td class="num">${t.holding_days}d</td>
        <td class="num ${colorClass(t.gross_return_pct)}">${fmtPct(t.gross_return_pct)}</td>
        <td class="num val-neg">-${fmtNum(t.cost_pct, 2)}%</td>
        <td class="num ${colorClass(t.net_return_pct)}" style="font-weight:700;">${fmtPct(t.net_return_pct)}</td>
        <td style="color:var(--text-secondary);">${escapeHtml(t.exit_reason)}</td>
      </tr>
    `
    )
    .join('');
}

function exportTradesToCSV(trades, runId) {
  if (!trades || !trades.length) return;
  const headers = [
    'trade_id',
    'split',
    'crossed_split',
    'direction',
    'signal_date',
    'signal_value',
    'entry_date',
    'entry_price_a',
    'entry_price_b',
    'leg_weight_a',
    'leg_weight_b',
    'exit_date',
    'exit_price_a',
    'exit_price_b',
    'holding_days',
    'gross_return_pct',
    'cost_pct',
    'net_return_pct',
    'exit_reason',
  ];
  const lines = [
    headers.join(','),
    ...trades.map((t) => headers.map((h) => JSON.stringify(t[h] ?? '')).join(',')),
  ];
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${runId}-trades.csv`;
  a.click();
}

async function loadChallengeEvidence() {
  try {
    const res = await fetch('/api/challenge/summary');
    const data = await res.json();
    const box = document.getElementById('challenge-evidence-container');
    if (!box) return;
    const blocks = [
      `<div style="font-size:12px; color:var(--text-muted); margin-bottom:10px;">
        <strong>Component Scope:</strong> This 120-case benchmark evaluates the standalone causal walk-forward decision stepper (<code>engine/walk_forward_stepper.py</code> invoked via <code>adapter.py</code> on <code>task="market_step"</code>). It is an independent execution benchmark and is separate from the 6-step historical strategy audit report above.
      </div>`,
    ];

    for (const [key, label] of [
      ['real_markets', 'REAL-MARKETS CHALLENGE (60 UNSEEN BINANCE ARCHIVE PATHS — 50% INVERTED)'],
      ['synthetic_markets', 'SYNTHETIC MARKETS CHALLENGE (60 UNSEEN SCENARIO PATHS)'],
    ]) {
      const item = data[key]?.summary;
      if (!item) continue;
      const ms = item.market_summary || {};
      const ch = item.challenge || {};
      const warningsCount = (ch.provenance?.warnings || []).length;
      blocks.push(`
        <div class="kpi-card" style="margin-bottom:10px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-family:var(--font-mono); font-weight:700; color:var(--accent-cyan);">${label}</span>
            <span class="panel-badge ${warningsCount === 0 ? 'emerald' : 'amber'}">PROVENANCE WARNINGS: ${warningsCount}</span>
          </div>
          <div style="font-family:var(--font-mono); font-size:12px; color:var(--text-primary);">
            Completed: <strong>${ms.completed || 60}/60</strong> &nbsp;|&nbsp;
            Wins vs Buy &amp; Hold: <strong>${ms.wins_vs_buy_and_hold}W - ${ms.losses_vs_buy_and_hold}L</strong> (p = ${ms.win_rate_p_value}) &nbsp;|&nbsp;
            Median Excess Return: <strong class="val-pos">+${ms.median_excess_return_bps} bps</strong> &nbsp;|&nbsp;
            Median Sharpe: <strong>${ms.median_sharpe}</strong> (vs B&amp;H ${ms.median_buy_and_hold_sharpe})
          </div>
        </div>
      `);
    }

    if (blocks.length > 1) {
      box.innerHTML = blocks.join('');
    }
  } catch (err) {
    // Ignore if challenge summary not yet generated
  }
}

document.addEventListener('DOMContentLoaded', initWorkbench);
