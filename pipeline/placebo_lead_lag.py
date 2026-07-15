"""
Placebo test: leads vs lags (Granger causality check).

Tests whether FUTURE media attention (t+1, t+2) predicts PAST ODA (t).
If the model is well-specified this should be zero. If PAST media (t-1, t-2)
predicts FUTURE ODA, but not vice versa, this establishes Granger causality.

Specification:
  ODA_share_{i,t} = alpha_i + gamma_t + beta * Media_{i,t+k} + e_{i,t}

  where alpha_i = SDG fixed effects, gamma_t = year fixed effects
  k < 0 = lags (causal direction), k > 0 = leads (placebo)

Data: src/oda/panel_prelim.csv  (keyword-based proxy, 2010-2016)
"""

from __future__ import annotations

import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR  = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "src" / "oda" / "panel_prelim.csv"
OUT_DIR   = BASE_DIR / "src" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── OLS with SDG + year fixed effects ─────────────────────────────────────────

def fe_ols(
    data: pd.DataFrame,
    y_col: str,
    x_col: str,
    sdg_col: str = "sdg",
    year_col: str = "year",
) -> dict:
    """
    OLS with two-way fixed effects (SDG + year) via dummy absorption.
    Returns coefficient on x_col with heteroskedasticity-robust SE.
    """
    sub = data[[y_col, x_col, sdg_col, year_col]].dropna().copy()
    if len(sub) < 10:
        return {}

    n = len(sub)

    sdg_dummies  = pd.get_dummies(sub[sdg_col],  prefix="sdg", drop_first=True).astype(float)
    year_dummies = pd.get_dummies(sub[year_col], prefix="yr",  drop_first=True).astype(float)

    X = np.column_stack([
        sub[x_col].values,
        sdg_dummies.values,
        year_dummies.values,
        np.ones(n),
    ])
    y = sub[y_col].values

    # OLS
    coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coefs
    resid  = y - y_hat
    k_vars = X.shape[1]
    df_r   = n - k_vars

    # HC1 heteroskedasticity-robust covariance (sandwich estimator)
    XtXinv   = np.linalg.pinv(X.T @ X)
    meat     = X.T @ np.diag(resid ** 2) @ X
    cov_hc1  = (n / df_r) * (XtXinv @ meat @ XtXinv)
    se_robust = np.sqrt(np.diag(cov_hc1))

    coef   = coefs[0]
    se_x   = se_robust[0]
    t_stat = coef / se_x if se_x > 0 else np.nan
    p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_r)) if not np.isnan(t_stat) else np.nan

    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)

    return {
        "coef":   coef,
        "se":     se_x,
        "t_stat": t_stat,
        "p_val":  p_val,
        "n":      n,
        "df_r":   df_r,
        "r2":     r2,
    }


# ── Load and prepare ──────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
df_full    = df.copy()                        # includes partial 2016
df_clean   = df[~df["partial"]].copy()        # 2010-2015 only (6 full years)

print("=" * 65)
print("PLACEBO TEST: LEADS vs LAGS")
print("Granger Causality — Korean Media Attention → Korean ODA")
print("=" * 65)
print(f"\nFull panel:  {len(df_full):3d} obs  (2010-2016, 2016 partial)")
print(f"Clean panel: {len(df_clean):3d} obs  (2010-2015, full years only)")
print(f"\nSpecification: ODA_share_i,t = SDG_FE + Year_FE + β·Media_i,t+k")
print(f"SE: HC1 heteroskedasticity-robust\n")


# ── Create leads and lags ─────────────────────────────────────────────────────

for panel_df, label in [(df_clean, "CLEAN (2010-2015)"), (df_full, "FULL (2010-2016)")]:
    panel_df = panel_df.sort_values(["sdg", "year"]).copy()
    grp = panel_df.groupby("sdg")["media_share_pct"]

    for k in [1, 2, 3]:
        panel_df[f"media_L{k}"] = grp.shift(k)    # LAG: past media at t-k
        panel_df[f"media_F{k}"] = grp.shift(-k)   # LEAD: future media at t+k

    print(f"\n{'─'*65}")
    print(f"  {label}")
    print(f"{'─'*65}")
    print(f"  {'Term':<24} {'Coef':>8} {'SE':>8} {'t':>7} {'p':>8} {'N':>5}  Sig")
    print(f"  {'-'*60}")

    rows = []

    # Contemporaneous
    res = fe_ols(panel_df, "oda_share_pct", "media_share_pct")
    if res:
        sig = "***" if res["p_val"] < 0.01 else "**" if res["p_val"] < 0.05 else "*" if res["p_val"] < 0.10 else ""
        print(f"  {'Media_t  (contemp.)':<24} {res['coef']:>8.3f} {res['se']:>8.3f} {res['t_stat']:>7.2f} {res['p_val']:>8.4f} {res['n']:>5}  {sig}")
        rows.append({"panel": label, "term": "L0 (contemp)", "k": 0, "direction": "contemp", **res, "sig": sig})

    # Lags (causal hypothesis: past media → future ODA)
    for k in [1, 2, 3]:
        res = fe_ols(panel_df, "oda_share_pct", f"media_L{k}")
        if res:
            sig = "***" if res["p_val"] < 0.01 else "**" if res["p_val"] < 0.05 else "*" if res["p_val"] < 0.10 else ""
            label_str = f"Media_t-{k} (lag {k}yr)"
            print(f"  {label_str:<24} {res['coef']:>8.3f} {res['se']:>8.3f} {res['t_stat']:>7.2f} {res['p_val']:>8.4f} {res['n']:>5}  {sig}")
            rows.append({"panel": label, "term": f"L{k} (lag {k}yr)", "k": -k, "direction": "lag", **res, "sig": sig})

    # Leads (PLACEBO: future media should NOT predict past ODA)
    for k in [1, 2, 3]:
        res = fe_ols(panel_df, "oda_share_pct", f"media_F{k}")
        if res:
            sig = "***" if res["p_val"] < 0.01 else "**" if res["p_val"] < 0.05 else "*" if res["p_val"] < 0.10 else ""
            label_str = f"Media_t+{k} (lead {k}yr) ← PLACEBO"
            print(f"  {label_str:<24} {res['coef']:>8.3f} {res['se']:>8.3f} {res['t_stat']:>7.2f} {res['p_val']:>8.4f} {res['n']:>5}  {sig}")
            rows.append({"panel": label, "term": f"F{k} (lead {k}yr)", "k": k, "direction": "lead", **res, "sig": sig})

    # --- Granger summary ---
    lag_rows  = [r for r in rows if r["direction"] == "lag"  and r["panel"] == label]
    lead_rows = [r for r in rows if r["direction"] == "lead" and r["panel"] == label]

    if lag_rows and lead_rows:
        lag_sig  = sum(1 for r in lag_rows  if r["p_val"] < 0.10)
        lead_sig = sum(1 for r in lead_rows if r["p_val"] < 0.10)
        print(f"\n  Lag coefficients significant (p<0.10):  {lag_sig}/{len(lag_rows)}")
        print(f"  Lead coefficients significant (p<0.10): {lead_sig}/{len(lead_rows)}")
        if lag_sig > 0 and lead_sig == 0:
            print(f"\n  Granger causality holds: lags predict, leads do not.")
        elif lead_sig > 0:
            print(f"\n  Warning: {lead_sig} lead(s) are significant — check for omitted variable bias.")
        else:
            print(f"\n  Neither lags nor leads are significant (insufficient power or no effect).")

# ── Save results ──────────────────────────────────────────────────────────────

all_rows = []
for panel_df, label in [(df_clean, "CLEAN (2010-2015)"), (df_full, "FULL (2010-2016)")]:
    panel_df = panel_df.sort_values(["sdg", "year"]).copy()
    grp = panel_df.groupby("sdg")["media_share_pct"]
    for k in [1, 2, 3]:
        panel_df[f"media_L{k}"] = grp.shift(k)
        panel_df[f"media_F{k}"] = grp.shift(-k)
    for x_col, direction, k in (
        [("media_share_pct", "contemp", 0)] +
        [(f"media_L{k}", "lag", k) for k in [1,2,3]] +
        [(f"media_F{k}", "lead", k) for k in [1,2,3]]
    ):
        res = fe_ols(panel_df, "oda_share_pct", x_col)
        if res:
            all_rows.append({"panel": label, "direction": direction, "k": k, **res})

out_df = pd.DataFrame(all_rows)
out_path = OUT_DIR / "placebo_lead_lag.csv"
out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n{'='*65}")
print(f"Results saved → {out_path.relative_to(BASE_DIR)}")
print(f"{'='*65}")
