#!/usr/bin/env python3
"""
FOH Predictor (per-server + shift-combo)

Features
--------
- Accepts either:
  A) Per-server CSV: one row per server per shift with server_sales
  B) Shift-level CSV: one row per shift with 6 servers and total sales
- Trains TWO models in one run:
  (1) Per-server model -> predicts each server's individual sales for a scenario.
      Optional leakage-safe target encoding for server×DOW/meal (--target-encode).
  (2) Shift-combo model -> predicts (host, manager, team-of-6) totals, ranks lineups,
      assigns roles (Closer/Mid/First-Cut), optional week plan + schedule export.

Install
-------
pip install pandas numpy scikit-learn scipy matplotlib
"""

import argparse
import itertools
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

# Optional plotting
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ---------------- Defaults (overridable by CLI) ----------------
DEFAULTS = {
    "csv": "shift_sales_6servers.csv",    # input file (auto-detect schema)
    "events_csv": None,                   # optional (shift-level only)
    "output_dir": "outputs",

    # common columns
    "date_col": "date",
    "host_col": "host",
    "manager_col": "manager",
    "meal_col": "meal",
    "rain_col": "rain",
    "dish_col": "dishwasher_down",
    "inv_col": "low_inventory",
    "sales_col": "sales",

    # per-server specific
    "server_col": "server",
    "server_sales_col": "server_sales",

    # shift-level specific
    "server_cols": ["server1","server2","server3","server4","server5","server6"],

    # modeling
    "use_random_forest": True,
    "rf_estimators": 200,
    "tree_max_depth": 5,
    "test_size": 0.2,

    # combo generation
    "max_candidate_combos": 50_000,
    "top_n": 25,

    # scenario flags
    "score_dow": None,           # None -> use mode of data
    "score_event_day": 0,
    "score_rain": 0,
    "score_meal": "Dinner",      # Lunch or Dinner
    "score_dish": 0,
    "score_inv": 0,

    # availability filters (shift model)
    "must_include_servers": [],
    "exclude_servers": [],
    "restrict_hosts_to": [],
    "restrict_managers_to": [],

    # fairness (shift model)
    "enable_fairness_penalty": True,
    "recently_overused": [],
    "fairness_penalty": 75.0,

    # plotting
    "plot_top10": False,
    "plot_filename": "top_combos_top10.png",

    # week plan
    "make_week_plan": False,

    # role assignment
    "closer_count": 2,
    "firstcut_count": 2,

    # extras
    "target_encode": False,   # per-server leakage-safe target encoding
    "export_schedule": False, # write a human-friendly schedule with roles (requires --week-plan)
}

# ---------------- Utilities ----------------
def _is_per_server(df: pd.DataFrame, cfg: dict) -> bool:
    return all(c in df.columns for c in [cfg["server_col"], cfg["server_sales_col"]])

def _make_ohe_backward_compatible():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)  # sklearn ≥1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)         # sklearn <1.2

def _read_csv_robust(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", sep=",", on_bad_lines="skip")

def parse_date_and_dow(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["dow"] = df[date_col].dt.dayofweek
    return df

# ---------------- PER-SERVER MODEL ----------------
def load_per_server(path: str, cfg: dict) -> pd.DataFrame:
    df = _read_csv_robust(path)
    need = [cfg["date_col"], cfg["meal_col"], cfg["host_col"], cfg["manager_col"],
            cfg["server_col"], cfg["server_sales_col"], cfg["rain_col"], cfg["dish_col"], cfg["inv_col"]]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Per-server CSV missing columns: {miss}")
    df = parse_date_and_dow(df, cfg["date_col"])
    return df

def _build_te_maps(df: pd.DataFrame, cfg: dict) -> dict:
    te = {}
    te["global"] = df[cfg["server_sales_col"]].mean()
    te["server"] = df.groupby(cfg["server_col"])[cfg["server_sales_col"]].mean().to_dict()
    te["server_dow"] = df.groupby([cfg["server_col"], "dow"])[cfg["server_sales_col"]].mean().to_dict()
    te["server_meal"] = df.groupby([cfg["server_col"], cfg["meal_col"]])[cfg["server_sales_col"]].mean().to_dict()
    return te

def _apply_te_row(server, dow, meal, te_maps):
    g = te_maps["global"]
    s = te_maps["server"].get(server, g)
    sd = te_maps["server_dow"].get((server, dow), s)
    sm = te_maps["server_meal"].get((server, meal), s)
    # Blend (tunable weights)
    return 0.2*s + 0.4*sd + 0.4*sm

def _make_target_encoded_matrix(df: pd.DataFrame, cfg: dict, ohe: OneHotEncoder):
    # Prepare columns
    cat_cols = [cfg["meal_col"], cfg["host_col"], cfg["manager_col"], cfg["server_col"]]
    num_base = ["dow", cfg["rain_col"], cfg["dish_col"], cfg["inv_col"]]

    # Pre-fit OHE on full X cats (OK; leakage is handled in TE)
    X_cat_full = ohe.fit_transform(df[cat_cols])
    X_num_full = csr_matrix(df[num_base].to_numpy(dtype=float))

    # Leakage-safe TE via KFold
    te_feature = np.zeros(len(df), dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df):
        dtr = df.iloc[train_idx]
        dval = df.iloc[val_idx]
        te_maps = _build_te_maps(dtr, cfg)
        te_vals = [
            _apply_te_row(row[cfg["server_col"]], int(row["dow"]), row[cfg["meal_col"]], te_maps)
            for _, row in dval.iterrows()
        ]
        te_feature[val_idx] = te_vals

    X_te = csr_matrix(te_feature.reshape(-1, 1))
    X_enc = hstack([X_num_full, X_te, X_cat_full])
    return X_enc

def train_per_server(df: pd.DataFrame, cfg: dict):
    X = df[["dow", cfg["meal_col"], cfg["host_col"], cfg["manager_col"], cfg["server_col"],
             cfg["rain_col"], cfg["dish_col"], cfg["inv_col"]]]
    y = df[cfg["server_sales_col"]].astype(float)

    cat_cols = [cfg["meal_col"], cfg["host_col"], cfg["manager_col"], cfg["server_col"]]
    num_cols = ["dow", cfg["rain_col"], cfg["dish_col"], cfg["inv_col"]]

    ohe = _make_ohe_backward_compatible()

    if cfg.get("target_encode"):
        X_enc = _make_target_encoded_matrix(df, cfg, ohe)
        te_maps = _build_te_maps(df, cfg)
    else:
        X_cat = ohe.fit_transform(X[cat_cols])
        X_num = csr_matrix(X[num_cols].to_numpy(dtype=float))
        X_enc = hstack([X_num, X_cat])
        te_maps = None

    # Model
    if DEFAULTS["use_random_forest"]:
        model = RandomForestRegressor(n_estimators=DEFAULTS["rf_estimators"], random_state=42, n_jobs=-1)
    else:
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(max_depth=DEFAULTS["tree_max_depth"], random_state=42)

    X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y, test_size=DEFAULTS["test_size"], random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"[Per-Server] R²: {r2_score(y_te, pred):.3f}   MAE: {mean_absolute_error(y_te, pred):.2f}")
    return model, ohe, cat_cols, num_cols, te_maps

def predict_servers_for_scenario(df: pd.DataFrame, model, ohe, cat_cols, num_cols, cfg: dict,
                                 dow: int, meal: str, rain: int, dish: int, inv: int,
                                 te_maps=None) -> pd.DataFrame:
    hosts = df[cfg["host_col"]].dropna().unique()
    managers = df[cfg["manager_col"]].dropna().unique()
    servers = df[cfg["server_col"]].dropna().unique()

    rows = []
    for h in hosts:
        for m in managers:
            for s in servers:
                rows.append({
                    "dow": dow, cfg["meal_col"]: meal, cfg["host_col"]: h, cfg["manager_col"]: m,
                    cfg["server_col"]: s, cfg["rain_col"]: rain, cfg["dish_col"]: dish, cfg["inv_col"]: inv
                })
    c = pd.DataFrame(rows)

    # Build design matrix (with optional TE)
    if cfg.get("target_encode") and te_maps is not None:
        base_num = csr_matrix(c[["dow", cfg["rain_col"], cfg["dish_col"], cfg["inv_col"]]].to_numpy(dtype=float))
        te_vals = [
            _apply_te_row(row[cfg["server_col"]], int(row["dow"]), row[cfg["meal_col"]], te_maps)
            for _, row in c.iterrows()
        ]
        X_te = csr_matrix(np.array(te_vals).reshape(-1, 1))
        X_cat = ohe.transform(c[[cfg["meal_col"], cfg["host_col"], cfg["manager_col"], cfg["server_col"]]])
        X_enc = hstack([base_num, X_te, X_cat])
    else:
        X_cat = ohe.transform(c[cat_cols])
        X_num = csr_matrix(c[num_cols].to_numpy(dtype=float))
        X_enc = hstack([X_num, X_cat])

    c["predicted_server_sales"] = model.predict(X_enc)
    c = c.sort_values("predicted_server_sales", ascending=False)
    return c

# Build shift-level rows from per-server (top 6 by server_sales)
def aggregate_per_server_to_shift(per_server_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    keys = [cfg["date_col"], cfg["meal_col"], cfg["host_col"], cfg["manager_col"],
            cfg["rain_col"], cfg["dish_col"], cfg["inv_col"]]
    rows = []
    for key, g in per_server_df.groupby(keys, dropna=False):
        date, meal, host, manager, rain, dw, inv = key
        g = g.sort_values(cfg["server_sales_col"], ascending=False).head(6).copy()
        servers = g[cfg["server_col"]].tolist()
        while len(servers) < 6:
            servers.append("FILL")
        total_sales = float(g[cfg["server_sales_col"]].sum())
        rows.append({
            cfg["date_col"]: pd.to_datetime(date).strftime("%Y-%m-%d"),
            cfg["host_col"]: host,
            cfg["manager_col"]: manager,
            "server1": servers[0], "server2": servers[1], "server3": servers[2],
            "server4": servers[3], "server5": servers[4], "server6": servers[5],
            cfg["sales_col"]: total_sales,
            cfg["rain_col"]: int(rain),
            cfg["meal_col"]: meal,
            cfg["dish_col"]: int(dw),
            cfg["inv_col"]: int(inv),
        })
    out = pd.DataFrame(rows).sort_values([cfg["date_col"], cfg["meal_col"], cfg["host_col"], cfg["manager_col"]])
    return out

# ---------------- SHIFT-LEVEL MODEL ----------------
def load_shift_level(path: str, cfg: dict) -> pd.DataFrame:
    df = _read_csv_robust(path)
    need = [cfg["date_col"], cfg["host_col"], cfg["manager_col"], cfg["sales_col"],
            cfg["rain_col"], cfg["meal_col"], cfg["dish_col"], cfg["inv_col"]] + cfg["server_cols"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Shift-level CSV missing columns: {miss}")
    df = parse_date_and_dow(df, cfg["date_col"])
    if "event_day" not in df.columns:
        df["event_day"] = 0
    df["meal_flag"] = df[cfg["meal_col"]].map({"Lunch":0,"Dinner":1})
    for c in cfg["server_cols"]:
        df[c] = df[c].astype(str)
    df["server_team"] = df[cfg["server_cols"]].apply(lambda r: tuple(sorted(set(r.values))), axis=1)
    df[cfg["sales_col"]] = pd.to_numeric(df[cfg["sales_col"]], errors="coerce")
    df = df.dropna(subset=[cfg["sales_col"]])
    return df

def build_shift_design_matrices(df: pd.DataFrame, cfg: dict):
    ohe = _make_ohe_backward_compatible()
    hm_X = ohe.fit_transform(df[[cfg["host_col"], cfg["manager_col"]]])
    mlb = MultiLabelBinarizer()
    srv_X = mlb.fit_transform(df["server_team"]); srv_X = csr_matrix(srv_X)
    num_cols = ["dow", "event_day", cfg["rain_col"], cfg["dish_col"], cfg["inv_col"], "meal_flag"]
    num_X = csr_matrix(df[num_cols].to_numpy(dtype=float))
    X = hstack([hm_X, srv_X, num_X])
    enc = {"ohe": ohe, "mlb": mlb, "num_cols": num_cols,
           "server_classes": list(mlb.classes_),
           "server_freq": pd.Series(df[cfg["server_cols"]].values.ravel()).value_counts().to_dict()}
    y = df[cfg["sales_col"]].values
    return X, y, enc

def transform_shift_candidates(cand: pd.DataFrame, enc: dict, cfg: dict):
    hm_X = enc["ohe"].transform(cand[[cfg["host_col"], cfg["manager_col"]]])
    srv_X = enc["mlb"].transform(cand["server_team"]); srv_X = csr_matrix(srv_X)
    num_X = csr_matrix(cand[enc["num_cols"]].to_numpy(dtype=float))
    return hstack([hm_X, srv_X, num_X])

def build_roster(df: pd.DataFrame, cfg: dict):
    hosts = (cfg["restrict_hosts_to"] or sorted(df[cfg["host_col"]].dropna().unique().tolist()))
    mgrs  = (cfg["restrict_managers_to"] or sorted(df[cfg["manager_col"]].dropna().unique().tolist()))
    servers = sorted(pd.unique(df[cfg["server_cols"]].values.ravel()))
    servers = [s for s in servers if pd.notna(s) and s not in set(cfg["exclude_servers"])]
    must_set = set(cfg["must_include_servers"])
    return hosts, mgrs, servers, must_set

def apply_fairness_penalty(df_ranked: pd.DataFrame, recently_overused: list[str], penalty: float) -> pd.DataFrame:
    recent = set(recently_overused)
    out = df_ranked.copy()
    if not recent:
        out["score_fair"] = out["predicted_sales"]
        return out
    def penalize(row):
        count = 0
        team = set(row["server_team"])
        count += len(team & recent)
        if row["host"] in recent: count += 1
        if row["manager"] in recent: count += 1
        return row["predicted_sales"] - penalty * count
    out["score_fair"] = out.apply(penalize, axis=1)
    return out

def _pick_neutral_server(enc: dict, team: Tuple[str, ...]) -> str:
    freq: Dict[str, int] = enc["server_freq"]
    classes: List[str] = enc["server_classes"]
    candidates = [s for s in classes if s not in set(team)]
    if not candidates:
        return classes[0]
    candidates.sort(key=lambda s: freq.get(s, 0))
    return candidates[len(candidates)//2]

def assign_roles_for_lineup(row, cfg, enc, model):
    meal_flag = int(row.get("meal_flag", 1 if cfg["score_meal"]=="Dinner" else 0))
    base = pd.DataFrame([{
        cfg["host_col"]: row[cfg["host_col"]],
        cfg["manager_col"]: row[cfg["manager_col"]],
        "server_team": tuple(sorted(row["server_team"])),
        "dow": int(row["dow"]),
        "event_day": int(row["event_day"]),
        cfg["rain_col"]: int(row[cfg["rain_col"]]),
        cfg["dish_col"]: int(row[cfg["dish_col"]]),
        cfg["inv_col"]: int(row[cfg["inv_col"]]),
        "meal_flag": meal_flag,
    }])
    Xb = transform_shift_candidates(base, enc, cfg)
    baseline = float(model.predict(Xb)[0])

    servers = list(row["server_team"])
    results = []
    neutral = _pick_neutral_server(enc, tuple(servers))
    for s in servers:
        new_team = tuple(sorted([neutral if name == s else name for name in servers]))
        tmp = base.copy(); tmp.at[0, "server_team"] = new_team
        Xn = transform_shift_candidates(tmp, enc, cfg)
        new_pred = float(model.predict(Xn)[0])
        results.append({"server": s, "marginal": baseline - new_pred})

    results.sort(key=lambda x: x["marginal"], reverse=True)
    closer_n = int(DEFAULTS["closer_count"]); firstcut_n = int(DEFAULTS["firstcut_count"])
    for i, r in enumerate(results):
        if i < closer_n: r["role"] = "Closer"
        elif i >= len(results) - firstcut_n: r["role"] = "First Cut"
        else: r["role"] = "Mid Shift"
    return results, baseline

def maybe_plot_top10(ranked: pd.DataFrame, cfg: dict, metric_name: str, title: str):
    if not DEFAULTS["plot_top10"]:
        return
    if not _HAS_MPL:
        print("matplotlib not available; skipping plot.")
        return
    outdir = Path(cfg["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    data = ranked.head(10).copy()
    labels = [f"{h} / {m}\n" + ", ".join(team)
              for h, m, team in zip(data[cfg["host_col"]], data[cfg["manager_col"]], data["server_team"])]
    plt.figure(figsize=(12,7))
    plt.barh(range(len(data)), data[metric_name].values)
    plt.yticks(range(len(data)), labels); plt.gca().invert_yaxis()
    plt.xlabel("Score" + (" (fairness-adjusted)" if metric_name=="score_fair" else " (predicted sales)"))
    plt.title(title); plt.tight_layout()
    plt.savefig(outdir / DEFAULTS["plot_filename"], dpi=120)
    print(f"Saved plot: {outdir / DEFAULTS['plot_filename']}")

# ---------------- MAIN RUN ----------------
def run(cfg: dict):
    outdir = Path(cfg["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)

    # Load data & detect schema
    df_raw = _read_csv_robust(cfg["csv"])
    per_server_mode = _is_per_server(df_raw, cfg)

    # Train per-server model (if per-server CSV)
    per_server_results = None
    if per_server_mode:
        df_ps = load_per_server(cfg["csv"], cfg)
        ps_model, ps_ohe, ps_cat, ps_num, te_maps = train_per_server(df_ps, cfg)
        # scenario
        dow = int(df_ps["dow"].mode().iloc[0]) if cfg["score_dow"] is None else int(cfg["score_dow"])
        meal = cfg["score_meal"]
        rain, dish, inv = int(cfg["score_rain"]), int(cfg["score_dish"]), int(cfg["score_inv"])
        per_server_results = predict_servers_for_scenario(
            df_ps, ps_model, ps_ohe, ps_cat, ps_num, cfg,
            dow, meal, rain, dish, inv, te_maps=te_maps
        )
        per_server_results.to_csv(outdir / "per_server_rankings.csv", index=False)
        print(f"Saved: {outdir/'per_server_rankings.csv'} (ranked individual server predictions)")
        print("\nTop 10 servers for this scenario:")
        print(per_server_results.head(10)[[cfg["server_col"], cfg["host_col"], cfg["manager_col"], "predicted_server_sales"]])

        # Aggregate to shift-level for combo model
        df_shift = aggregate_per_server_to_shift(df_ps.rename(columns={cfg["server_sales_col"]: "server_sales"}), cfg)
    else:
        # already shift-level
        df_shift = load_shift_level(cfg["csv"], cfg)

    # --- SHIFT-LEVEL MODEL ---
    df_shift = parse_date_and_dow(df_shift, cfg["date_col"])
    if "event_day" not in df_shift.columns:
        df_shift["event_day"] = 0
    df_shift["meal_flag"] = df_shift[cfg["meal_col"]].map({"Lunch":0,"Dinner":1})
    for c in cfg["server_cols"]:
        df_shift[c] = df_shift[c].astype(str)
    df_shift["server_team"] = df_shift[cfg["server_cols"]].apply(lambda r: tuple(sorted(set(r.values))), axis=1)

    X, y, enc = build_shift_design_matrices(df_shift, cfg)
    if DEFAULTS["use_random_forest"]:
        model = RandomForestRegressor(n_estimators=DEFAULTS["rf_estimators"], random_state=42, n_jobs=-1)
    else:
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(max_depth=DEFAULTS["tree_max_depth"], random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=DEFAULTS["test_size"], random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"[Shift-Combo] R²: {r2_score(y_te, pred):.3f}   MAE: {mean_absolute_error(y_te, pred):.2f}")

    # Roster
    hosts, mgrs, servers, must_set = build_roster(df_shift, cfg)
    all_teams = itertools.combinations(servers, 6)
    if must_set:
        all_teams = (t for t in all_teams if must_set.issubset(t))
    all_teams = list(all_teams)

    total = len(hosts) * len(mgrs) * len(all_teams)
    print(f"Potential combos to score: {total:,}")
    if total > DEFAULTS["max_candidate_combos"]:
        rng = np.random.default_rng(42)
        keep = DEFAULTS["max_candidate_combos"] // max(1, len(hosts) * len(mgrs))
        keep = max(keep, 1)
        idx = rng.choice(len(all_teams), size=min(len(all_teams), keep), replace=False)
        all_teams = [all_teams[i] for i in idx]
        print(f"Sampling server teams to {len(all_teams):,} to keep compute quick.")

    # Scenario flags
    mode_dow = int(df_shift["dow"].mode().iloc[0])
    dow = mode_dow if cfg["score_dow"] is None else int(cfg["score_dow"])
    meal_flag = {"Lunch":0, "Dinner":1}[cfg["score_meal"]]
    rows = []
    for h in hosts:
        for m in mgrs:
            for team in all_teams:
                rows.append({
                    cfg["host_col"]: h, cfg["manager_col"]: m,
                    "server_team": tuple(sorted(team)),
                    "dow": dow, "event_day": int(cfg["score_event_day"]),
                    cfg["rain_col"]: int(cfg["score_rain"]),
                    cfg["dish_col"]: int(cfg["score_dish"]),
                    cfg["inv_col"]: int(cfg["score_inv"]),
                    "meal_flag": meal_flag,
                })
    cand = pd.DataFrame(rows)
    X_cand = transform_shift_candidates(cand, enc, cfg)
    cand["predicted_sales"] = model.predict(X_cand)
    ranked = cand.sort_values("predicted_sales", ascending=False).reset_index(drop=True)
    if DEFAULTS["enable_fairness_penalty"]:
        ranked = apply_fairness_penalty(ranked, DEFAULTS["recently_overused"], DEFAULTS["fairness_penalty"])
        ranked = ranked.sort_values("score_fair", ascending=False).reset_index(drop=True)
        metric_name = "score_fair"
    else:
        metric_name = "predicted_sales"

    # Highest / Lowest explicit
    best, worst = ranked.iloc[0], ranked.iloc[-1]
    def fmt_row(r):
        meal_txt = "Dinner" if int(r.get("meal_flag", meal_flag)) == 1 else "Lunch"
        team = ", ".join(r["server_team"])
        return (f"Host: {r[cfg['host_col']]} | Manager: {r[cfg['manager_col']]} | "
                f"DOW:{int(r['dow'])} Event:{int(r['event_day'])} Rain:{int(r[cfg['rain_col']])} "
                f"DW:{int(r[cfg['dish_col']])} INV:{int(r[cfg['inv_col']])} Meal:{meal_txt} | "
                f"${r['predicted_sales']:,.0f} | Team: {team}")
    print("\nHIGHEST predicted (scenario):")
    print(fmt_row(best))
    print("\nLOWEST predicted (scenario):")
    print(fmt_row(worst))

    # Roles for top N
    role_rows = []
    topN = ranked.head(DEFAULTS["top_n"]).copy()
    if "meal_flag" not in topN.columns:
        topN["meal_flag"] = meal_flag
    if "event_day" not in topN.columns:
        topN["event_day"] = int(cfg["score_event_day"])

    for _, r in topN.iterrows():
        server_roles, baseline = assign_roles_for_lineup(r, cfg, enc, model)
        for sr in server_roles:
            role_rows.append({
                "host": r[cfg["host_col"]],
                "manager": r[cfg["manager_col"]],
                "dow": int(r["dow"]),
                "event_day": int(r["event_day"]),
                cfg["rain_col"]: int(r[cfg["rain_col"]]),
                cfg["dish_col"]: int(r[cfg["dish_col"]]),
                cfg["inv_col"]: int(r[cfg["inv_col"]]),
                "meal": "Dinner" if int(r["meal_flag"])==1 else "Lunch",
                "predicted_sales": float(r["predicted_sales"]),
                "server": sr["server"],
                "server_marginal": round(sr["marginal"], 2),
                "assigned_role": sr["role"],
            })
    roles_df = pd.DataFrame(role_rows)

    # Save shift-level outputs
    cols = ["host","manager","server_team","dow","event_day",cfg["rain_col"],cfg["dish_col"],cfg["inv_col"],"meal_flag","predicted_sales"]
    if "score_fair" in ranked.columns: cols.append("score_fair")
    ranked.head(DEFAULTS["top_n"])[cols].to_csv(outdir/"top_combos.csv", index=False)
    ranked.tail(DEFAULTS["top_n"])[cols].to_csv(outdir/"bottom_combos.csv", index=False)
    roles_df.to_csv(outdir/"top_combos_with_roles.csv", index=False)
    print(f"\nSaved: {outdir/'top_combos.csv'}  (top {DEFAULTS['top_n']})")
    print(f"Saved: {outdir/'bottom_combos.csv'} (bottom {DEFAULTS['top_n']})")
    print(f"Saved: {outdir/'top_combos_with_roles.csv'} (roles for top lineups)")

    # Chart
    if DEFAULTS["plot_top10"]:
        title = (f"Top 10 — DOW:{cfg.get('score_dow','mode')} Meal:{cfg['score_meal']} "
                 f"Rain:{cfg['score_rain']} DW:{cfg['score_dish']} INV:{cfg['score_inv']} Event:{cfg['score_event_day']}")
        maybe_plot_top10(ranked, cfg, metric_name, title)

    # Week plan + weekly summary (+ optional schedule export)
    if DEFAULTS["make_week_plan"]:
        week_rows = []
        schedule_rows = []  # for export_schedule
        for dow_i in range(7):
            rows = []
            for h in hosts:
                for m in mgrs:
                    for team in all_teams:
                        rows.append({
                            cfg["host_col"]: h, cfg["manager_col"]: m,
                            "server_team": tuple(sorted(team)),
                            "dow": dow_i,
                            "event_day": int(cfg["score_event_day"]),
                            cfg["rain_col"]: int(cfg["score_rain"]),
                            cfg["dish_col"]: int(cfg["score_dish"]),
                            cfg["inv_col"]: int(cfg["score_inv"]),
                            "meal_flag": meal_flag,
                        })
            day_df = pd.DataFrame(rows)
            X_day = transform_shift_candidates(day_df, enc, cfg)
            day_df["predicted_sales"] = model.predict(X_day)
            if DEFAULTS["enable_fairness_penalty"]:
                day_df = apply_fairness_penalty(day_df, DEFAULTS["recently_overused"], DEFAULTS["fairness_penalty"])
                metric_day = "score_fair"
            else:
                metric_day = "predicted_sales"
            best_day = day_df.sort_values(metric_day, ascending=False).iloc[0]
            week_rows.append({
                "dow": dow_i,
                "meal": cfg["score_meal"],
                "rain": cfg["score_rain"],
                "event_day": cfg["score_event_day"],
                "dishwasher_down": cfg["score_dish"],
                "low_inventory": cfg["score_inv"],
                "host": best_day["host"],
                "manager": best_day["manager"],
                "server_team": ", ".join(best_day["server_team"]),
                "predicted_sales": best_day["predicted_sales"],
                **({"score_fair": best_day["score_fair"]} if "score_fair" in best_day else {})
            })

            # roles for best-of-day (for schedule export)
            if DEFAULTS["export_schedule"]:
                roles, _ = assign_roles_for_lineup(best_day, cfg, enc, model)
                closers = [r["server"] for r in roles if r["role"] == "Closer"]
                first_cut = [r["server"] for r in roles if r["role"] == "First Cut"]
                mid = [r["server"] for r in roles if r["role"] == "Mid Shift"]
                schedule_rows.append({
                    "dow": dow_i,
                    "meal": cfg["score_meal"],
                    "host": best_day["host"],
                    "manager": best_day["manager"],
                    "team": ", ".join(best_day["server_team"]),
                    "closers": ", ".join(closers),
                    "mid_shift": ", ".join(mid),
                    "first_cut": ", ".join(first_cut),
                    "predicted_sales": round(float(best_day["predicted_sales"]), 2),
                })

        week = pd.DataFrame(week_rows)
        week.to_csv(outdir/"week_plan.csv", index=False)
        print(f"Saved: {outdir/'week_plan.csv'}")

        # Weekly summary: best servers/managers/hosts across best daily lineups
        server_counts, server_weighted = {}, {}
        for _, r in week.iterrows():
            for s in [x.strip() for x in r["server_team"].split(",")]:
                server_counts[s] = server_counts.get(s, 0) + 1
                server_weighted[s] = server_weighted.get(s, 0.0) + float(r["predicted_sales"])
        best_servers = sorted(server_counts.keys(),
                              key=lambda s: (server_counts[s], server_weighted[s]),
                              reverse=True)[:3]
        mgr_counts = week["manager"].value_counts().to_dict()
        mgr_weighted = week.groupby("manager")["predicted_sales"].sum().to_dict()
        best_managers = sorted(mgr_counts.keys(),
                               key=lambda m: (mgr_counts[m], mgr_weighted.get(m,0.0)),
                               reverse=True)[:3]
        host_counts = week["host"].value_counts().to_dict()
        host_weighted = week.groupby("host")["predicted_sales"].sum().to_dict()
        best_hosts = sorted(host_counts.keys(),
                            key=lambda h: (host_counts[h], host_weighted.get(h,0.0)),
                            reverse=True)[:3]

        print("\nWEEKLY SUMMARY:")
        print(f"- Best servers for this week are: {', '.join(best_servers)}.")
        print(f"- Most effective managers this week: {', '.join(best_managers)}.")
        print(f"- Most effective hosts this week: {', '.join(best_hosts)}.")
        print("  (based on frequency in best daily lineups, tiebreak by total predicted sales)")

        # Export human-friendly schedule with roles
        if DEFAULTS["export_schedule"]:
            sched = pd.DataFrame(schedule_rows).sort_values(["dow"])
            sched.to_csv(outdir/"weekly_schedule_with_roles.csv", index=False)
            print(f"Saved: {outdir/'weekly_schedule_with_roles.csv'}")

# ---------------- Args ----------------
def parse_args_to_cfg() -> dict:
    cfg = DEFAULTS.copy()
    p = argparse.ArgumentParser(description="FOH predictor (per-server + shift combo).")
    p.add_argument("--csv", help="Input CSV (per-server or shift-level).")
    p.add_argument("--events", dest="events_csv", help="Events CSV (optional; shift-level).")
    p.add_argument("--dow", type=int, help="0=Mon .. 6=Sun (default: mode of data)")
    p.add_argument("--meal", choices=["Lunch","Dinner"], help="Meal")
    p.add_argument("--rain", type=int, choices=[0,1], help="Rain (0/1)")
    p.add_argument("--event", type=int, choices=[0,1], help="Event day (0/1)")
    p.add_argument("--dishwasher", type=int, choices=[0,1], help="Dishwasher down (0/1)")
    p.add_argument("--inventory", type=int, choices=[0,1], help="Low inventory (0/1)")
    p.add_argument("--include", help="Comma-separated must-include servers")
    p.add_argument("--exclude", help="Comma-separated excluded servers")
    p.add_argument("--hosts", help="Comma-separated allowed hosts")
    p.add_argument("--managers", help="Comma-separated allowed managers")
    p.add_argument("--top", type=int, help="Top N to save/display (default 25)")
    p.add_argument("--max-combos", type=int, help="Safety cap on total combos (default 50k)")
    p.add_argument("--fairness-penalty", type=float, help="Penalty per overused staff")
    p.add_argument("--overused", help="Comma-separated names to penalize")
    p.add_argument("--no-fairness", action="store_true", help="Disable fairness penalty")
    p.add_argument("--plot", action="store_true", help="Enable Top10 chart")
    p.add_argument("--no-plot", action="store_true", help="Disable Top10 chart")
    p.add_argument("--week-plan", action="store_true", help="Generate week plan CSV")
    p.add_argument("--target-encode", action="store_true", help="Leakage-safe target encoding for per-server model")
    p.add_argument("--export-schedule", action="store_true", help="Write readable schedule with roles (requires --week-plan)")
    args = p.parse_args()

    if args.csv: cfg["csv"] = args.csv
    if args.events_csv: cfg["events_csv"] = args.events_csv
    if args.dow is not None: cfg["score_dow"] = args.dow
    if args.meal: cfg["score_meal"] = args.meal
    if args.rain is not None: cfg["score_rain"] = args.rain
    if args.event is not None: cfg["score_event_day"] = args.event
    if args.dishwasher is not None: cfg["score_dish"] = args.dishwasher
    if args.inventory is not None: cfg["score_inv"] = args.inventory
    if args.include: cfg["must_include_servers"] = [s.strip() for s in args.include.split(",") if s.strip()]
    if args.exclude: cfg["exclude_servers"] = [s.strip() for s in args.exclude.split(",") if s.strip()]
    if args.hosts: cfg["restrict_hosts_to"] = [s.strip() for s in args.hosts.split(",") if s.strip()]
    if args.managers: cfg["restrict_managers_to"] = [s.strip() for s in args.managers.split(",") if s.strip()]
    if args.top: DEFAULTS["top_n"] = args.top
    if args.max_combos: DEFAULTS["max_candidate_combos"] = args.max_combos
    if args.overused: DEFAULTS["recently_overused"] = [s.strip() for s in args.overused.split(",") if s.strip()]
    if args.fairness_penalty is not None: DEFAULTS["fairness_penalty"] = args.fairness_penalty
    if args.no_fairness: DEFAULTS["enable_fairness_penalty"] = False
    if args.plot: DEFAULTS["plot_top10"] = True
    if args.no_plot: DEFAULTS["plot_top10"] = False
    if args.week_plan: DEFAULTS["make_week_plan"] = True
    if args.target_encode: DEFAULTS["target_encode"] = True
    if args.export_schedule: DEFAULTS["export_schedule"] = True
    return cfg

if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    run(cfg)
