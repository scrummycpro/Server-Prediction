"""
predict_top_6_server_combos_plus.py

CLI-enabled lineup predictor for FOH:
- Predicts sales from host, manager, 6-server team, and features:
  DOW, event_day, rain, dishwasher_down, low_inventory, meal (Lunch/Dinner).
- Scores all combos for a chosen scenario, prints highest/lowest lines.
- Builds a week plan (best per DOW) if requested.
- Assigns roles inside each 6-server team: Closer / Mid Shift / First Cut
  via simple marginal impact (swap a server with a neutral replacement and re-predict).
- Produces a Top 10 chart and saves CSV outputs.

Install:
  pip install pandas numpy scikit-learn scipy matplotlib

Examples:
  # Saturday dinner, rainy, event day, dishwasher down, low inventory off
  python predict_top_6_server_combos_plus.py --dow 6 --meal Dinner --rain 1 --event 1 --dishwasher 1 --inventory 0

  # Lunch no-rain, include Olivia, exclude Lucas/James, fairness off
  python predict_top_6_server_combos_plus.py --meal Lunch --include Olivia --exclude Lucas,James --no-fairness

  # Make a full week plan for Dinner when raining
  python predict_top_6_server_combos_plus.py --meal Dinner --rain 1 --week-plan
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

# Optional plotting
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ---------- Defaults (can be overridden by CLI) ----------
DEFAULTS = {
    "train_csv": "shift_sales_6servers.csv",
    "events_csv": None,               # e.g., "events_calendar.csv"
    "output_dir": "outputs",

    "date_col": "date",
    "host_col": "host",
    "manager_col": "manager",
    "server_cols": ["server1","server2","server3","server4","server5","server6"],
    "sales_col": "sales",
    "rain_col": "rain",
    "meal_col": "meal",
    "dish_col": "dishwasher_down",
    "inv_col": "low_inventory",

    "date_format": None,

    "use_random_forest": True,
    "rf_estimators": 200,
    "tree_max_depth": 5,
    "test_size": 0.2,

    "max_candidate_combos": 50_000,
    "top_n": 25,

    "score_dow": None,    # None = use mode of training data
    "score_event_day": 0,
    "score_rain": 0,
    "score_meal": "Dinner",
    "score_dish": 0,      # dishwasher_down (0/1)
    "score_inv": 0,       # low_inventory (0/1)

    "must_include_servers": [],
    "exclude_servers": [],
    "restrict_hosts_to": [],
    "restrict_managers_to": [],

    "enable_fairness_penalty": True,
    "recently_overused": [],
    "fairness_penalty": 75.0,

    "plot_top10": True,
    "plot_filename": "top_combos_top10.png",

    "make_week_plan": False,   # off by default when using CLI

    # Role assignment config
    "closer_count": 2,
    "firstcut_count": 2,
}


def parse_date_series(s: pd.Series, fmt: str | None) -> pd.Series:
    if fmt:
        return pd.to_datetime(s, format=fmt, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def load_and_enrich_training(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["train_csv"])

    required = [cfg["date_col"], cfg["host_col"], cfg["manager_col"], cfg["sales_col"],
                cfg["rain_col"], cfg["meal_col"], cfg["dish_col"], cfg["inv_col"]] + cfg["server_cols"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Training CSV missing columns: {missing}")

    # Parse date & DOW
    df[cfg["date_col"]] = parse_date_series(df[cfg["date_col"]], cfg.get("date_format"))
    if df[cfg["date_col"]].isna().any():
        bad = df[df[cfg["date_col"]].isna()].head(3)
        raise ValueError(f"Unparseable dates found. Example rows:\n{bad}")
    df["dow"] = df[cfg["date_col"]].dt.dayofweek

    # Events join (optional)
    if cfg.get("events_csv"):
        ev = pd.read_csv(cfg["events_csv"])
        if not {"date","event_day"}.issubset(ev.columns):
            raise ValueError("events_csv must have columns: date, event_day")
        ev["date"] = parse_date_series(ev["date"], cfg.get("date_format")).dropna()
        ev = ev.drop_duplicates("date")
        df = df.merge(ev[["date","event_day"]],
                      left_on=cfg["date_col"], right_on="date", how="left")
        df["event_day"] = df["event_day"].fillna(0).astype(int)
        df.drop(columns=["date_y"], inplace=True, errors="ignore")
        if "date_x" in df.columns:
            df.rename(columns={"date_x": cfg["date_col"]}, inplace=True)
    else:
        df["event_day"] = 0

    # Normalize binary/numeric signals
    for bin_col in (cfg["rain_col"], cfg["dish_col"], cfg["inv_col"]):
        df[bin_col] = pd.to_numeric(df[bin_col], errors="coerce").fillna(0).clip(0,1).astype(int)

    df["meal_flag"] = df[cfg["meal_col"]].map({"Lunch":0,"Dinner":1})
    if df["meal_flag"].isna().any():
        raise ValueError("meal must be 'Lunch' or 'Dinner' in all rows.")

    # Team-of-6 as unordered tuple
    for c in cfg["server_cols"]:
        df[c] = df[c].astype(str)
    df["server_team"] = df[cfg["server_cols"]].apply(lambda r: tuple(sorted(set(r.values))), axis=1)

    # Sales numeric
    df[cfg["sales_col"]] = pd.to_numeric(df[cfg["sales_col"]], errors="coerce")
    df = df.dropna(subset=[cfg["sales_col"]])
    return df


def _make_ohe_backward_compatible():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)  # sklearn ≥1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)         # sklearn <1.2


def build_design_matrices(df: pd.DataFrame, cfg: dict):
    # One-hot host/manager
    ohe = _make_ohe_backward_compatible()
    hm_X = ohe.fit_transform(df[[cfg["host_col"], cfg["manager_col"]]])

    # Multi-hot servers
    mlb = MultiLabelBinarizer()
    srv_X = mlb.fit_transform(df["server_team"])
    srv_X = csr_matrix(srv_X)

    # Also record server frequency and classes (for neutral replacements)
    servers_flat = pd.Series(df[cfg["server_cols"]].values.ravel())
    server_freq = servers_flat.value_counts().to_dict()
    server_classes = list(mlb.classes_)

    # Numeric passthrough
    extra_numeric_cols = [
        "dow", "event_day", cfg["rain_col"], cfg["dish_col"], cfg["inv_col"], "meal_flag"
    ]
    num_X = csr_matrix(df[extra_numeric_cols].to_numpy(dtype=float))

    X = hstack([hm_X, srv_X, num_X])
    encoders = {
        "ohe": ohe,
        "mlb": mlb,
        "num_cols": extra_numeric_cols,
        "server_freq": server_freq,
        "server_classes": server_classes,
    }
    return X, encoders


def transform_candidates(cand: pd.DataFrame, enc: dict, cfg: dict):
    hm_X = enc["ohe"].transform(cand[[cfg["host_col"], cfg["manager_col"]]])
    srv_X = enc["mlb"].transform(cand["server_team"])
    srv_X = csr_matrix(srv_X)
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
        if row["host"] in recent:
            count += 1
        if row["manager"] in recent:
            count += 1
        return row["predicted_sales"] - penalty * count

    out["score_fair"] = out.apply(penalize, axis=1)
    return out


def maybe_plot_top10(ranked: pd.DataFrame, cfg: dict):
    if not cfg["plot_top10"] or not _HAS_MPL:
        if cfg["plot_top10"] and not _HAS_MPL:
            print("matplotlib not available; skipping plot.")
        return
    metric = "score_fair" if cfg["enable_fairness_penalty"] else "predicted_sales"
    data = ranked.head(10).copy()
    labels = [f"{h} / {m}\n" + ", ".join(team)
              for h, m, team in zip(data[cfg["host_col"]], data[cfg["manager_col"]], data["server_team"])]
    title = (f"Top 10 — DOW:{cfg.get('score_dow','mode')} Meal:{cfg['score_meal']} "
             f"Rain:{cfg['score_rain']} DW:{cfg['score_dish']} INV:{cfg['score_inv']} Event:{cfg['score_event_day']}")
    outdir = Path(cfg["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / cfg["plot_filename"]
    plt.figure(figsize=(12, 7))
    plt.barh(range(len(data)), data[metric].values)
    plt.yticks(range(len(data)), labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Score" + (" (fairness-adjusted)" if metric == "score_fair" else " (predicted sales)"))
    plt.title(title); plt.tight_layout(); plt.savefig(fname, dpi=120)
    print(f"Saved plot: {fname}")


def _pick_neutral_server(enc: dict, team: Tuple[str, ...]) -> str:
    """
    Choose a neutral replacement server that already exists in the training classes,
    and is not in the current team. We pick one with median frequency to avoid extremes.
    """
    freq: Dict[str, int] = enc["server_freq"]
    classes: List[str] = enc["server_classes"]
    candidates = [s for s in classes if s not in set(team)]
    if not candidates:
        # fallback: just return the first class (shouldn't happen)
        return classes[0]
    # sort by frequency and pick median
    candidates.sort(key=lambda s: freq.get(s, 0))
    return candidates[len(candidates)//2]


def assign_roles_for_lineup(row, cfg, enc, model):
    """
    Given a ranked lineup row with columns:
      host, manager, server_team, dow, event_day, rain, dishwasher_down, low_inventory, meal_flag
    return a list of dicts per server: {"server", "marginal", "role"}, and the baseline prediction.
    """
    # Build a single-row candidate DF for baseline
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

    Xb = transform_candidates(base, enc, cfg)
    baseline = float(model.predict(Xb)[0])

    servers = list(row["server_team"])
    results = []
    neutral = _pick_neutral_server(enc, tuple(servers))

    for s in servers:
        # team with server s replaced by a neutral existing server
        new_team = tuple(sorted([neutral if name == s else name for name in servers]))
        tmp = base.copy()
        tmp.at[0, "server_team"] = new_team
        Xn = transform_candidates(tmp, enc, cfg)
        new_pred = float(model.predict(Xn)[0])
        marginal = baseline - new_pred
        results.append({"server": s, "marginal": marginal})

    # Rank servers by marginal contribution
    results.sort(key=lambda x: x["marginal"], reverse=True)
    closer_n = int(cfg.get("closer_count", 2))
    firstcut_n = int(cfg.get("firstcut_count", 2))

    for i, r in enumerate(results):
        if i < closer_n:
            r["role"] = "Closer"
        elif i >= len(results) - firstcut_n:
            r["role"] = "First Cut"
        else:
            r["role"] = "Mid Shift"
    return results, baseline


def run_scenario(cfg: dict):
    # 1) Load & prepare
    df = load_and_enrich_training(cfg)
    X, enc = build_design_matrices(df, cfg)
    y = df[cfg["sales_col"]].values

    # 2) Train model
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg["test_size"], random_state=42)
    if cfg["use_random_forest"]:
        model = RandomForestRegressor(n_estimators=cfg["rf_estimators"], random_state=42, n_jobs=-1)
    else:
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(max_depth=cfg["tree_max_depth"], random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print("Model  |  R²:", round(r2_score(y_te, pred), 3), "  MAE:", round(mean_absolute_error(y_te, pred), 2))

    # 3) Roster & teams
    hosts, mgrs, servers, must_set = build_roster(df, cfg)
    teams_gen = itertools.combinations(servers, 6)
    if must_set:
        teams_gen = (t for t in teams_gen if must_set.issubset(t))
    all_teams = list(teams_gen)

    # Cap
    total = len(hosts) * len(mgrs) * len(all_teams)
    print(f"Potential combos to score: {total:,}")
    if total > cfg["max_candidate_combos"]:
        rng = np.random.default_rng(42)
        keep = cfg["max_candidate_combos"] // max(1, len(hosts) * len(mgrs))
        keep = max(keep, 1)
        idx = rng.choice(len(all_teams), size=min(len(all_teams), keep), replace=False)
        all_teams = [all_teams[i] for i in idx]
        print(f"Sampling server teams to {len(all_teams):,} to keep compute quick.")

    # 4) Scenario numeric flags
    mode_dow = int(df["dow"].mode().iloc[0])
    dow = mode_dow if cfg["score_dow"] is None else int(cfg["score_dow"])
    meal_flag = {"Lunch":0,"Dinner":1}[cfg["score_meal"]]

    # Build candidate rows
    rows = []
    for h in hosts:
        for m in mgrs:
            for team in all_teams:
                rows.append({
                    cfg["host_col"]: h,
                    cfg["manager_col"]: m,
                    "server_team": tuple(sorted(team)),
                    "dow": dow,
                    "event_day": int(cfg["score_event_day"]),
                    cfg["rain_col"]: int(cfg["score_rain"]),
                    cfg["dish_col"]: int(cfg["score_dish"]),
                    cfg["inv_col"]: int(cfg["score_inv"]),
                    "meal_flag": meal_flag,
                })
    cand = pd.DataFrame(rows)
    X_cand = transform_candidates(cand, enc, cfg)
    cand["predicted_sales"] = model.predict(X_cand)
    ranked = cand.sort_values("predicted_sales", ascending=False).reset_index(drop=True)
    if cfg["enable_fairness_penalty"]:
        ranked = apply_fairness_penalty(ranked, cfg["recently_overused"], cfg["fairness_penalty"])
        ranked = ranked.sort_values("score_fair", ascending=False).reset_index(drop=True)

    # 5) Explicit statements (best/worst)
    metric = "score_fair" if cfg["enable_fairness_penalty"] else "predicted_sales"
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

    # 6) Role assignment for TOP N rows
    role_rows = []
    top_for_roles = ranked.head(cfg["top_n"]).copy()

    # Ensure required condition cols exist in top_for_roles
    for c in ["event_day", cfg["rain_col"], cfg["dish_col"], cfg["inv_col"]]:
        if c not in top_for_roles.columns:
            top_for_roles[c] = cfg["score_event_day"] if c=="event_day" else 0
    if "meal_flag" not in top_for_roles.columns:
        top_for_roles["meal_flag"] = meal_flag

    for _, r in top_for_roles.iterrows():
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

    # 7) Save outputs + plot
    outdir = Path(cfg["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    cols = ["host","manager","server_team","dow","event_day",cfg["rain_col"],cfg["dish_col"],cfg["inv_col"],"meal_flag","predicted_sales"]
    if "score_fair" in ranked.columns: cols.append("score_fair")
    ranked.head(cfg["top_n"])[cols].to_csv(outdir/"top_combos.csv", index=False)
    ranked.tail(cfg["top_n"])[cols].to_csv(outdir/"bottom_combos.csv", index=False)
    roles_df.to_csv(outdir/"top_combos_with_roles.csv", index=False)
    print(f"\nSaved: {outdir/'top_combos.csv'}  (top {cfg['top_n']})")
    print(f"Saved: {outdir/'bottom_combos.csv'} (bottom {cfg['top_n']})")
    print(f"Saved: {outdir/'top_combos_with_roles.csv'} (roles for top lineups)")

    if cfg["plot_top10"]:
        metric_name = "score_fair" if cfg["enable_fairness_penalty"] else "predicted_sales"
        data = ranked.head(10).copy()
        labels = [f"{h} / {m}\n" + ", ".join(team)
                  for h, m, team in zip(data[cfg["host_col"]], data[cfg["manager_col"]], data["server_team"])]
        title = (f"Top 10 — DOW:{cfg.get('score_dow','mode')} Meal:{cfg['score_meal']} "
                 f"Rain:{cfg['score_rain']} DW:{cfg['score_dish']} INV:{cfg['score_inv']} Event:{cfg['score_event_day']}")
        if _HAS_MPL:
            plt.figure(figsize=(12,7))
            plt.barh(range(len(data)), data[metric_name].values)
            plt.yticks(range(len(data)), labels); plt.gca().invert_yaxis()
            plt.xlabel("Score" + (" (fairness-adjusted)" if metric_name == "score_fair" else " (predicted sales)"))
            plt.title(title); plt.tight_layout()
            plt.savefig(outdir / cfg["plot_filename"], dpi=120)
            print(f"Saved plot: {outdir / cfg['plot_filename']}")
        else:
            print("matplotlib not available; skipping plot.")

    # 8) Week plan (best per DOW) + weekly summary
    if cfg["make_week_plan"]:
        week_rows = []
        for dow_i in range(7):
            rows = []
            for h in hosts:
                for m in mgrs:
                    for team in all_teams:
                        rows.append({
                            cfg["host_col"]: h,
                            cfg["manager_col"]: m,
                            "server_team": tuple(sorted(team)),
                            "dow": dow_i,
                            "event_day": int(cfg["score_event_day"]),
                            cfg["rain_col"]: int(cfg["score_rain"]),
                            cfg["dish_col"]: int(cfg["score_dish"]),
                            cfg["inv_col"]: int(cfg["score_inv"]),
                            "meal_flag": meal_flag,
                        })
            day_df = pd.DataFrame(rows)
            X_day = transform_candidates(day_df, enc, cfg)
            day_df["predicted_sales"] = model.predict(X_day)
            if cfg["enable_fairness_penalty"]:
                day_df = apply_fairness_penalty(day_df, cfg["recently_overused"], cfg["fairness_penalty"])
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
        week = pd.DataFrame(week_rows)
        week.to_csv(outdir/"week_plan.csv", index=False)
        print(f"Saved: {outdir/'week_plan.csv'}")

        # Weekly summary: top servers / top managers across the best lineups for the week
        # Count frequency (ties broken by total predicted_sales when available)
        server_counts = {}
        server_weighted = {}
        for _, r in week.iterrows():
            team = [s.strip() for s in r["server_team"].split(",")]
            for s in team:
                server_counts[s] = server_counts.get(s, 0) + 1
                server_weighted[s] = server_weighted.get(s, 0.0) + float(r["predicted_sales"])
        # sort by (count, weighted sales)
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

        # Explicit statement
        print("\nWEEKLY SUMMARY:")
        print(f"- Best servers for this week are: {', '.join(best_servers)}.")
        print(f"- Most effective managers this week: {', '.join(best_managers)}.")
        print(f"- Most effective hosts this week: {', '.join(best_hosts)}.")
        print("  (based on frequency in best daily lineups, breaking ties by total predicted sales)")
    # Done
    return


def parse_args_to_cfg() -> dict:
    cfg = DEFAULTS.copy()
    p = argparse.ArgumentParser(description="Predict top FOH lineups (host, manager, 6 servers).")

    p.add_argument("--csv", dest="train_csv", help="Path to training CSV")
    p.add_argument("--events", dest="events_csv", help="Path to events CSV")
    p.add_argument("--dow", type=int, help="0=Mon .. 6=Sun (default: mode of data)")
    p.add_argument("--meal", choices=["Lunch","Dinner"], help="Meal (Lunch/Dinner)")
    p.add_argument("--rain", type=int, choices=[0,1], help="Rain flag (0/1)")
    p.add_argument("--event", type=int, choices=[0,1], help="Event day flag (0/1)")
    p.add_argument("--dishwasher", type=int, choices=[0,1], help="Dishwasher down (0/1)")
    p.add_argument("--inventory", type=int, choices=[0,1], help="Low inventory (0/1)")
    p.add_argument("--include", help="Comma-separated must-include servers")
    p.add_argument("--exclude", help="Comma-separated excluded servers")
    p.add_argument("--hosts", help="Comma-separated allowed hosts")
    p.add_argument("--managers", help="Comma-separated allowed managers")
    p.add_argument("--top", type=int, help="Top N to save/display")
    p.add_argument("--max-combos", type=int, help="Safety cap on total combos")
    p.add_argument("--fairness-penalty", type=float, help="Penalty per overused staff")
    p.add_argument("--overused", help="Comma-separated names to penalize")
    p.add_argument("--no-fairness", action="store_true", help="Disable fairness penalty")
    p.add_argument("--plot", action="store_true", help="Enable Top10 plot")
    p.add_argument("--no-plot", action="store_true", help="Disable Top10 plot")
    p.add_argument("--week-plan", action="store_true", help="Generate week plan CSV")

    args = p.parse_args()

    if args.train_csv: cfg["train_csv"] = args.train_csv
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
    if args.top: cfg["top_n"] = args.top
    if args.max_combos: cfg["max_candidate_combos"] = args.max_combos
    if args.overused: cfg["recently_overused"] = [s.strip() for s in args.overused.split(",") if s.strip()]
    if args.fairness_penalty is not None: cfg["fairness_penalty"] = args.fairness_penalty
    if args.no_fairness: cfg["enable_fairness_penalty"] = False
    if args.plot: cfg["plot_top10"] = True
    if args.no_plot: cfg["plot_top10"] = False
    if args.week_plan: cfg["make_week_plan"] = True

    return cfg


if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    run_scenario(cfg)
