# FOH Predictor — README

Predict and plan Front-of-House lineups using your own data. This tool can:

* Rank **individual servers** for a specific scenario (e.g., Sat Dinner, raining).
* Recommend the **best (host, manager, 6-server)** lineup(s).
* Assign **Closer / Mid / First-Cut** roles inside each top lineup.
* Generate a **Mon–Sun week plan** and an optional **human-readable schedule** CSV.
* (Optional) Improve per-server accuracy using **leakage-safe target encoding**.

---

## Quick Start

```bash
# 1) Install dependencies
pip install pandas numpy scikit-learn scipy matplotlib

# 2) Put your data in one of the supported CSV formats (see below)

# 3) Run a scenario (per-server input example)
python foh_predictor.py --csv server_daily_sales.csv --dow 6 --meal Dinner --rain 1 --event 1 --plot

# 4) Build a week plan + export schedule with roles
python foh_predictor.py --csv server_daily_sales.csv --meal Dinner --week-plan --export-schedule --plot
```

Output files are written to `outputs/`:

* `per_server_rankings.csv` (if per-server data provided)
* `top_combos.csv`, `bottom_combos.csv`
* `top_combos_with_roles.csv`
* `week_plan.csv` (with `--week-plan`)
* `weekly_schedule_with_roles.csv` (with `--export-schedule`)
* `top_combos_top10.png` (with `--plot`)

---

## Input Data Formats (choose one)

### A) Per-Server Format (recommended for richer modeling)

**File:** `server_daily_sales.csv`

One row **per server per shift**.

| column            | type         | description                         |
| ----------------- | ------------ | ----------------------------------- |
| `date`            | YYYY-MM-DD   | Calendar date of the shift          |
| `meal`            | Lunch/Dinner | Shift meal                          |
| `host`            | string       | Host on that shift                  |
| `manager`         | string       | Manager on that shift               |
| `server`          | string       | Server’s name                       |
| `server_sales`    | number       | That server’s sales on that shift   |
| `rain`            | 0/1          | Rained during shift                 |
| `dishwasher_down` | 0/1          | Dishwasher out of service           |
| `low_inventory`   | 0/1          | Inventory constraint impacted shift |

**Example:**

```csv
date,meal,host,manager,server,server_sales,rain,dishwasher_down,low_inventory
2025-07-01,Lunch,Alice,John,Mia,420,0,0,0
2025-07-01,Lunch,Alice,John,Noah,380,0,0,0
2025-07-01,Dinner,David,Sarah,Ethan,980,0,0,0
2025-07-01,Dinner,David,Sarah,Emma,920,0,0,0
```

**What happens with this format?**

* The script trains a **per-server** model to predict `server_sales`.
* It also **auto-aggregates** your per-server rows into shift rows (top 6 servers per shift, sales summed) to train the lineup model.

---

### B) Shift-Level Format

**File:** `shift_sales_6servers.csv`

One row **per shift**, listing the 6 servers and the **total** sales.

| column             | type         | description                  |
| ------------------ | ------------ | ---------------------------- |
| `date`             | YYYY-MM-DD   | Calendar date                |
| `host`             | string       | Host on shift                |
| `manager`          | string       | Manager on shift             |
| `server1..server6` | strings      | The six servers on the shift |
| `sales`            | number       | Total sales for the shift    |
| `rain`             | 0/1          | Rained during shift          |
| `meal`             | Lunch/Dinner | Shift meal                   |
| `dishwasher_down`  | 0/1          | Dishwasher out               |
| `low_inventory`    | 0/1          | Inventory constraint         |

**Example:**

```csv
date,host,manager,server1,server2,server3,server4,server5,server6,sales,rain,meal,dishwasher_down,low_inventory
2025-07-01,Alice,John,Mia,Noah,Ella,Liam,Olivia,Mason,4850,0,Lunch,0,0
2025-07-01,David,Sarah,Ethan,Emma,James,Isabella,Lucas,Sophia,7200,0,Dinner,0,0
```

---

## Command-Line Flags (Arguments)

### Scenario & Input

* `--csv PATH`
  Input CSV path. Auto-detects whether it’s **per-server** or **shift-level**.

* `--events PATH` (shift-level optional)
  Events CSV with columns `date,event_day` (0/1). If provided, event signal is used in modeling.

* `--dow INT`
  Day of week to score (0=Mon … 6=Sun). Default: most common in your data.

* `--meal Lunch|Dinner`
  Meal for scenario scoring. Default: `Dinner`.

* `--rain 0|1`
  Rain flag for the scenario. Default: `0`.

* `--event 0|1`
  Special event day. Default: `0`. (Used by lineup model.)

* `--dishwasher 0|1`
  Dishwasher down flag for the scenario. Default: `0`.

* `--inventory 0|1`
  Low inventory flag for the scenario. Default: `0`.

### Availability / Roster Filters (lineup model)

* `--include "A,B"`
  Servers that **must** appear in candidate teams.

* `--exclude "C,D"`
  Servers to **exclude** (PTO/unavailable).

* `--hosts "H1,H2"`
  Restrict host pool to specific names.

* `--managers "M1,M2"`
  Restrict manager pool to specific names.

### Ranking / Performance Controls

* `--top INT`
  How many top/bottom combos to save. Default: `25`.

* `--max-combos INT`
  Safety cap to avoid combinatorial explosion. Default: `50000`. If potential combos exceed this, server teams are sampled.

* `--plot` / `--no-plot`
  Enable/disable the Top-10 combos chart PNG. Default: off.

* `--week-plan`
  Also produce best lineup **for each DOW** under the same conditions.

* `--export-schedule`
  With `--week-plan`, also write `weekly_schedule_with_roles.csv` (Host/Manager/Team, Closers/Mid/First-Cut).

### Fairness Rotation (lineup model)

* `--overused "Name1,Name2"`
  People you want to rotate out (too many recent prime shifts).

* `--fairness-penalty FLOAT`
  Amount subtracted from the predicted total for **each** overused person in a lineup (servers + host + manager).
  Example: `--fairness-penalty 100` means “I’m willing to trade \~\$100 in predicted sales to rest one overused person.”

* `--no-fairness`
  Disable fairness re-scoring. Default: fairness **enabled** with penalty `75.0`.

### Accuracy Booster (per-server model)

* `--target-encode`
  Adds leakage-safe target encoding features for **server×DOW** and **server×meal**. Often improves ranking accuracy of individual servers.

---

## What the Script Prints

* **Fit Metrics**

  * `[Per-Server] R² … MAE …` (only if per-server input)
  * `[Shift-Combo] R² … MAE …`

* **Scenario Summary**

  * “Potential combos to score: …”
  * **HIGHEST predicted** lineup line
  * **LOWEST predicted** lineup line

* **Weekly Summary** (with `--week-plan`)

  * “Best servers for this week are: X, Y, Z.”
  * “Most effective managers: A, B, C.”
  * “Most effective hosts: …”

---

## Output Files

* `outputs/per_server_rankings.csv`
  Ranked list of **(host, manager, server)** with `predicted_server_sales` for your scenario.

* `outputs/top_combos.csv`
  Top N (host, manager, team-of-6) lineups with predicted sales (or fairness-adjusted score if enabled).

* `outputs/bottom_combos.csv`
  Worst N lineups (avoid).

* `outputs/top_combos_with_roles.csv`
  For each top lineup, each server’s **marginal** impact and assigned role: **Closer / Mid Shift / First Cut**.

* `outputs/week_plan.csv` (with `--week-plan`)
  Best lineup per day of week for the same scenario flags.

* `outputs/weekly_schedule_with_roles.csv` (with `--export-schedule`)
  Human-readable schedule: DOW, meal, Host, Manager, Team, Closers, Mid, First-Cut, Predicted Sales.

* `outputs/top_combos_top10.png` (with `--plot`)
  Horizontal bar chart of the top 10 lineups (fairness label shown when enabled).

---

## Example Commands

**1) Per-server input, rainy Sat Dinner, event day, chart:**

```bash
python foh_predictor.py --csv server_daily_sales.csv --dow 6 --meal Dinner --rain 1 --event 1 --plot
```

**2) Same, with target encoding for extra accuracy:**

```bash
python foh_predictor.py --csv server_daily_sales.csv --dow 6 --meal Dinner --rain 1 --event 1 --target-encode --plot
```

**3) Build a week plan & export a readable schedule:**

```bash
python foh_predictor.py --csv server_daily_sales.csv --meal Dinner --week-plan --export-schedule --plot
```

**4) Shift-level input, restrict roster & enforce rotation:**

```bash
python foh_predictor.py \
  --csv shift_sales_6servers.csv \
  --dow 5 --meal Dinner --rain 0 \
  --include Olivia \
  --exclude Lucas,James \
  --hosts Alice,David \
  --managers John,Sarah \
  --overused Olivia,Isabella \
  --fairness-penalty 100 \
  --plot
```

---

## Tips & Troubleshooting

* **CSV parse errors**
  Make sure the header names are **exact**. Use a plain comma-separated CSV, no stray quotes or tabs.

* **Small datasets**
  Start with per-server ranking (individual predictions) and treat lineup results as directional. Add more history for better generalization.

* **Fairness confusion**
  When fairness is enabled, the chart and CSVs will use the **fairness-adjusted** metric. The y-axis label will say “(fairness-adjusted)”.

* **Dishwasher/Inventory/Rain**
  These are treated as binary signals (0/1). If you have richer weather or inventory detail, you can extend the CSV and model later.

---

## Project Layout (suggested)

```
.
├── foh_predictor.py
├── server_daily_sales.csv          # or shift_sales_6servers.csv
├── outputs/
│   ├── per_server_rankings.csv
│   ├── top_combos.csv
│   ├── bottom_combos.csv
│   ├── top_combos_with_roles.csv
│   ├── week_plan.csv
│   ├── weekly_schedule_with_roles.csv
│   └── top_combos_top10.png
└── README.md
```

---

