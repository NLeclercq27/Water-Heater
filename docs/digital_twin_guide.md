# Digital Twin — Water-Heater Pool Simulation Guide

## Overview

`scripts/digital_twin.py` simulates a **pool of electric water heaters** using a digital-twin approach. It runs a minute-by-minute control loop that:

1. Applies a per-heater **setpoint-tracking** control strategy (hysteresis-based).
2. Optionally **sheds power** (demand reduction) by switching off the hottest heaters.
3. Optionally **forces heaters on** by switching on the coldest heaters.
4. Records and exports electrical consumption, available storage, and per-heater temperature data.

---

## Quick start

```bash
# 1. Activate the conda environment
conda activate waterheaters

# 2. Move into the scripts folder
cd scripts/

# 3. Run the simulation
python digital_twin.py
```

Outputs are saved under `data/Simulations/` and the project root.

---

## File structure

| File / folder | Purpose |
|---|---|
| `scripts/digital_twin.py` | Main simulation script (this guide) |
| `scripts/digital_twin_clean.py` | Variant that imports real PV / home-consumption data from `simon.py` |
| `source/processFunctions_DT.py` | `WaterHeaterPool` class — pool-level logic (control strategies, simulation helpers, I/O) |
| `source/processFunctions_WH.py` | `WaterHeater` class — single-heater thermo-hydraulic model |
| `source/__init__.py` | Exposes `source.wh` and `source.dlt` |
| `data/Simulations/` | Simulation output CSVs |

---

## Configuration parameters

All tuneable parameters live in **Section 2** of the script.

### Heater counts

| Variable | Description |
|---|---|
| `N_random_HP` | Number of random heat-pump water heaters |
| `N_random_E` | Number of random electric-resistance water heaters |
| `N_VELIS` | Number of Ariston Velis units |
| `N_NUOS` | Number of Ariston Nuos units |

### Physical conditions

| Variable | Default | Description |
|---|---|---|
| `nx` | 40 | Vertical cells per heater (spatial discretisation) |
| `T_amb` | 19 + 273.15 K | Ambient temperature |
| `T_w_supply` | 14 + 273.15 K | Cold water supply temperature |
| `NDay` | 1 | Simulation duration in days |

### Schedules

#### Demand-reduction (switch-off)

```python
demand_reduction_schedule = [
    {
        'start_minute': 7 * 60 + 30,           # 07:30
        'end_minute':   8 * 60 + 30,           # 08:30
        'percentage_to_switch_off': 1,          # 0.0 – 1.0
    },
]
```

- **How it works**: During the time window the pool sorts heaters by temperature (hottest first) and switches them off until the cumulative shed power reaches the requested fraction.
- Set `percentage_to_switch_off` to `0` or set `start_minute == end_minute` to disable.

#### Forced switch-on

```python
switch_on_schedule = [
    {
        'start_minute': 3 * 60,                # 03:00
        'end_minute':   5 * 60,                # 05:00
        'power_to_switch_on': 0,               # [W]
    },
]
```

- **How it works**: During the time window the pool sorts heaters by temperature (coldest first) and forces them on until the cumulative activated power reaches the target.
- Set `power_to_switch_on` to `0` to disable.

### CSV inputs

| Variable | Description |
|---|---|
| `csv_file_path` | Semicolon-separated CSV with heater characteristics (columns: `Type`, `Volume (L)`, `Height (m)`, `Diameter (m)`, `Electric Power (W)`) |
| `fixed_profiles_path` | Semicolon-separated CSV with pre-defined water draw-off profiles (columns named `vdot_*`) |
| `use_csv` | `True` = load from CSV, `False` = generate random heaters |

---

## Control strategies (library)

The per-heater control strategy is set via the `control_strategy` argument of `WaterHeaterPool`. Available strategies are defined in `source/processFunctions_DT.py → control_functions()`:

| Strategy | Description |
|---|---|
| `tracking_SP` | Default. Hysteresis around setpoint: heats when T < SP − 3 K, stops when T > SP + 3 K |
| `mid_day_night` | Only starts heating at midnight or midday; stops at SP + 3 K |
| `full_load` | Heats to maximum temperature (≈ 90 °C) — useful for max storage estimation |
| `PV_strategy` | Turns resistor on when PV surplus exceeds home + EV consumption, or when an external activation signal is received. Requires extra kwargs: `P_Home`, `P_EV`, `PV_Gen`, `activation` |

### Pool-level overrides (new)

These are methods on the `WaterHeaterPool` class that override individual control decisions:

| Method | Description |
|---|---|
| `pool.collect_step_data(t)` | Gathers temperature, power and switch-status for every heater at step `t` |
| `pool.apply_demand_reduction(step_data, fraction, power_kW)` | Switches off hottest heaters — either by `fraction` (0–1) of active power **or** an absolute `power_kW` cap. If both are given, the absolute cap wins. |
| `pool.apply_forced_switch_on(step_data, power_W)` | Switches on coldest heaters until `power_W` watts are activated |
| `pool.load_pool_from_charact_csv(path)` | Loads heater specs from a CSV file and populates the pool |

---

## Outputs

| Output | Location | Description |
|---|---|---|
| Characteristics CSV | `data/Simulations/TEST_digital_twin_WH_charact_N.csv` | Physical parameters of each heater |
| Time-series CSV | `data/Simulations/TEST_digital_twin_WH_timeseries_N.csv` | Per-minute electrical consumption, water flow, mean temperature per heater |
| Step data CSV | `step_data_sorted.csv` (project root) | Per-heater per-step temperature and power data, sorted by temperature |
| Interactive plots | Opens in browser | Electrical consumption and available storage (Plotly) |

---

## Example: adding a second demand-reduction window

```python
demand_reduction_schedule = [
    {'start_minute': 7*60+30, 'end_minute': 8*60+30, 'percentage_to_switch_off': 1},
    {'start_minute': 18*60,   'end_minute': 20*60,   'percentage_to_switch_off': 0.5},
]
```

This sheds 100 % of active power between 07:30–08:30, and 50 % between 18:00–20:00.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'source'` | Make sure you run from `scripts/` or that the project root is on `PYTHONPATH` |
| `FileNotFoundError` for CSV | Check `csv_file_path` points to an existing file. The script falls back to random generation if the CSV is missing |
| Plotly plots don't open | Install a browser or use `fig.write_html()` / `fig.write_image()` instead of `fig.show('browser')` |
| `conda activate waterheaters` fails | Create the env with `conda env create -f environment.yml` |
