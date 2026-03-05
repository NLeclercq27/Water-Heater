# =============================================================================
# digital_twin.py — Water-Heater Pool Simulation (Digital Twin)
# =============================================================================
#
# This script simulates a pool of electric water heaters using a digital-twin
# approach.  It can load heater characteristics from a CSV, run a minute-by-
# minute control loop with hysteresis-based setpoint tracking, and optionally
# apply pool-level demand-reduction (switch-off) and forced switch-on schedules.
#
# Usage:
#   conda activate waterheaters
#   cd scripts/
#   python digital_twin.py
#
# See docs/digital_twin_guide.md for a full description.
# =============================================================================

# ── 1. Imports ───────────────────────────────────────────────────────────────

import os
import sys
import time

import pandas as pd

# Add project root to path so the `source` package can always be found
root_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_folder)

import source as procF  # exposes source.wh and source.dlt

# ── 2. Simulation parameters ────────────────────────────────────────────────

# Wall-clock timer
tic = time.perf_counter()

# Number of water heaters by type
N_random_HP = 0        # Random heat-pump water heaters
N_random_E  = 10       # Random electric-resistance water heaters
N_VELIS     = 0        # Ariston Velis units
N_NUOS      = 0        # Ariston Nuos units

# Simulation duration
NDay = 1               # Number of days

# Spatial discretisation & ambient conditions
nx         = 40                  # Vertical cells per water heater
T_amb      = 19 + 273.15        # Ambient temperature [K]
T_w_supply = 14 + 273.15        # Cold-water supply temperature [K]

# ── Demand-reduction schedule ──
# Each entry defines a time window (minutes from midnight) and how much power
# to shed.  Hottest heaters are switched off first.
# You can specify the target in THREE ways (pick one per entry):
#   • 'percentage_to_switch_off': fraction 0.0–1.0 of active power
#   • 'power_to_shed_kW':         shed exactly this many kW
#   • 'power_cap_kW':             cap total pool power at this level [kW]
# Priority: power_cap_kW > power_to_shed_kW > percentage_to_switch_off
# Heaters shed inside a window stay off for the ENTIRE window (latching).
demand_reduction_schedule = [
    {
        'start_minute': 7 * 60 + 30,   # 07:30
        'end_minute':   8 * 60 + 30,   # 08:30
        # 'percentage_to_switch_off': 1,   # 100 % of active power
        # 'power_to_shed_kW': 500,         # shed exactly 500 kW
        'power_cap_kW': 300,               # allow at most 300 kW total
    },
]

# ── Forced switch-on schedule ──
# Each entry defines a time window and the absolute power [W] to force on.
# Coldest heaters are switched on first.
switch_on_schedule = [
    {
        'start_minute': 3 * 60,        # 03:00
        'end_minute':   5 * 60,        # 05:00
        'power_to_switch_on': 0,        # [W] — 0 = disabled
    },
]

# ── CSV paths ──
# Heater characteristics CSV (semicolon-separated)
csv_file_path       = os.path.join(root_folder, 'TEST_digital_twin_WH_charact_1.csv')
# Pre-defined hot-water draw-off profiles (optional)
fixed_profiles_path = os.path.join(root_folder, 'TEST_digital_twin_WH_timeseries_1.csv')

# Set to True to load heater specs from CSV; False to generate randomly
use_csv = True

# ── 3. Build the water-heater pool ──────────────────────────────────────────

pool = procF.dlt.WaterHeaterPool(
    N_random_HP=N_random_HP,
    N_random_E=N_random_E,
    N_VELIS=N_VELIS,
    N_NUOS=N_NUOS,
    nx=nx,
    T_w_supply=T_w_supply,
    T_amb=T_amb,
    control_strategy='tracking_SP',
)

# Load heater specs from CSV or generate a random pool
if use_csv and os.path.exists(csv_file_path):
    pool.load_pool_from_charact_csv(csv_file_path)
else:
    pool.generate_pool()

# ── 4. Time-step simulation loop ────────────────────────────────────────────

# Initial switch states
switch1 = False
switch2 = False

# Initial probe temperatures for every heater: (T_tank1, T_tank2)
T_probe = [(55 + 273.15, 55 + 273.15)] * len(pool.pool_WH)

# Allocate time vectors, result arrays, etc.
pool.initialize_sim(NDay)

# Collector for per-step sorted heater data (saved to CSV at the end)
step_data_all = []

# Persistent latch: heaters shed by demand reduction stay off for the
# entire DR window.  Cleared automatically when outside any window.
latched_off_heaters = set()

for t in range(len(pool.time_vect_com)):
    # Reset cumulative electrical power for this time step
    pool.P_el_vect_cum = 0

    turned_on_heaters = []
    heaters_already_on = []
    step_data = []

    # ------------------------------------------------------------------
    # 4a. From the second time step onward, apply the individual control
    #     strategy to every heater and collect state data.
    # ------------------------------------------------------------------
    if t > 0:
        for cnt_wh, WH in enumerate(pool.pool_WH):
            T_SP = 55 + 273.15  # Set-point temperature [K]
            switch1, switch2 = pool.control_functions(
                WH, t * 60,
                pool.T_probe_2Dlist[cnt_wh][t - 1],
                T_SP,
                strategy=pool.pool_control_strategy,
            )

        # Re-enforce latched-off heaters BEFORE collecting data,
        # because control_functions above may have turned them back on.
        for WH in latched_off_heaters:
            WH.switch1 = False

        # Collect per-heater state using the library helper
        step_data = pool.collect_step_data(t)

        # Sort hottest-first (for demand reduction) and store for CSV
        step_data_sorted_off = sorted(
            step_data, key=lambda x: x['temperature_1'], reverse=True
        )
        step_data_all.extend(step_data_sorted_off)

        # ------------------------------------------------------------------
        # 4b. Demand reduction — switch off hottest heaters (latched)
        # ------------------------------------------------------------------
        current_minute = t  # time vector is 1-min steps
        in_dr_window = False
        for schedule in demand_reduction_schedule:
            if schedule['start_minute'] <= current_minute < schedule['end_minute']:
                in_dr_window = True
                result = pool.apply_demand_reduction(
                    step_data,
                    fraction_to_shed=schedule.get('percentage_to_switch_off'),
                    power_to_shed_kW=schedule.get('power_to_shed_kW'),
                    power_cap_kW=schedule.get('power_cap_kW'),
                    already_shed=latched_off_heaters,
                )
                latched_off_heaters = set(result)

        if not in_dr_window:
            latched_off_heaters = set()

        # ------------------------------------------------------------------
        # 4c. Forced switch-on — start coldest heaters
        # ------------------------------------------------------------------
        for schedule in switch_on_schedule:
            if schedule['start_minute'] <= current_minute < schedule['end_minute']:
                if schedule['power_to_switch_on'] != 0:
                    turned_on_heaters, heaters_already_on = pool.apply_forced_switch_on(
                        step_data, schedule['power_to_switch_on']
                    )

    # ------------------------------------------------------------------
    # 4d. Simulate every heater for this time step
    # ------------------------------------------------------------------
    for cnt_wh, WH in enumerate(pool.pool_WH):
        if WH in heaters_already_on:
            # Heater was already ON — keep it on
            T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, True, False)

        elif WH in turned_on_heaters:
            # Forced on by the switch-on schedule — run one step, then reset
            T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, WH.switch1, False)
            WH.switch1 = False
            turned_on_heaters.remove(WH)

        elif WH in latched_off_heaters:
            # Shed by demand reduction — force off (latched)
            T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, False, False)

        else:
            # Default: hysteresis-based setpoint tracking
            T_SP = 55 + 273.15
            switch1, switch2 = pool.control_functions(
                WH, t * 60, T_probe[cnt_wh], T_SP,
                strategy=pool.pool_control_strategy,
            )
            T_probe[cnt_wh] = pool.WH_iteration(WH, t, cnt_wh, switch1, switch2)

    # Record pool-level results for this time step
    pool.record_results(t)

# ── 5. Post-processing ──────────────────────────────────────────────────────

# Standard pool plots (electrical consumption, available storage)
pool.plot_consumption()
pool.plot_available_storage()
pool.save_results_csv('TEST_digital_twin')

# Save per-step sorted heater data
df_sorted = pd.DataFrame(step_data_all)
df_sorted.to_csv(os.path.join(root_folder, 'step_data_sorted.csv'), index=False)

# Wall-clock time
toc = time.perf_counter()
print(f'Simulation time: {toc - tic:.2f} s')
