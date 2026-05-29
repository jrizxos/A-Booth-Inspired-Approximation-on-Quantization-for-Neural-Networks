# ============================================================
# LIBRARIES
# ============================================================
set_app_var search_path [list ../Systolic/Multiplier_only/asp2 ../freepdk-45nm $search_path]

set synthetic_library  dw_foundation.sldb
set target_library     stdcells.db
set link_library       "* $target_library $synthetic_library"


# ============================================================
# READ RTL
# ============================================================

analyze -format sverilog {asp2.v}

elaborate ASP2
link


# ============================================================
# CLOCKS / CONSTRAINTS
# ============================================================
set_units -time ns
create_clock -name main_clk -period 10 [get_ports clk]
set_input_delay 0 -clock main_clk [all_inputs]
set_output_delay 0 -clock main_clk [all_outputs]

# ============================================================
# RUN SYNTHESIS
# ============================================================
compile_ultra

# ============================================================
# REPORTS
# ============================================================
file mkdir reports

report_area   > reports/area_dc.rpt
report_timing -max_paths 1 > reports/timing_dc.rpt
report_power  > reports/power_dc.rpt

quit