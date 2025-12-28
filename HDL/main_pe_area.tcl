set_db init_lib_search_path {.} -quiet
read_lib stdcells.lib 

set_db init_hdl_search_path {.} -quiet 
read_hdl -sv {processing_element.v}                     
                           
elaborate
vcd design:PE

set_time_unit -nanoseconds
create_clock [get_ports clk] -name main_clk -period 10

set_db dp_area_mode true       -quiet

set_db syn_generic_effort high -quiet
set_db syn_map_effort     high -quiet
set_db syn_opt_effort     high -quiet

syn_generic
syn_map
syn_opt

report_area > area.rpt
report_timing > timing.rpt
report_power > power.rpt

puts "RUNTIME : [get_db real_runtime]"

exit