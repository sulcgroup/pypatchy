from input_output import *
from patchy_result_vis import *

results = choose_results()
target = choose_target(results)
chart = showPatchyResults(results, target, plot_relative=False)
