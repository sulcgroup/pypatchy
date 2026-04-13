import libtlm

from pypatchy.polycubeutil.polycubesRule import PolycubesRule
from pypatchy.polycubeutil.polycube_structure import PolycubeStructure
from pypatchy.structure import TypedStructure

rule = PolycubesRule(rule_str="1|1|1|1|1|1_-1|-1|-1|-1|-1|-1")

params = libtlm.TLMParameters() # placeholder. would be good to have kwargs option

my_topology = TypedStructure() # placeholder

# callback to halt simulation. later we will need to write a libtlm.does_not_match_topology function that generates a c++ callback. can be a placeholder for now.
my_halt_condition = libtlm.does_not_match_topology(my_topology)

with PolycubesProgram(params, "RankedValence") as program: # pass simultion type as string ID
    program.set_halt_condition(my_halt_condition)
    program.run_for(500) # run for 500 steps
    sim_results = []
    for sim in program.get_simulation_results():
        polycubes: list[PolycubesStructure] = sim.get_polycubes()
        sim_results.append(polycubes)
# exit to context mgr releases c++ assets