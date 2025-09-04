"""
goal for this example is to construct, setup, and run a patchy particle ensemble without
using find_ensemble

further goals: should use other external jsons (e.g. observables and server specs) as little as possible

"""

import datetime

from pypatchy.patchy.simulation_ensemble import build_ensemble
from pypatchy.patchy.patchy_sim_observable import Observable
from ipy_oxdna.utils.observable import Observable, ObservableColumn
from pypatchy.polycubeutil.polycubesRule import PolycubesRule

rule = PolycubesRule("|1||-4||3:1_||||-3:3|-3:2_5||4:1|-1:2||_2:2||2:2||2:2|_-2:1|-2||-5:1||-5:2")
particle_types = polycubes_rule_to_pl(rule)

# hardcode interaction strengths
particle_types.interaction_matrix()[(-21, 21)] = 2.
particle_types.interaction_matrix()[(-22, 22)] = 2.
particle_types.interaction_matrix()[(-22, 22)] = 2.66
particle_types.interaction_matrix()[(-23, 23)] = 2.
particle_types.interaction_matrix()[(-24, 24)] = 2.

e = build_ensemble(
    cfg={
        EXPORT_NAME: "menger_crystal",
        PARTICLE_TYPES: particle_types,
        OBSERVABLES_KEY: [
            Observable(
                name= "clusters.txt",
                print_every= 1e7,
                *(ObservableColumn(name="PatchyBonds", show_types=1))
            )
        ],
        "server_config": PatchyServerConfig(),
        DEFAULT_PARAM_SET_KEY: None
    },
    mdt={
        ENSEMBLE_SETUP_DATE_KEY: datetime.datetime.now()
    }
)

# if we use ensemble as a context manager it will automatically save the changes when we exit context
with e:
    pass
