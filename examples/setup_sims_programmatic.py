"""
goal for this example is to construct, setup, and run a patchy particle ensemble without
using find_ensemble

further goals: should use other external jsons (e.g. observables and server specs) as little as possible

"""

import datetime

from pypatchy.patchy.param_val_types import PARTICLE_TYPES_KEY
from pypatchy.patchy.simulation_ensemble import build_ensemble, EXPORT_NAME_KEY, OBSERABLES_KEY, \
    DEFAULT_PARAM_SET_KEY, ENSEMBLE_SETUP_DATE, CONST_PARAMS_KEY, ENSEMBLE_PARAMS_KEY, SERVER_SETTINGS_KEY
from pypatchy.patchy.pl.plpatchylib import polycube_rule_to_PL
from oxpy_utils.utils.observable import Observable, ObservableColumn
from pypatchy.polycubeutil.polycubesRule import PolycubesRule
from pypatchy.server_config import PatchyServerConfig

rule = PolycubesRule(rule_str="|1||-4||3:1_||||-3:3|-3:2_5||4:1|-1:2||_2:2||2:2||2:2|_-2:1|-2||-5:1||-5:2")
particle_types = polycube_rule_to_PL(rule)

# hardcode interaction strengths
particle_types.interaction_matrix()[(-21, 21)] = 2.
particle_types.interaction_matrix()[(-22, 22)] = 2.
particle_types.interaction_matrix()[(-22, 22)] = 2.66
particle_types.interaction_matrix()[(-23, 23)] = 2.
particle_types.interaction_matrix()[(-24, 24)] = 2.

e = build_ensemble(
    cfg={
        EXPORT_NAME_KEY: "menger_crystal",
        PARTICLE_TYPES_KEY: particle_types,
        OBSERABLES_KEY: [
            Observable(
                "clusters.txt",
                1e7,
                ObservableColumn(name="PatchyBonds", show_types=1)
            )
        ],
        CONST_PARAMS_KEY: {
            "num_assemblies": 20,
            "steps": 1e9,
            "density": 0.06,
        },
        ENSEMBLE_PARAMS_KEY: [
            ("T", [0.04, 0.06, 0.08]),
        ],
        SERVER_SETTINGS_KEY: PatchyServerConfig(
            name="menger_crystal_local",
            oxdna_path="~/oxDNA",
            patchy_format="flavio",
        ),
        DEFAULT_PARAM_SET_KEY: None
    },
    mdt={
        ENSEMBLE_SETUP_DATE: datetime.datetime.now()
    }
)

# if we use ensemble as a context manager it will automatically save the changes when we exit context
with e:
    pass
