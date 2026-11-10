"""
Parameters of HOOMD-engine simulations, and the checks that keep oxDNA parameters out of them.

HOOMD ensembles use HOOMD-native parameter names. The only names shared with oxDNA ensembles are those whose
meaning is identical in both engines: step counts (`steps`, stage lengths, `print_conf_interval`,
`print_energy_every`) and pypatchy's own engine-neutral settings (density, particle counts, staging, ...).
Nothing is translated: an oxDNA input parameter (`T`, `interaction_type`, `backend`, ...) in a HOOMD ensemble
is an error naming it, rather than being silently converted or ignored.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Union


class HOOMDParameterError(ValueError):
    """
    A HOOMD-engine simulation has parameters it can't use, or is missing ones it needs
    """
    pass


@dataclass(frozen=True)
class HOOMDParam:
    name: str
    kind: type  # float, int or str
    doc: str
    required: bool = False
    # default value, or a function of the other (resolved) parameters
    default: Union[None, Any, Callable[[dict[str, Any]], Any]] = None
    choices: Union[None, tuple] = None


HOOMD_PARAMS: dict[str, HOOMDParam] = {p.name: p for p in [
    HOOMDParam("kT", float, "Temperature, in energy units (the WCA repulsion strength sets the energy scale).",
               required=True),
    HOOMDParam("dt", float, "MD timestep.", required=True),
    HOOMDParam("patch_epsilon", float, "Depth of the Gaussian well between two complementary patches, before "
                                       "scaling by the interaction matrix and patch strengths.", required=True),
    HOOMDParam("patch_sigma", float, "Width (standard deviation) of the Gaussian patch well.", required=True),
    HOOMDParam("patch_rcut", float, "Cutoff of the patch-patch potential. Default: 3 * patch_sigma.",
               default=lambda p: 3 * p["patch_sigma"]),
    HOOMDParam("particle_epsilon", float, "Strength of the WCA repulsion between particle centres. "
                                          "Default: 1.", default=1.0),
    HOOMDParam("gamma", float, "Langevin translational drag coefficient. Default: 1 (HOOMD's default).",
               default=1.0),
    HOOMDParam("gamma_r", float, "Langevin rotational drag coefficient, about each body axis. "
                                 "Default: 1 (HOOMD's default).", default=1.0),
    HOOMDParam("nlist_buffer", float, "Neighbour list buffer distance. Default: 0.4.", default=0.4),
    HOOMDParam("device", str, "Device to run on. Default: gpu.", default="gpu", choices=("gpu", "cpu")),
    HOOMDParam("seed", int, "Random number seed (HOOMD uses the low 16 bits). Default: drawn at random when the "
                            "stage is built, and recorded in its stage.json.",
               default=lambda p: random.randrange(65536)),
    HOOMDParam("print_conf_interval", int, "Steps between trajectory frames (same meaning as in oxDNA).",
               required=True),
    HOOMDParam("print_energy_every", int, "Steps between energy log lines (same meaning as in oxDNA). "
                                          "Default: print_conf_interval.",
               default=lambda p: p["print_conf_interval"]),
]}

# pypatchy settings that mean the same thing whichever engine runs the simulation. particle type names
# (particle counts, e.g. "CT0") are also allowed; they depend on the particle set, so they're passed separately
ENGINE_NEUTRAL_PARAMS = {
    "steps",  # simulation / stage end step
    "allow_shortfall",  # staging
    "density",
    "num_assemblies",
    "box_size",
    "cluster_bond_energy",  # used by Stage.apply() when a stage resizes the box
}

# extra explanation for oxDNA names people are likely to carry over
OXDNA_HINTS = {
    "T": "HOOMD ensembles take the temperature as `kT`, in energy units. `T` is not translated.",
    "backend": "choose the HOOMD device with `device` (\"gpu\" or \"cpu\").",
    "CUDA_device": "the HOOMD engine runs on the first GPU visible to it; use CUDA_VISIBLE_DEVICES.",
    "interaction_type": "the HOOMD engine has its own patchy model (see pypatchy.patchy.hoomd_backend.model).",
}


def _coerce(param: HOOMDParam, value: Any) -> Any:
    """
    converts a value to the parameter's type, refusing anything that would change it (e.g. 2.5 for a step count)
    """
    if param.kind is int:
        as_float = float(value)
        if as_float != int(as_float):
            raise HOOMDParameterError(f"`{param.name}` must be a whole number, got {value!r}")
        return int(as_float)
    if param.kind is float:
        return float(value)
    value = str(value)
    if param.choices is not None and value not in param.choices:
        raise HOOMDParameterError(f"`{param.name}` must be one of {', '.join(param.choices)}; got {value!r}")
    return value


def resolve_hoomd_params(values: dict[str, Any],
                         particle_type_names: Iterable[str] = ()) -> tuple[dict[str, Any], set[str]]:
    """
    Checks a HOOMD simulation's parameters and fills in defaults.

    :param values: parameter name -> value, already merged by precedence (stage > sim > ensemble > server)
    :param particle_type_names: names of the simulation's particle types (their counts are parameters too)
    :return: (HOOMD parameters, including defaults; names of the parameters that took their default)
    :raises HOOMDParameterError: naming every parameter that isn't a HOOMD-engine or engine-neutral parameter,
        and every missing required one
    """
    allowed = set(HOOMD_PARAMS) | ENGINE_NEUTRAL_PARAMS | set(particle_type_names)
    unknown = sorted(set(values) - allowed)
    missing = sorted(name for name, p in HOOMD_PARAMS.items() if p.required and name not in values)
    if unknown or missing:
        lines = []
        if unknown:
            lines.append("These parameters aren't used by the HOOMD engine (oxDNA input parameters are not "
                         "translated to HOOMD ones): " + ", ".join(f"`{u}`" for u in unknown) + ".")
            lines.extend(f"  `{u}`: {OXDNA_HINTS[u]}" for u in unknown if OXDNA_HINTS.get(u))
            lines.append("  If they come from the oxDNA default parameter set, set \"default_param_set\" to null "
                         "(HOOMD ensembles don't load it unless asked to). Server configs' input_file_params "
                         "count too.")
        if missing:
            lines.append("Missing required HOOMD parameters: " + ", ".join(
                f"`{m}` ({HOOMD_PARAMS[m].doc})" for m in missing))
        raise HOOMDParameterError("\n".join(lines))

    resolved: dict[str, Any] = {}
    defaulted: set[str] = set()
    # explicit values first, so defaults that depend on other parameters can see them
    for name, p in HOOMD_PARAMS.items():
        if name in values:
            resolved[name] = _coerce(p, values[name])
    for name, p in HOOMD_PARAMS.items():
        if name not in resolved:
            resolved[name] = _coerce(p, p.default(resolved) if callable(p.default) else p.default)
            defaulted.add(name)

    for name in ("kT", "dt", "patch_sigma", "patch_rcut", "print_conf_interval", "print_energy_every"):
        if resolved[name] <= 0:
            raise HOOMDParameterError(f"`{name}` must be positive, got {resolved[name]}")
    return resolved, defaulted
