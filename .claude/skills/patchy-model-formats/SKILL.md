---
name: patchy-model-formats
description: Use when the task touches the specifics of a patchy-particle file format or model - which oxDNA contrib plugin/interaction_type goes with which pypatchy patchy_format writer and observable, what patches.txt/topology.top/interactions.txt actually contain for a given format (flavio, josh_flavio, lorenzo, subhajit, raspberry), model-specific params (DPS_sigma_ss, PATCHY_radius, repulsion sites), multidentate patch conversion (n_teeth/dental_radius/to_multidentate), or debugging a malformed-file/wrong-plugin error. Not for the ensemble-building mechanics themselves - see setup-run-ensemble for that - or for the analysis pipeline - see analyze-ensemble.
---

# Patchy particle models and file formats

oxDNA's patchy-particle support is spread across several `contrib/` plugins,
each with its own file format and observable, read/written through
`pypatchy/patchy/pl/patchyio.py`'s writer classes (`get_writer()`,
`writer_options()`). This is the reference for how those pieces fit together —
used from [[setup-run-ensemble]] when choosing a server config/format, and from
[[analyze-ensemble]] when loading a scene with `get_writer(...)`/`get_scene()`.

Picking pieces from different rows below produces files oxDNA can't parse, or
an interaction that doesn't match the topology — check
`get_writer(patchy_format).reqd_args()` for the exact file-name params a format
needs (these go in `CONST_PARAMS_KEY` or the server config's `input_file_params`).

## Plugin / format / observable concordance

| `contrib/` plugin | `interaction_type` | `patchy_format` (writer) | observable compiled into this plugin |
|---|---|---|---|
| `romano` | `PatchyShapeInteraction` | `"flavio"` (`FWriter`) — `patches.txt`/`particles.txt`; `"josh_flavio"` (`JFWriter`) adds allostery; `"subhajit"` (`SWriter`) is a third, differently-shaped file format also seen paired with this plugin | `PLClusterTopology` (`contrib/romano/src/Observables/PLClusterTopology.cpp`) |
| `rovigatti` | `DetailedPatchySwapInteraction` | `"lorenzo"` (`LWriter`) — needs `DPS_interaction_matrix_file` | `PatchyBonds` (`contrib/rovigatti/src/Observables/PatchyBonds.cpp`) |
| `evans` | `RaspberryPatchyInteraction` | `"raspberry"` (`RWriter`) | `RaspberryPatchyBonds` (no premade json in `~/.pypatchy/spec_files/observables/` — construct it inline: `Observable("clusters.txt", 1e7, ObservableColumn(name="RaspberryPatchyBonds", show_types=1))`) |

Each contrib plugin compiles to its own shared library; `plugin_search_path`
controls which one oxDNA loads, so an observable only works if it's compiled
into *that* plugin — this is why `PatchyBonds` needs `rovigatti` even though
its name suggests it's the generic option, and `PLClusterTopology` needs
`romano` despite sounding like a generic clusters observable too. Verify with
`grep -rl <ObservableName> $OXDNA_PATH/contrib/*/src/Observables/` if unsure,
rather than guessing from the name. This table reflects what's on this
machine's `~/.pypatchy/spec_files/server_configs/` (e.g. `home_flavio.json` =
romano/flavio, `subhajit.json` = romano/subhajit, `home.json` = rovigatti/lorenzo,
`raspberry.json` = evans/raspberry) — treat it as a starting point, not
guaranteed exhaustive or a strict one-plugin-one-format rule; confirm against
the actual server config and its `plugin_search_path` rather than assuming.
Also recall each server config's `oxdna_path` is a *specific checkout*, and
`plugin_search_path` names one `contrib/<model>/` inside it that must actually
be compiled there — different checkouts/machines may have different subsets built.

Existing premade observable jsons live in `~/.pypatchy/spec_files/observables/`
(`patchybonds.json`, `plclustertopology.json`), loadable by name via
`observable_from_file("patchybonds")` (`patchy_sim_observable.py`); a missing
json (like raspberry's, above) isn't a sign of breakage, just build the
`Observable` by hand.

## Multidentate conversion

**What it's for**: a plain ("unidentate") patch is one point on the particle
surface with a single binding direction (`a1`, optionally a torsion vector
`a2`). A single point contact can't mechanically resist twisting around the
bond axis, and can't represent a wide "sticky face" that should bind with
multiple simultaneous contacts. Multidentate conversion replaces each
unidentate patch with `n_teeth` smaller real sub-patches ("teeth") arranged in
a ring around the original patch direction, so two docked particles form
several simultaneous point-bonds instead of one — approximating a torsion-
resistant, extended patch out of several ordinary point patches. This is the
standard trick (Romano/Sciortino-style) for getting orientation-specific
"faces" out of oxDNA's inherently point-like patchy interactions, and is why a
polycube rule with 6 flat faces per cube often needs multidentate conversion
before it becomes a physically stable oxDNA simulation.

Call it via `PLParticleSet.to_multidentate(mdt_params)` (on a bare particle
set) or `PLPSimulation.to_multidentate(mdt_params)` (on a whole scene, doing
the same conversion to its particle types) — see the reference example
[setup_sims_programmatic.py](../../../examples/setup_sims_programmatic.py)
(via `polycube_rule_to_PL(rule).to_multidentate(...)`) and
[[analyze-ensemble]]'s scene-loading section for the two call sites.

`MultidentateConvertSettings` (`plparticleset.py`) fields:
- `n_teeth: int` — how many sub-patches to split each patch into.
- `dental_radius: float` — how far each tooth is displaced from the original
  patch center; bigger = a wider ring = stronger torsion-locking, but too big
  relative to particle size risks teeth colliding with neighboring patches or
  leaving the particle's surface. At least one of `n_teeth > 1` or
  `dental_radius > 0` is required.
- `follow_surf: bool` (default `False`) — how that displacement is applied.
  `False` offsets the tooth in a flat plane tangent to the patch (`position +=
  a2 * dental_radius`) — simple, fine for small radii. `True` instead rotates
  the tooth around the particle center so it moves along the sphere's actual
  curved surface (arc length `dental_radius`) — more accurate at larger radii
  or smaller particles, more expensive to reason about.
- `torsion: bool` (default `True`) — whether the source patches have a
  meaningful `a2`. If `True` and a source patch has no valid `a2` (missing, or
  not orthogonal to `a1`), conversion **raises**. If `False` and `a2` is
  missing, it fabricates an arbitrary orthogonal one and **prints a warning
  that alignment will be bad** — don't ignore that warning; it means the
  resulting teeth ring has no principled orientation.
- `energy_scale_method` (default `ENERGY_SCALE_LINEAR`) — since splitting one
  patch into `n_teeth` real patches would otherwise multiply that face's total
  bond energy by `n_teeth` when fully docked, this rescales each tooth's
  strength back down. `"linear"`/`ENERGY_SCALE_LINEAR` divides by `n_teeth`
  (the sane default); `"log"`/`ENERGY_SCALE_LOG` divides by `n_teeth *
  ln(n_teeth)` (the code's own comment calls this "basically made up" — treat
  as experimental); `"none"`/`ENERGY_SCALE_NONE` disables scaling entirely, so
  a fully-docked face ends up `n_teeth`× stronger than the original patch —
  rarely what you want.
- `alpha_scale_method` — a Lipari-Szabo-model scaling option; effectively
  unused (`ALPHA_SCALE_NONE` default, marked TODO in the source) — leave it
  alone unless you're specifically working on that.

Constraints and gotchas: conversion asserts `interaction_matrix().is_one_to_one()`
first — every color must bind exactly one other color; particle sets with more
exotic binding rules aren't supported. New tooth patches get freshly derived
colors and patch IDs (not the original color) — the algorithm mirrors the
angular winding direction for one side of each bonding pair so the two rings
of teeth actually line up tooth-for-tooth when the particles dock face-on;
don't try to predict a specific tooth's color from the original by hand, use
`mdt_rep(original_patch_id) -> list[PLPatch]` (on the resulting
`PLParticleSet`) to look it up instead.

Introspection: `particle_set.is_multidentate()`, `get_src_map()` (the
`PLMultidentateSourceMap` back to the pre-conversion unidentate set),
`mdt_rep(udt_id)` (teeth implementing one original unidentate patch). This is
exactly what backs the format-specific handling already covered below —
`raspberry`'s `# Patch Groups`/`iG` lines and `josh_flavio`'s `patch_idx =
i // n_teeth` indexing both exist specifically to record or reconstruct which
teeth belong to the same original patch.

## Per-format file layout and model-specific params

Detail on each writer in `patchyio.py`, for when the table above isn't enough
(e.g. writing a reader/converter, or debugging a malformed-file error from oxDNA).

**`flavio` (`FWriter`) / `josh_flavio` (`JFWriter`)** — three files: `particles.txt`
(one `particle_N = { type = N, patches = id,id,... }` block per type),
`patches.txt` (one `patch_N = { id, color, strength, position, a1, a2 }` block
per patch — full `a1`+`a2` means these patches support torsion), plus a
topology file whose 2 lines are just `nParticles nParticleTypes` and a flat
list of per-particle type IDs. `josh_flavio` adds one extra field per patch,
`allostery_conditional` — a boolean expression over other patches' bound-state
(from `PolycubeRuleCubeType.patch_conditional()`; `"true"` for an
unconditional/non-allosteric patch) — and needs the real `PolycubeRuleCubeType`
particle types passed as `particle_types=` at write time (`reqd_args()`
includes `particle_types`, unlike plain `flavio`), not just any `PLParticleSet`.

**`lorenzo` (`LWriter`)** — topology lists particle types (`nInstances nPatches
patchIDs patches_<type_id>.dat` per line, one geometry file per type containing
just raw XYZ patch offsets — no color/strength/a1/a2 stored there at all), plus
a separate `DPS_interaction_matrix_file` (default `interactions.txt`) of
`patchy_eps[i][j] = value` lines. The key subtlety: those `i`/`j` are **patch
IDs, not colors** — this format has no native color concept, so on read,
pypatchy reverse-engineers colors from the ID-keyed matrix via `assign_colors()`
just to fit its own `PLPatch`/color-based model. Model params: `DPS_sigma_ss`
(needed by the `lr_patchy` potential), `DPS_normalise_patches`,
`max_density_multiplier`; `use_edge` shows up in CUDA configs but isn't
patchy-model-specific.

**`subhajit` (`SWriter`)** — everything in one `topology.top` (no separate
patches/particles/interaction files): patch lines `iP idx color strength x y z
a1x a1y a1z a3x a3y a3z` (stores `a1`+`a3`, derives `a2 = a1 × a3`), particle
type lines `iC type_id patch_id,patch_id,...`, then one plain instance line per
particle: `-3 0 type_id radius` — per-type radius is baked directly into the
particle instances (defaults to 0.5 if omitted), rather than being a shared
particle-type property like the other formats. No interaction-matrix section
or file at all in this format as implemented — same-machine's `subhajit.json`
still declares `interaction_type: PatchyShapeInteraction` (a `romano`-family
interaction, like `flavio`), but the topology shape here doesn't match
`FWriter`'s `particles.txt`/`patches.txt`, so this only works against a
build/fork of `contrib/romano` customized to parse this single-file layout —
not a stock one (its server config's `plugin_search_path` pointing at
`alloassembly`, a separate checkout, is consistent with that).

**`raspberry` (`RWriter`)** — also one combined topology file, and the format
this project uses **repulsion sites**: a `# Repulsion points` section (`iR
x,y,z r` lines) defines a shared set of extra excluded-volume spheres, and each
particle type's `iC` line references which of those sites it uses by index
(`iC type_id count patch_id,... rsite_idx,...`). The default geometry
(`RPatchyTopology.__init__`, when not loaded from a file) is 8 sites at the
corners of a cube, `(±0.25, ±0.25, ±0.25)`, radius `0.25` each — i.e. every
particle type gets the same cubic steric "shell" around its patchy sphere
unless that default is overridden. The code currently **asserts all particle
types share identical repulsion-site geometry** on read — per-type custom
shapes aren't supported yet. Patches here store only `iP idx strength color
x,y,z a1x,a1y,a1z` (comma-separated coords, **no `a2`/`a3` at all** — this
format has no torsional patches). A `# Patch Groups` section (`iG` lines) lists
multidentate "tooth" groupings when `particle_types.is_multidentate()`. Unlike
`lorenzo`, the interaction matrix (`# Interactions`, `iI color1 color2
strength` lines) is embedded directly in the topology and **is** keyed by
color. `RWriter.write_as_mgl(...)` is a bonus visualization-only helper unique
to this writer — renders repulsion sites as extra spheres in `.mgl` output,
since standard viewers won't otherwise show them.

## Scene energies come from the real plugin (`interaction_potentials` is gone)

pypatchy-side scene analysis (`PLPSimulation` energies, `compute_scene_graph`,
`split_scene_by_clusters`, overlap checks when placing particles) uses the same
oxDNA interaction the simulation ran with, via oxpy: an `OxpyEnergyModel`
(`pypatchy/patchy/pl/oxpy_energy.py`) names the `interaction_type`,
`plugin_search_path` and model params, and `scene.energy_session()` loads the
scene into oxpy (written with the format's writer, CPU backend). Ensembles build
the model for you (`e.get_energy_model(sim, stage)`, attached by `get_scene`
and `Stage.apply`), so whatever plugin row above the run uses is automatically
the one analysis uses.

The old pure-Python re-implementations (`lr_excl_vol`/`fr_excl_vol`/`lr_patchy`/
`fr_patchy`, `plpotential.py`) were removed; an `interaction_potentials` key in
an old config is ignored with a `FutureWarning`. oxpy gives energies per
particle pair only, so patch-to-patch bonds (`compute_scene_multigraph`,
`iter_binding_patches`) are matched from geometry within bound pairs and raise
`PatchBondAmbiguityWarning` when not unique (e.g. wide multidentate teeth).
Note the bound-pair criterion is total pair energy (excluded volume included),
whereas e.g. `PLClusterTopology` uses the patchy term alone.
