# Known issues

Problems found while writing these docs (November 2026). They were traced in the source and not
all reproduced by running code. Paths are relative to the repository root unless noted.
**Remove entries as they're fixed.**

## Setup and configuration

- **`configure` overwrites spec files.** `python -m pypatchy.configure` copies the repository's
  `spec_files/` over `~/.pypatchy/spec_files/` every time it runs (`pypatchy/configure.py:42-58`),
  and writes a `settings.cfg` template with one user's paths and the retired `agave_classic`
  config (`:20-26`).
- **Stale templates.** `spec_files/server_configs/agave_*.json` target a retired cluster;
  `raspberry.json` (local) and the Dockerfile's `fr.json` still carry the deprecated
  `interaction_potentials`. The shipped `default.json` still has `num_teeth`/`dental_radius` and
  uses `init.conf` where installed copies use `init.dat`.
- **Observable spec name mismatch.** The repo ships `spec_files/observables/patchbonds.json`.
  Some installs have `patchybonds.json`.
- **`subhajit.json`** points at the alloassembly Romano plugin, which can't read the
  `subhajit` format.
- **`test_prerequisites.py::test_patchy_plugins`** checks every server config, so it fails on the
  template configs `configure` installs.

## Ensembles

- **Default-param-set values override server `input_file_params`** (merge order at
  `simulation_ensemble.py:533`, lookup at `:889-900`). That's surprising; server-specific
  settings arguably should win.
- **`find_ensemble(cfg=...)`** overwrites existing metadata (`:564`). Its "metadata already
  exists" check uses the cfg file name rather than `export_name` (`:233`), so it can check the
  wrong file.
- **Old nested particle-level groups** `{"name": ..., "value": {"particle_type_levels": {...}}}`
  silently don't work (`ensemble_parameter.py:28-32`, only unwrapped when `"type"` is present).
- **Old key names** `num_steps`, `print_conf_every` are silently ignored rather than warned about.
- **Implicit `"default"` stage has `allow_shortfall=True`** (`simulation_ensemble.py:1071-1088`),
  so interrupted unstaged runs count as complete in `sim_done`, `check_stage`, and
  `start_simulations`.
- **`start_simulations()` with an interrupted staged sim** raises `IncompleteStageError` out of
  `is_setup_done` (`:2062`), which only catches `NoStageTrajError`.
- **`sim_done()`** raises `NoStageTrajError` for a sim that never ran (`:1322-1327`).
- **Premade-observable keys** differ between the creating session (spec name, `:491`) and after
  reload (file name, via `:1794`).
- **`show_last_conf(**kwargs)`** calls `sim_get_param` without a parameter name (`:1456`), and
  checks the global default writer instead of the ensemble's (`:1459`).
- **`run_serial` / `is_run_serial`** are never read (`:2257`). `examples/run_ramped_octohedra.py`
  sets it to no effect.
- **Unused constants** `PARTICLES_KEY`, `METADATA_FILE_KEY`, `LAST_CONTINUE_COUNT_KEY`
  (`:66`, `:76-77`).
- **`analysis_lib.py:68`** docstring refers to the removed `get_input_file_param()`.

## Analysis

- **Only DataFrame steps are cached to disk.** `AnalysisPipelineStep.cache_data` ends in a
  placeholder comment (`pypatchy/analysis/analysis_pipeline_step.py:101`); graph/raw caching was
  removed in `c4de095`. `analysis_status()` always reports graph steps as missing.
- **Cache is keyed by step name only**, and a result with fewer rows never overwrites a cached
  one with more (`analysis_pipeline_step.py:89-95`).
- **Custom steps break ensemble reload.** Saved as type + name only (`sim_ensemble_json.py:272-277`).
  `build_ensemble` always uses the base `SimEnsembleJsonEncoder` (`simulation_ensemble.py:542`),
  ignoring `find_ensemble(loader=...)`, so reload raises `unknown step type` (`sim_ensemble_json.py:265`).
- **Step (de)serialization bugs** (`sim_ensemble_json.py`):
  - `ClassifyPolycubeClusters` saves `expected_edge_length` and `edge_distance_tolerance` swapped (`:286-287`).
  - `ComputeSpecGroupClusterYield` saves `aggregate_over`, but the attribute is
    `params_aggregate_over`, so saving raises `AttributeError`.
  - `MatchGraphToCrystal` reloads `target["bindings"]` from a list (`:213-216`).
- **`set_nocache(True)`** disables only the in-memory cache, not the disk cache.
- **`LoadEnergies`**: `raise ValueEr(...)` typo (`analysis_lib.py:193`).
- **`BlobsFromClusters`** stores `[]` at every timepoint (`analysis_lib.py:232`).
- **`MatchGraphToCrystal`** calls `target.compare`, which no `Structure` defines.
- **`ClassifyPolycubeClusters`**: `expected_edge_length`/`edge_distance_tolerance` only feed
  `engage_filter`, which is never called.
- **`ComputeClusterYield` docstring** says yield is at most 1. It's a sum of size ratios, so the
  maximum is about `num_assemblies` (`analysis_lib.py:831`).
- **`ComputeSpecGroupClusterYield`** renames index labels instead of columns (`:1012-1015`), so
  it likely outputs NaN.
- **`ClassifyClusters`** type-hints `nx.Graph` targets, but `Structure(graph=...)` asserts a
  `MultiDiGraph` with `dirIdx` edges (`structure.py:74-82`).
- **`get_data` with a complete parameter tuple** appears to access `.param_vals` on a plain
  tuple (`simulation_ensemble.py:2446`).
- **`get_data` with no `time_steps`** excludes the final timepoint (exclusive range stop,
  `:2390-2396`).
- **Parallel `get_data`**: the `multiprocessing.Lock()` is created per call (`:2429-2438`), so it
  doesn't lock across processes. Worker results don't reach the parent's memory cache.
  `can_parallelize()` is defined but never called.
- **Default `AnalysisPipelineStep.draw()`** indexes a 4-item list with data type values up to 4
  (`analysis_pipeline_step.py:146,155`).
- **`vis_lib`**: `YieldCurveFigure`/`PolycubesFigure` never store `plt_args`, and the `"col"`
  branch reads `kwargs["cols"]` (`vis_lib.py:566-573`).
  `plot_last_conf_cluster_size_histograms` uses undefined names. `plot_total_graph` requires
  the step object, not a name.
- **`configure` still creates `input/targets/`**, which nothing reads any more.

## Formats and models

- **Raspberry patch width is doubled.** The plugin's pair α is the sum of the two patches' α
  (`oxDNA/contrib/evans/src/Interactions/RaspberryInteraction.cpp:479`), and `RWriter` writes no
  per-patch α, so each patch gets `PATCHY_alpha` and the effective width is 2 × `PATCHY_alpha`.
  Existing raspberry runs with `PATCHY_alpha = 0.12` have an effective α of 0.24. Decide
  whether that's intended.
- **Raspberry strength may be squared.** `iI` values are already `sqrt(s1·s2)`
  (`plpatch.py:256-263`), and the plugin multiplies by `sqrt(s_i·s_j)` again
  (`RaspberryInteraction.cpp:481`). Non-unit strengths end up squared.
- **Raspberry without angmod** isn't shifted at the cutoff (`RaspberryInteraction.cpp:1116-1117`),
  a ~0.01ε energy jump.
- **Raspberry signal passing (`iO`)** looks non-functional: the parser gets the whole line,
  including the `iO` token (`:409`, `:614`), and `m_ParticlesCanChangeState` is never filled.
  `RaspberryParticleStates` declares `get_settings` but never defines it.
- **`RWriter.read_particle_types`** unpacks exactly 6 tokens from `iP` lines (`patchyio.py:1345`),
  so it fails on files with α/state fields. `RWriter.read_scene` defaults to conf 0 while other
  readers default to the last conf.
- **`FWriter` writes `a2 = 0,0,0`** for patches without `a2` (`patchyio.py:451-455`). oxDNA
  normalizes that to NaN, and torsion is on by default.
- **`JFWriter`** needs a `num_teeth` kwarg that isn't in `reqd_args()` (`patchyio.py:682`), and
  writes the conf twice.
- **`LWriter`** sorts the caller's scene in place (`:905`), never writes self-pair interactions
  (`:950-959`), and its reader's regex rejects negative or scientific-notation values (`:832`).
- **`SWriter` doesn't match `PHBInteraction`**: it writes `a3` where PHB reads `a2`, and strand 0
  for every particle, which disables patch binding under PHB. `read_top` passes the particle list
  as `intmat` (`:1142`), radius is recorded only from a type's second instance (`:1129-1133`),
  and `read_scene` adds no particles (`:1150-1161`).
- **Color self-binding thresholds disagree:**
  - `PLPatch.can_bind`: |c| ≥ 20 binds −c;
  - `FWriter.load_patches`: 0 < c < 21 binds itself;
  - Romano C++: |c| < 10 binds itself;
  - Raspberry C++ default: |c| < 20 binds itself.
- **Torsional multidentate colors** start at ±20 and assume |c| ≥ 21 (`plparticleset.py:580-604`).
- **CUDA DPS ignores `DPS_no_three_body`** (`oxDNA/contrib/rovigatti/.../CUDADetailedPatchySwapInteraction.cu:416`).
- **Romano partial torsion**: the force adds an a2×a2 torque that the energy doesn't have
  (`PatchyShapeInteraction.cpp:817`).
- **`patchy_scripts.py` helpers are broken:**
  - `convert_multidentate` swaps `n_teeth`/`dental_radius` (`:27`);
  - `convert_udt_files_to_mdt` passes `follow_surf` into the `torsion` slot and slices a `Path` (`:62`, `:64`);
  - `lorenzian_to_flavian` sets the directory on one writer instance and reads with another (`:113-118`);
  - `convert_flavian_to_lorenzian` instantiates an abstract class (`:83`);
  - `best_traj_contiguous_conf` never updates for positive scores (`:157`).
- **`pljsonrw.py`**: the `PLPatch` branch is missing a `return` (`:87`), and `decode_patch` passes
  the wrong keyword names.
- **`mgl_to_pl`** calls `PLParticleSet.normalize`, which is commented out
  (`plparticleset.py:475`).
- **`register_writer`** is type-hinted as taking an instance, but must be given the class.

## Patchy → origami

- **3'/5' ends.** `strand_3p()` returns the 5' base, given oxpy-utils' 3'→5' storage
  (`oxpy-utils/.../dna_structure.py:868-870`). Code using it for alignment and sticky-end start
  positions (`dna_particle.py:159-161,291`; `patchy_origami_convert.py:302,344,374,558-559,619`)
  may be using the wrong end, while other code uses `[0]` (the actual 3' end).
- **`assign_color_sequence(update_rc=True)`** writes the reverse complement over the color you
  assigned (`patchy_origami_convert.py:243-246`).
- **`position_particles()`** is never called automatically, because `dna_particles` is
  initialized to `{}`, not `None` (`origami_deabstractor.py:39,168`).
- **Unused constructor arguments:** `particle_delta`, `bond_length`, `cos_theta_max`,
  `flexable_patch_distances`.
- **`save_oxview`**:
  - `prettyprint_json` is inverted (`:724`);
  - the docstring says RGB triplets but the code expects `"#rrggbb"`;
  - `color_by_type` indexes `dna_particles` by cluster index rather than UID (`:715`).
- **`export_stickys_staples`**:
  - the flag docstrings don't match their behavior;
  - `is_polycube = True` is hard-coded (`:824`);
  - `iter_sticky_staples` drops its flags when recursing (`:764`).
- **Smaller bugs:**
  - the assert at `:616` has a misplaced parenthesis;
  - `patch_strand_id` has an off-by-one (`dna_particle.py:152`);
  - `DNAParticle.linked_particle` is shadowed by an instance attribute.
- **MGL scenes** likely fail the unidentate-source check (`:282`).
- **Tests:**
  - `test_oxview_recolor.py` uses `get_writer` without importing it (`:148`);
  - `icosahedron()` uses the path `test_file/`;
  - `dump_monomers` in tests writes to the real `~/.pypatchy/output`.
