# Visualizing configurations

## ppview (recommended)

[ppview](https://github.com/m-matthies/ppview) is a browser-based viewer for patchy-particle
simulations. It reads the files oxDNA and pypatchy already write, so nothing needs converting.
Drag the files in and it detects the format from their contents.

- **Hosted:** <https://m-matthies.github.io/ppview>. Drag files onto the page, or Ctrl/Cmd+O.
- **Local:** `git clone https://github.com/m-matthies/ppview && cd ppview && npm install && npm start`
  (needs a Rust toolchain with the `wasm32-unknown-unknown` target).

Which files to give it:

| Format | Files |
|---|---|
| `raspberry` | `init.top` + a conf (`last_conf.dat`) or the trajectory (`trajectory.dat`) |
| `flavio` / `josh_flavio` | `init.top` + conf/trajectory + `particles.txt` + `patches.txt` |
| `lorenzo` | topology + conf/trajectory (listed as supported by ppview; less tested with pypatchy output) |

Useful extras:

- the oxDNA **`input`** file, so ppview knows `PATCHY_radius` and the print intervals;
- **`observables.json` plus the observable output** (e.g. `clusters.txt`). ppview then draws
  bonds and opens a clustering panel, matching observable output to trajectory frames by step
  number.

To find the files for a simulation:

```python
with find_ensemble(name="3dX_T_sweep", date="2026-11-18") as e:
    sim = e.get_simulation(T=0.025, duplicate=0)
    stage = e.sim_most_recent_stage(sim)
    print(e.folder_path(sim, stage))
```

Keys: `←`/`→` step frames (with Shift, 10 at a time), `Home`/`End` jump to the first/last
frame, `Space` plays, `P` takes a screenshot.

## oxView

`e.show_conf(sim, timepoint)` writes a temporary configuration and opens it in oxView through
`oxDNA_analysis_tools`. This only works in a Jupyter notebook and for the `flavio` format (oxView
needs `particles.txt`/`patches.txt`). `e.show_last_conf(...)` is currently unreliable. ppview is
the better option in most cases.

oxView is still the right tool for the DNA-origami output of the
[patchy → origami converter](08-patchy-to-origami.md).

## Other routes

- **glTF**, for rendering in Blender and similar:
  `pypatchy.patchy.pl.patchy_exports.to_gltf(scene, "out.glb", particle_shape="sphere")` (or
  `"cube"`).
- **MGL**, for older patchy viewers: `scene.export_to_mgl(filename, patches_w)`.
  `RWriter.write_as_mgl(...)` also draws raspberry repulsion sites.
- The polycubes **klossar** web app
  (<https://akodiat.github.io/polycubes/klossar>) accepts dropped `particles.txt`/`patches.txt`
  files and assembles the particle set in its own (non-oxDNA) stochastic model. That's a quick
  way to check a particle set's binding logic, but make sure the interaction matrix is what you
  expect.
- **Cluster graphs** (not 3D), from analysis: `vis_lib.show_clusters(...)`; see
  [Analysis](06-analysis.md#plotting).
