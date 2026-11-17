---
name: ppview-visualization
description: Use when asked to look at, visualize, screenshot, or sanity-check a patchy-particle configuration or trajectory (raspberry, flavio, lorenzo formats) in ppview - the browser-based patchy particle viewer (github.com/m-matthies/ppview). Covers which pypatchy/oxDNA files to hand ppview for each format, opening it interactively (hosted or `npm start`), and rendering a PNG headlessly so Claude can see a conf itself (render.js in this skill), including stepping to a trajectory frame and showing cluster/bond observables. Not for matplotlib plots of analysis data (see analyze-ensemble), for oxDNA nucleotide/DNA-origami confs (oxView, or see dna-structure), or for the file formats themselves (see patchy-model-formats).
---

# Visualizing patchy-particle confs with ppview

ppview is a React/Three.js viewer for patchy-particle (and oxDNA) simulations. It reads the
files oxDNA and pypatchy already write, so there is no conversion step: drop the topology and
conf/trajectory in and it detects the format **by content**, not by file name. This replaces
the old oxView route (`PatchySimulationEnsemble.show_conf`, which still calls oxView's
`from_path`, and `show_last_conf`, which carries a `TODO: ppview`).

Checkout used here: `~/git/ppview-matthies` (clone of `m-matthies/ppview`). The older
`~/git/ppview` is a different fork (`zoombya/ppview`) with local edits — don't mix them up.
Its README and CLAUDE.md are the authority on the viewer itself; read them before assuming a
feature exists.

## Which files to hand it

| pypatchy `patchy_format` | Drop these | Notes |
|---|---|---|
| `raspberry` (`RWriter`) | `init.top` + `last_conf.dat` or `trajectory.dat` | `.top` is self-contained (`iP`/`iR`/`iC` lines). Only the outer repulsion-site beads are drawn; each type uses just the `iR` sites listed in its `iC` line. |
| `flavio` / `josh_flavio` (`FWriter`) | `init.top` + conf/trajectory + `particles.txt` + `patches.txt` | companion files are found by name (`*particles*.txt`, `patches.txt`/`*.patch.txt`) or via `particle_file`/`patchy_file` in a dropped `input`. |
| `lorenzo` (`LWriter`) | `.top` + conf/trajectory | per ppview's docs; **not tested here**. |
| `subhajit` (`SWriter`) | — | not listed in ppview's supported formats; unverified. |

Optional extras that help:
- **`input`** (the oxDNA input file): ppview reads `PATCHY_radius` for the particle radius and `print_every` for lining up observables.
- **`observables.json` + the observable output** (e.g. `clusters.txt` from `PLClusterTopology`, `PatchyBonds`, `RaspberryPatchyBonds`): ppview recognises them by content, streams them (the 22 MB raspberry `clusters.txt` loaded fine), matches them to frames by step number, and draws bonds patch to patch. Loading one **opens the Clustering panel over the scene**.
- You don't need `energy.dat`; the readout takes time/energy from the conf header.

Conf vs trajectory: `last_conf.dat` gives one frame; `trajectory.dat` gives all frames
(`Frame n/N` in the readout). ppview's file-type priority is `traj > last > init > conf`.

### Getting the paths for an ensemble sim

```python
from pypatchy.patchy.simulation_ensemble import find_ensemble

with find_ensemble(name="triskellion_crystal_raspberry", date="2026-11-10") as e:
    sim = e.ensemble()[0]
    stage = e.sim_most_recent_stage(sim, True)
    top, traj = e.sim_get_stage_top_traj(sim, stage)        # init.top, trajectory.dat
    last = e.sim_get_stage_last_conf(sim, stage)             # last_conf.dat
    folder = e.folder_path(sim, stage)
    extras = [folder / "input", folder / "observables.json"]  # + particles.txt/patches.txt for flavio
```

(Run in the `pypatchy` conda env; this exact lookup was checked against a real ensemble.) Run
folders live under `simulation_run_dir()`, not `~/.pypatchy/`.

## Option 1: interactive

- Hosted: https://m-matthies.github.io/ppview — drag the files in (or Ctrl/Cmd+O).
- Local: `cd ~/git/ppview-matthies && npm install && npm start` (needs a Rust toolchain
  with the `wasm32-unknown-unknown` target, because `prestart` rebuilds the WebAssembly core).

Useful keys: `←/→` step a frame (Shift = 10), `Home`/`End` first/last, `Space` play/pause
(or frame the selection), `P` screenshot. Patches are always drawn as cones; the patch
legend, simulation box, axes and particle legend are toggles in the bottom bar.

## Option 2: render a PNG headlessly (so Claude can look at it)

`render.js` (next to this file) serves ppview's **committed `build/`** (no Rust, no dev
server), starts headless Chrome, drops the files into the app like a user would, optionally
steps to a frame, screenshots, and cleans everything up. Then `Read` the PNG.

One-time setup: `cd ~/git/ppview-matthies && npm install --ignore-scripts` (render.js uses
ppview's own `visual-tests/driver.js`, which needs `ws`). Needs Node and Chrome/Chromium
(`--chrome` or `$CHROME_PATH` if not at a standard location). Override the checkout with
`--ppview DIR` or `$PPVIEW_DIR`.

```bash
R=~/git/pypatchy/.claude/skills/ppview-visualization/render.js
D=<run folder>

# last conf
node $R --out /tmp/conf.png $D/init.top $D/last_conf.dat $D/input

# last trajectory frame, with the cluster observable, panel closed
node $R --out /tmp/last.png --frame last --close-panels \
        $D/init.top $D/trajectory.dat $D/input $D/observables.json $D/clusters.txt

# flavio: patch legend on, zoomed in
node $R --out /tmp/f.png --zoom -10 --click "Patch legend" \
        $D/init.top $D/last_conf.dat $D/input $D/particles.txt $D/patches.txt
```

Options: `--frame first|last|N` (N is 0-based; the readout's `Frame` is 1-based, so `--frame 3` reads `Frame 4/13`), `--size WxH`, `--settle MS`, `--zoom N` (wheel notches,
negative = in), `--click "<aria-label>"` (repeatable; e.g. `Simulation box`, `Patch legend`,
`Particle legend`, `Bonds`), `--close-panels`, `--hide-controls`. It prints the control-bar
readout (`Frame 13/13 …`) so you can confirm which frame you got, and any page errors.

Verified on this machine with a raspberry run (single conf, 13-frame trajectory, cluster
observable) and a flavio run (patches visible as cones, patch legend).

### Limits and gotchas

- **Small systems render tiny** at the default camera, and `--zoom` only goes so far. Raise
  `--size` rather than expecting a tight crop.
- **Software WebGL** (SwiftShader) in headless Chrome: fine for thousands of particles, slow
  for very large systems, and the picture is a screenshot, not something to measure from.
- The cluster **observable is only matched to frames when step numbers line up**; give it
  `observables.json` and `input` so `print_every` is known.
- `render.js` throws "scene never drew any geometry" if nothing renders — usually the wrong
  file set (e.g. missing `particles.txt`/`patches.txt` for flavio) rather than a viewer bug.
- The committed `build/` can lag `src/`; check `git log -1 -- build` vs `-- src` in the
  checkout, and run `npm run build` there if you need an unreleased feature.
- Don't kill leftover servers with `pkill -f <pattern>` from a Bash call without a
  `[x]` bracket trick — the pattern matches the shell running it.
- The oxDNA nucleotide `.top` (strand/base/n3/n5 lines) triggers ppview's DNA renderer, not
  the patchy one; that's for oxDNA DNA confs, not pypatchy output.
