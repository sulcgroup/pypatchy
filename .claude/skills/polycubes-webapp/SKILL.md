---
name: polycubes-webapp
description: Use when asked to look at, screenshot, or sanity-check a polycube structure, rule, or SAT solve spec in the polycubes web app (sulcgroup/polycubes-clone, `web/`) - the browser viewer/rule editor, the multi-structure gallery, or the SAT solver page. Covers which pypatchy outputs each page accepts (structure JSON from PolycubeStructure.to_json/save_polycube and PolysatSolution.exportScene, rule strings via ?rule=, solve-spec JSON), the hosted and local ways to open them, render.js in this skill, which builds the app and renders a PNG headlessly (also the flattened rule diagram as SVG and the environment panel's per-cube data), and how to read the same cube/patch information directly from a PolycubeStructure in pypatchy (describe_structure.py). Not for patchy-particle (oxDNA/PL) confs (see ppview-visualization), for the rule string syntax itself (see polycubes-rules), or for libtlm/VMMC simulation (see tlm-simulations) - the web app runs its own separate stochastic assembler.
---

# The polycubes web app

`~/git/polycubes-clone/web/` (`sulcgroup/polycubes-clone`, TypeScript + Three.js, built with
Parcel) is the browser front end for polycubes. Three separate pages:

| Page | Hosted | What it is |
|---|---|---|
| `index.html` | https://sulcgroup.github.io/polycubes-clone/ | viewer + rule editor. Runs its **own in-browser stochastic assembler** on the rule |
| `gallery.html` | …/polycubes-clone/gallery.html | any number of structure JSONs rendered side by side |
| `solve.html` | …/polycubes-clone/solve.html | draw/load a target topology, set options, run the SAT solver (in a Web Worker) |

The older upstream app (`~/git/polycubes`, `Akodiat/polycubes`, with `view.html`/`edit.html`/
`klossar.html`) is a different codebase. Everything here is about the `polycubes-clone` one.

**What it is not:** the viewer's assembler is not oxDNA and not libtlm's VMMC (see
[[tlm-simulations]]). It is a quick way to see whether a rule's binding logic assembles what
you expect, not a physical result. A finite rule just stops growing when it runs out of moves.
The duck's own rule string (`1:3|3:3||3|3|3_|||3:1|-2:2|_-1||||2:1|_-3:1|||||`) assembles all 8
of its cubes there, with the same cube-type counts as the structure.

## What each page takes from pypatchy

| Input | Made by | Goes into |
|---|---|---|
| **Structure JSON** `{"cube_types": [...], "cubes": [...]}` | `PolycubeStructure.to_json()`; `save_polycube(fp)` (a relative path lands under `~/.pypatchy/input/scenes/`); `PolysatSolution.exportScene(name)` (a bare name lands in `~/.pypatchy/output/SAT/`) | gallery (any number); viewer's *upload structure* button (one file; picking several hands them to the gallery). Read back with `load_polycube`. |
| **Rule string** | `str(structure.rule)` (a `PolycubesRule`, format in [[polycubes-rules]]) | viewer URL `?rule=<string>` (also `?hexRule=`, `?decRule=`; `?nMaxCubes=`, `?maxCoord=`, `?randseed=`) |
| **Rule JSON** | `rule.toJSON()` | viewer's *upload rules* button (adds cube types to the current rule) |
| **Solve spec JSON** `{"bindings": [...], "torsion": true, "nO": 4, ...}` | the solve page's save button; `~/.pypatchy/input/topologies/*.json` are the same files and are what `multisolve`/`solve_exec` read; `CommonComponent.to_solve_spec()` writes a minimal one | solve page: drop it on the page or use the upload button. Invalid specs are refused with a message and the current design is kept |

The viewer's *download structure* button writes the structure JSON; `save_polycube`'s docstring
says its output "can be read … into the web app", and this was checked with
`tests/test_files/cube_structures/*.json`.

The gallery assumes torsion was on (the export doesn't record it).

## Interactive

- Hosted (above), or locally: `cd ~/git/polycubes-clone/web && npm install && npm start`
  (also `npm run start-solve`, `npm run start-gallery`).
- `npm run build*` scripts write into the **repo root** with a `/polycubes-clone/` URL prefix
  (the GitHub Pages `deploy` layout). Don't run them casually; `render.js` below builds to a
  cache instead.

## Render a PNG headlessly (so Claude can look at it)

`render.js` (next to this file) builds the three pages with Parcel into
`~/.cache/pypatchy-polycubes-web/<key>/` the first time (about 15 s; the key follows the
checkout's HEAD plus any uncommitted `web/src` changes, so edits trigger a rebuild), serves them
on a random port, drives headless Chrome over the DevTools protocol, and cleans up. It leaves
the repo's build directories and `package.json` alone. Then `Read` the PNG.

One-time setup: `cd ~/git/polycubes-clone/web && npm install && npm install --no-save ws`
(`node_modules` is gitignored and `--no-save` leaves `package.json`/the lockfile alone). Needs
Node and Chrome/Chromium (`--chrome`/`$CHROME_PATH`); `--repo`/`$POLYCUBES_DIR` overrides the
checkout.

```bash
R=~/git/pypatchy/.claude/skills/polycubes-webapp/render.js

# structures side by side (labelled with file names)
node $R --out /tmp/g.png a.json b.json c.json

# a rule string, run in the browser's assembler until step 60 (or until it stops)
node $R --page viewer --rule "$(python -c 'print(str(rule))')" --steps 60 --seed 1 --out /tmp/v.png

# one structure in the viewer (shows the rule panel next to it)
node $R --page viewer --structure a.json --out /tmp/v.png

# a solve spec on the solver page (loads it; does not run the solver)
node $R --page solve --spec ~/.pypatchy/input/topologies/3x3x3.json --out /tmp/s.png
```

Other options: `--size WxH`, `--settle MS`, `--max-cubes N`, `--rebuild`. It prints the gallery
cell count (and any cells that failed to load), the viewer's `Step:` readout, or the solve page's
voxel count, plus any page errors.

### The flattened rule diagram (`--show-rule`, `--rule-svg`)

The viewer's `img#toggle-rule` button opens a panel with a flat picture of the cube types:
one cross-net per type (six 64 px faces filled with the type's colour, a glyph on each face
that carries a patch, labelled `CT<n>`). The glyph's colour is the patch colour, its outline
is black for a positive colour and white for a negative one, and with the Torsion checkbox on
(the default) its rotation shows the patch orientation. The app draws it into
`<svg id="rule-display-canvas">`, only while the panel is open.

`render.js` takes that SVG out of the page rather than screenshotting the panel (which is a
fixed, scrolling overlay and clips large rules):

```bash
# PNG of just the diagram (Read it), the SVG file, and 5 cube types per row at 1.5x
node $R --page viewer --structure a.json --show-rule --rule-svg /tmp/rule.svg \
        --rule-columns 5 --rule-scale 1.5 --out /tmp/rule.png
node $R --page viewer --rule "$RULE" --rule-svg /tmp/rule.svg --out /tmp/3d.png   # SVG + the 3D scene
```

- The SVG is standalone (`xmlns`, a tight `viewBox`, one `<g transform="translate(x,y)">` per
  cube type) so cube types can be rearranged, dropped or restyled afterwards by editing those
  groups. `--rule-columns N` does the common re-flow for you (cells sized to the largest type;
  the app's own layout wraps at about 800 px). `--rule-scale S` scales width/height only.
- Any rule source works: `--rule`, `--structure`, or none (the default two-type rule).
- The app's own source also draws arrows for activation effects (allostery); that path was
  not exercised here.

Verified here with `tests/test_files/cube_structures/{duck,human,4x4tile}_polycube.json` in
the gallery, a finite 7-cube rule and the duck's rule string in the viewer (its rule diagram is
identical to the uploaded structure's), the duck uploaded to the viewer, `3x3x3_angled_layers.json` on the solve page (27 voxels, plates and rods
drawn), and the rule diagram (PNG and SVG, re-flowed to 5 columns) for the 10-type `human.json`
and the 4-type duck.

### The environment panel (`--show-env`, `--env-json`)

The viewer's `#envToggle` button shows a panel on the right (`#env-browser`) with one
`<table class="env-vis-box">` per cube instance: **Cube Type**, **Cube Name**, **Position**, and a
row for every *coloured* patch with the face it is on (`left/right/bottom/top/back/front`), its
orientation (only when the type has torsion), its colour, and the state/activation variables
(only when the type has allostery). The tables are in the DOM even while the panel is hidden.

```bash
node $R --page viewer --structure a.json --env-json /tmp/env.json --out /tmp/v.png   # data only
node $R --page viewer --structure a.json --show-env --out /tmp/v.png                 # panel in the shot
```

`--env-json` writes `[{name, type, cubeName, position, patches: [{face, ori, color, stateVar,
actVar}]}]`. Caveats, all checked against the duck:

- **Cube names are random** (`ewgd`): pypatchy writes `"name": "Cube_<uid>"`, but the web app
  reads `personal_name` and invents a name when it is missing. Match cubes to pypatchy by
  **position**, never by name.
- **Orientation is unsigned**: the panel shows the angle between the face's default orientation and
  the patch's, so 3 quarter turns shows as 1. Compare with `min(n, 4 - n)`.
- Row order within a cube differs from pypatchy's (`front` before `back`); compare as sets.

### Gotchas

- **Cubes are opaque, so one camera angle can hide cubes, and similar colours make it worse**
  (the duck's 8th cube sits behind a same-coloured one at the viewer's default angle). A
  render can't confirm a cube count. Count from the JSON (`len(list(structure.particles()))`).
- **Don't trust the viewer's `Step:` readout after an upload.** For a rule-driven run it is the
  simulation's step count (the seed counts as 1, so a growing assembly reads as its cube count).
  After *upload structure* the app never refreshes it, so it still shows the default rule's
  value: the duck (8 cubes) and the 4x4 tile (16 cubes) both read `Step: 7`.
- `--steps` stops at the target or after the counter has been still for 3 s ("the assembly has
  likely finished", printed as a note). For infinite-growth rules cap it with `--max-cubes`.
- The gallery's camera is a fixed isometric view per cell; there's no per-cell zoom in the
  render. Use the viewer for a closer look.
- Scope hoisting breaks the solve page, so it must be built with `--no-scope-hoist` (and
  each page is built separately). `render.js` does this; do it too if you build by hand.
- First run after a `git pull`/edit in `web/src` rebuilds (about 15 s); `--rebuild` forces it.
- Software WebGL (SwiftShader): fine for these scenes, slow for thousands of cubes.

## Reading the same information in pypatchy

You usually don't need the browser for this: `pypatchy.polycubeutil.polycube_structure` has all of
it, and it also counts cubes reliably.

```python
from pypatchy.polycubeutil.polycube_structure import load_polycube

structure = load_polycube("/path/to/structure.json")   # {"cube_types", "cubes"}; a bare relative path
                                                       # is looked up under ~/.pypatchy/input
structure.rule                  # PolycubesRule; str(structure.rule) is the rule string
structure.cube_type_counts()    # cubes per type id
for cube in structure.particles():                   # PolycubesStructureCube
    cube.get_uid(), cube.position()                  # integer grid position
    cube.get_cube_type().type_id()                   # -> "CT<id>" in the web panel
    cube.rotation()                                  # scipy Rotation (cube.rot_mat() for the matrix)
    for patch in cube.patches():                     # already rotated into world orientation
        patch.colornum()                             # 0 = no patch
        patch.dirIdx()                               # index into left,right,bottom,top,back,front
        patch.get_align_rot_num()                    # orientation, 0-3 quarter turns
structure.cubeAtPosition(np.array([x, y, z]))        # lookup by grid position
```

Other useful members: `structure.to_json()`, `structure.graph` (a networkx MultiDiGraph with each bond as two
directed edges, so the duck has 14 edges; `num_connections()` counts bonds: 7), `structure.centroid()`, `structure.transform(rot, tran)`, and
`structure.substructure(nodes)`. `describe_structure.py` (next to this file) prints exactly the
web panel's table from a structure file:

```bash
python describe_structure.py a.json            # per cube: type, position, each patch's face/ori/colour
python describe_structure.py a.json --json     # same data, same field names as --env-json
python describe_structure.py a.json --types    # rule string and cubes per type
```

(Run in the `pypatchy` conda env.) On the duck, `describe_structure.py --json` and
`render.js --env-json` agree on all 8 cubes: positions, cube types, and each patch's face, colour
and orientation (after the unsigned-orientation fold above).

**Which to use:** reach for `describe_structure.py` / the Python API for counts, types, bindings and
anything you will compute with. Use the browser extraction when the question is what the *web app*
shows or does (for example, what it makes of a file, or the cubes a rule assembles there with
`--rule ... --env-json`).
