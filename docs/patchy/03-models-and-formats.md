# Models and file formats

Each oxDNA patchy plugin reads its own file format, and supports its own observables. pypatchy
hides most of this behind **writers** (`pypatchy/patchy/pl/patchyio.py`): one class per format
that can both write a pypatchy scene out as simulation files and read simulation files back in.
Which writer an ensemble uses is set by `patchy_format` in its server config.

See [Background](01-background.md) for what each model actually computes.

## Which pieces go together

A working setup needs a matching **format, plugin, `interaction_type`, and observable**. Mixing
rows doesn't fail at setup time. It fails when oxDNA starts (can't parse the file), or worse,
runs with the wrong interaction.

| `patchy_format` | Writer | `plugin_search_path` | `interaction_type` | Bond observable | Required file-name params |
|---|---|---|---|---|---|
| `flavio` | `FWriter` | `<oxDNA>/contrib/romano` | `PatchyShapeInteraction` | `PLClusterTopology` | `patchy_file`, `particle_file` |
| `josh_flavio` | `JFWriter` | `<alloassembly>/contrib/romano` | `PatchyShapeInteraction` | `PLClusterTopology` | `patchy_file`, `particle_file` |
| `lorenzo` | `LWriter` | `<oxDNA>/contrib/rovigatti` | `DetailedPatchySwapInteraction` | `PatchyBonds` | `DPS_interaction_matrix_file` |
| `raspberry` | `RWriter` | `<oxDNA fork>/contrib/evans` | `RaspberryPatchyInteraction` | `RaspberryPatchyBonds` | — |
| `subhajit` | `SWriter` | — | — | — | not currently usable; see [below](#subhajit) |

All formats also use `topology` and `conf_file`; the default param set provides both (as
`init.top` and `init.conf`). `get_writer(fmt).reqd_args()` lists what a writer needs.

An observable only works if its plugin is loaded. `PatchyBonds` doesn't work with the Romano
model and `PLClusterTopology` doesn't work with Rovigatti's, despite their generic-sounding
names.

### Model parameters

Set these in the server config's `input_file_params` (or `const_params`). Defaults are the
plugin's own.

**Romano (`flavio`, `josh_flavio`)**

| Parameter | Default | Notes |
|---|---|---|
| `shape` | *required* | `sphere` (use this) or `hs` (hard sphere, MC only). In `default.json`. |
| `patchy_file`, `particle_file` | *required* | Usually `patches.txt`, `particles.txt`. |
| `use_torsion` | on | `1`, `0`, or `partial`. |
| `narrow_type` | 0 | 0–4; see [the table](01-background.md#romano-model-patchyshapeinteraction-contribromano). |
| `PATCHY_alpha` | 0.12 | Patch width. |
| `PATCHY_radius` | 0.5 | Particle radius. |
| `PATCHY_rcut` | 1.2 | Center-to-center cutoff for checking patch interactions. |
| `no_multipatch` | on | One bond per patch. |
| `same_type_bonding` | on | |

**Rovigatti (`lorenzo`)**

| Parameter | Default | Notes |
|---|---|---|
| `DPS_interaction_matrix_file` | *required* | pypatchy writes `interactions.txt`. |
| `DPS_sigma_ss` | 0.4 | Patch interaction range; lab configs use 0.082. |
| `DPS_lambda` | 1.0 | Three-body (swap) strength. Use 0 to disable on GPU. |
| `DPS_no_three_body` | 0 | Disables the three-body term (**CPU only**). |
| `DPS_normalise_patches` | 1 | Rescale patch positions to the surface. Lab configs set 0. |
| `max_density_multiplier` | | Cell-list sizing; raise it if oxDNA complains about density. |

**Raspberry (`raspberry`)**

| Parameter | Default | Notes |
|---|---|---|
| `PATCHY_alpha` | *effectively required* | Applied to every patch. **The effective pair width is 2 × this.** |
| `narrow_type` | *required with angmod* | Or give all of `PATCHY_theta_ts`, `_tc`, `_a`, `_b`. |
| `patchy_angmod` | on | Angular modulation of patch binding. |
| `PATCHY_bond_energy` | −0.1 | Threshold for considering patches bonded. |
| `dt` | *required* | |

## The formats

### `flavio`

Three files: a topology, a patches file, and a particles file.

The **topology** has the particle count and number of types, then one type ID per particle:

```
56 2
0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 ...
```

**`particles.txt`** lists each particle type's patches, by index into the patches file:

```
particle_0 = {
 type = 0
 patches = 0,1,2,3,4,5
}
particle_1 = {
 type = 1
 patches = 6
}
```

**`patches.txt`** has one block per patch. Blocks must be numbered consecutively from 0:

```
patch_0 = {
    id = 0
    color = 21
    strength = 1.0
    position = 0,0,-0.5
    a1 = 0,0,-1
    a2 = 0,1,0
}
```

The writer also adds `particle_types_N` and `patch_types_N` to the input file. Mainline oxDNA
ignores them; the alloassembly fork requires them.

> If a patch has no `a2` (for example, particle types read from a `lorenzo` set), the writer
> writes `a2 = 0,0,0`. oxDNA normalizes that to NaN, so with torsion on you'll get NaN
> energies. Either set `use_torsion = 0` or give every patch a real `a2`.

### `josh_flavio`

Same as `flavio`, plus an `allostery_conditional` line in each patch: a boolean expression over
the binding state of the particle's other patches (`true` for an ordinary patch). Only the
Romano plugin in [`sulcgroup/alloassembly`](https://github.com/sulcgroup/alloassembly) reads
this; mainline oxDNA ignores it. The particle types must be real polycube cube types (from a
rule with allostery), and the writer also needs `num_teeth`.

### `lorenzo`

The **topology** lists particle *types*, not particles. Each line has the instance count, the
number of patches, the patch type IDs, and a file of patch positions:

```
2 2
1 4 0,1,2,3 patches_0.dat
1 4 4,5,6,7 patches_1.dat
```

Each **`patches_<type>.dat`** has one `x y z` line per patch, in the same order as the IDs
above:

```
0 0.5 -0.5
0.5 0 -0.5
0 -0.5 -0.5
```

**`interactions.txt`** (the `DPS_interaction_matrix_file`) gives the energy for each pair of
**patch types**. Pairs not listed don't interact:

```
patchy_eps[0][4] = 0.25
patchy_eps[0][5] = 0.25
```

There's no color concept in this format. When reading one back, pypatchy reconstructs colors
from the matrix, numbering them ±1, ±2, … (not ±21). Patch orientations aren't stored, so
particle sets read from this format have no `a1`/`a2`.

> The writer never writes a self-pair (`patchy_eps[i][i]`), so a patch type can't bind another
> copy of itself in this format.

### `raspberry`

A single topology file with sections. Patches (`iP id strength color position a1`),
repulsion sites (`iR position radius`), particle types
(`iC type count patch_ids repulsion_site_ids`), multidentate groups (`iG`), and color
interactions (`iI color color strength`). From a real run:

```
2048 6
# Patch types
iP 0 0.25 20 0,0.5,-0.5 0,0,-1
iP 1 0.25 21 0.5000000000000001,0,-0.5 0,0,-1
...

# Repulsion points
iR 0.25,0.25,0.25 0.25
iR 0.25,0.25,-0.25 0.25
...

# Particle Types
iC 0 128 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 0,1,2,3,4,5,6,7
iC 1 384 24,25,26,27,28,29,30,31 0,1,2,3,4,5,6,7

# Patch Groups
iG 0 1 2 3
iG 4 5 6 7

# Interactions
iI -20 20 1.0
iI -21 21 1.0
```

The repulsion sites are a shared pool, and each type lists the ones it uses. Different types
can have different shapes. If you don't supply sites, the writer uses 8 sites of radius 0.25 at
the corners of a cube, (±0.25, ±0.25, ±0.25). The `iG` lines record which teeth came from the
same multidentate patch; oxDNA ignores them, but pypatchy uses them on read.

To give particles custom shapes, pass a topology with your own repulsion sites to the writer:
`get_writer("raspberry").write(scene, scene_top=RWriter.RPatchyTopology(..., repulsion_sites=...))`.
`pypatchy/patchy/hash/slat_to_raspberry.py` is a worked example that builds particle sets and
repulsion sites from crisscross slat designs.

> Reading a raspberry scene back (`read_scene`) defaults to the **first** configuration in the
> file, while the `flavio` and `lorenzo` readers default to the **last**. Pass `conf_idx`
> explicitly if it matters.

### `subhajit`

The `subhajit` writer (`SWriter`) produces a single-file `iP`/`iC` topology meant for Subhajit
Roy's patchy helix bundle interactions (`PHB`, in a separate oxDNA fork). It doesn't currently
work end-to-end:

- the column it writes as `a3`, PHB reads as `a2`;
- it puts every particle on strand 0, which disables patch binding between them under PHB;
- reading these files back in pypatchy doesn't work (`read_scene` adds no particles).

The repository's `subhajit.json` server config points at the alloassembly Romano plugin, which
can't read this format either. Don't use it without fixing these first.

## Reading and writing files directly

You don't need writers for normal ensemble work, since the ensemble calls them for you. They're
useful for converting between formats, or for loading simulation output that pypatchy didn't
set up.

```python
from pypatchy.patchy.pl.patchyio import get_writer

w = get_writer("lorenzo")             # a new writer instance each call
w.set_directory("/path/to/sim")
ptypes = w.read_particle_types(topology="init.top",
                               DPS_interaction_matrix_file="interactions.txt")
scene = w.read_scene("init.top", "last_conf.dat", ptypes)   # a PLPSimulation
```

`get_writer()` with no argument returns the writer for the active server config's format.
Because each call returns a new instance, keep the object you called `set_directory` on.
Calling `get_writer(...)` again gives you a writer with no directory set.

`read_particle_types` arguments differ by format:

| Format | `read_particle_types(...)` |
|---|---|
| `flavio` | `patchy_file=, particle_file=` |
| `lorenzo` | `topology=, DPS_interaction_matrix_file=` |
| `raspberry` | `topology=` |

`read_scene(top_file, traj_file, particle_types, conf_idx=None)` reads one configuration from a
conf or trajectory file.

To write:

```python
out = get_writer("flavio")
out.set_directory("/path/to/output")
out.directory().mkdir(parents=True, exist_ok=True)
out.write(scene, topology="init.top", conf_file="init.dat",
          particle_file="particles.txt", patchy_file="patches.txt")
```

`write()`'s keyword arguments are the format's file names (its `reqd_args()`).

### Converting between formats

Read with one writer and write with another:

```python
src = get_writer("lorenzo"); src.set_directory(lorenzo_dir)
ptypes = src.read_particle_types(topology="init.top", DPS_interaction_matrix_file="interactions.txt")
scene = src.read_scene("init.top", "last_conf.dat", ptypes)

dst = get_writer("flavio"); dst.set_directory(flavio_dir)
dst.write(scene, topology="init.top", conf_file="init.dat",
          particle_file="particles.txt", patchy_file="patches.txt")
```

Things that don't survive a conversion: `lorenzo` → anything loses patch orientations (`a1`,
`a2`); anything → `raspberry` loses torsion; anything → `lorenzo` loses self-binding patches.

The helper functions in `pypatchy/patchy/patchy_scripts.py` (`lorenzian_to_flavian`,
`convert_multidentate`, `convert_udt_files_to_mdt`, `convert_flavian_to_lorenzian`) predate the
current writer API and are broken; use the pattern above instead.

### Other exports

- `scene.export_to_mgl(filename, patches_w)` writes an MGL file (for older viewers);
  `pypatchy.patchy.mgl.load_mgl(path)` reads one.
- `pypatchy.patchy.pl.patchy_exports.to_gltf(scene, export_to, particle_shape="sphere"|"cube")`
  writes glTF for rendering in Blender or similar.
- `RWriter.write_as_mgl(...)` writes an MGL that shows raspberry repulsion sites as spheres.

## Observable output formats

You won't usually parse these by hand (the analysis pipeline does it), but it helps to know
what's in them when something looks wrong.

**`PLClusterTopology`** (Romano). One line per print interval: the number of clusters, then
for each cluster its members in parentheses (type IDs when `show_types = 1`) and an adjacency
list by particle index in brackets:

```
1 ( 0 1 )  [5 -> (18)]
```

The observable considers a pair bonded when the patch energy term alone is negative.
`show_types` must be set. Spacing between clusters is irregular.

**`PatchyBonds`** (Rovigatti). A block per print: a `# step S N N` header, then two lines per
particle: bond counts per patch, and the 1-based indices of bonded particles.

**`RaspberryPatchyBonds`** (raspberry). One line per print listing each bond as particle and
patch index pairs. It takes no options (a `show_types` setting is ignored).

## Adding a new format

Subclass `PLBaseWriter` in `patchyio.py` and implement its abstract methods (`reqd_args`,
`get_input_file_data`, `write`, `write_top`, `read_top`, `read_particle_types`, `read_scene`,
`get_scene_top`). Then either register it at the top of your script:

```python
from pypatchy.patchy.pl.patchyio import register_writer
register_writer("my_format", MyWriter)      # pass the class, not an instance
```

or add it to the built-in registry in `patchyio.py` and open a pull request. Then set
`"patchy_format": "my_format"` in a server config.
