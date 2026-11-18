# Converting patchy particles to DNA origami

The patchy → origami converter (`pypatchy/patchy/patchy_origami_convert.py`) turns a
patchy-particle configuration into a full oxDNA DNA-origami structure. Each particle becomes a
copy of a DNA-origami monomer, positioned and oriented as the particle was, and each patch
becomes a set of staple strands extended with sticky ends whose sequences encode the patch
colors. It also exports the staple sequences as an IDT-style plate spreadsheet.

You need three things:

1. a **multidentate patchy scene** (a `PLPSimulation`) of the structure to build;
2. a **DNA-origami monomer** as oxDNA `.top`/`.dat` files;
3. the **IDs of the staples** on that monomer whose 3' ends can carry sticky ends, grouped by
   which patch they make up.

The converter works on the **teeth** of a multidentate scene. Each tooth becomes one sticky
end, so `n_teeth` should equal the number of staples in each group (4 is typical).

## 1. The scene

### From a polycube

```python
from pypatchy.polycubeutil.polycube_structure import load_polycube
from pypatchy.patchy.pl.plpatchylib import polycube_to_pl
from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings

polycube = load_polycube("my_polycube.json")      # relative to ~/.pypatchy/input/
scene = polycube_to_pl(polycube, MultidentateConvertSettings(n_teeth=4, dental_radius=0.5))
```

`dental_radius` should reflect the monomer geometry. For a cube monomer whose sticky-end
staples sit near the edges of each face, use about half the face width (in particle units, so
0.5 for a unit particle).

### From simulation output

Load a configuration as a scene (see [Analysis](06-analysis.md#scenes-working-with-individual-configurations)),
for example `e.get_scene(sim)` or a cluster from `split_scene_by_clusters`. The particle set
must be multidentate, and must have come from converting a unidentate set
(`ptypes.to_multidentate(...)`). The converter uses the record of which teeth came from which
original patch. Sets read directly from multidentate files don't carry this record, so read the
unidentate files and convert.

For `lorenzo`-format files, which have no patch orientations, use
`MultidentateConvertSettings(..., torsion=False)`.

### Attach an energy model

The converter decides which particles are bonded using the oxDNA interaction, so the scene needs
an energy model. `e.get_scene` attaches one; for a scene built any other way, attach it
yourself:

```python
from pypatchy.patchy.pl.oxpy_energy import OxpyEnergyModel

scene.set_energy_model(OxpyEnergyModel("flavio", {
    "interaction_type": "PatchyShapeInteraction",
    "plugin_search_path": "/path/to/oxDNA/contrib/romano",
    "T": 0.1, "PATCHY_alpha": 0.12, "PATCHY_radius": 0.5, "shape": "sphere",
    "use_torsion": 1, "same_type_bonding": 1, "no_multipatch": 1,
}))
```

Without one, `convert()` raises `ValueError`.

## 2. The monomer

```python
from oxpy_utils.structure_editor.dna_structure import load_dna_structure
from pypatchy.patchy.dna_particle import DNAParticle

dna = load_dna_structure("/path/to/cube_monomer.top", "/path/to/cube_monomer.dat")
monomer = DNAParticle(dna, [
    [128, 131, 130, 136],      # strands that together form one patch
    [151, 155, 152, 160],
    [168, 163, 144, 156],
    [146, 147, 149, 150],
    [138, 140, 139, 141],
    [142, 167, 143, 153],
])
```

- `load_dna_structure` takes paths as given (no `~/.pypatchy/input` prefix).
- Strand IDs are **0-based indices** into the structure's strand list (the 1-based strand ID in
  the `.top` file, minus one).
- Only the grouping matters. Neither the order of the groups nor the order within a group needs
  to match the particle's patches: the converter finds the rotation that best aligns patches
  with strand groups, then pairs teeth with strands.
- A working cube monomer and its strand groups are in
  `tests/test_files/dna_confs/structure_cube/` and `tests/test_oxview_recolor.py`.

> Older lab documentation imported `load_dna_structure` from `ipy_oxdna.dna_structure`. That
> library has been folded into `oxpy_utils.structure_editor`.

## 3. Set up the converter

```python
from pypatchy.patchy.patchy_origami_convert import PatchyOrigamiConverter

conv = PatchyOrigamiConverter(
    scene,
    spacer_length=12,          # poly-T spacer between the staple and the sticky sequence
    sticky_length=10,          # if set, colors without an assigned sequence get random ones
    padding=1.1,               # scales up the spacing between monomers
    expected_num_edges=4 * polycube.num_connections(),   # sanity check: bonds per tooth
)
```

| Argument | Default | Meaning |
|---|---|---|
| `spacer_length` | 16 | Length of the poly-T spacer. Use 0 for none. |
| `sticky_length` | `None` | Sticky-end length. If `None`, you must assign every color's sequence yourself. |
| `padding` | 1.0 | Multiplies the automatically computed particle-to-origami scale factor. |
| `expected_num_edges` | −1 | If set, `convert()` checks that this many tooth–tooth bonds were made. Count per tooth: `n_teeth × number of particle–particle bonds`. |
| `rel_rms_tolerance` | 0.15 | Maximum allowed misfit when aligning a particle's patches to a monomer's strand groups (relative to the patch circumference). Raise it if alignment fails with an assertion. The tests use 1. |

`particle_delta`, `bond_length`, `cos_theta_max`, and `flexable_patch_distances` are accepted but
currently do nothing.

### Sequences

Assign a sticky-end sequence to each positive color. Its binding partner (the negative color)
gets the reverse complement automatically:

```python
from oxpy_utils.structure_editor.dna_structure import rand_seq

for color in conv.scene.particle_types().patch_colors():
    if color > 0:
        conv.assign_color_sequence(color, "ATGCATCGAT", update_rc=False)
```

- After multidentate conversion with torsion, **each tooth has its own color**, so each original
  patch color needs `n_teeth` sequences. Loop over `patch_colors()` as above rather than
  assuming the colors start at 21.
- Pass `update_rc=False`. With the default `True`, your sequence ends up on the partner color
  and its reverse complement on yours. The bonds still pair, but the sequences are swapped
  relative to what you asked for.
- Randomly generated sequences (from `sticky_length`, or `rand_seq`) are uniformly random, with
  **no checks for orthogonality, secondary structure, or melting temperature**. For anything you
  intend to order, design the sequences properly (e.g. with oxpy-utils' sequence design tools)
  and assign them.

### Monomers

```python
conv.assign_particles(monomer)                 # use this monomer for every particle type
conv.assign_particles(other_monomer, "CT2")    # or per type (by name, ID, or particle type)
```

## 4. Convert

```python
conv.position_particles()      # place a rotated copy of the monomer at each particle
conv.convert()                 # add sticky ends and join bonded staples
```

Call `position_particles()` explicitly first; `convert()` fails without it.

## 5. Export

```python
conv.save_top_dat("my_structure/dna.top", "my_structure/dna.dat")
conv.save_oxview("my_structure/dna.oxview", color_by_type=True)
conv.dump_monomers("my_structure")     # one .top/.dat per particle type, with sticky ends
conv.export_stickys_staples("/path/to/staples.xlsx")
```

Paths behave differently from function to function:

| Function | Relative paths resolve against | Notes |
|---|---|---|
| `save_top_dat` | `~/.pypatchy/output/` | The parent directory must already exist. |
| `save_oxview` | `~/.pypatchy/output/` | `color_by_type=True` colors monomers by particle type. A dict of type ID → `"#rrggbb"` sets your own colors. |
| `dump_monomers` | `~/.pypatchy/output/` | Creates the directory. |
| `export_stickys_staples` | the current directory | The parent directory must exist. |

`export_stickys_staples(path, by_row=True, incl_no_sticky=True, incl_absent=True, incl_original_patch_nums=False)`
writes an Excel workbook formatted for IDT plate orders: one sheet per 96-well plate, with
columns `Well`, `Name`, `Sequence` (5'→3'). Only the candidate (sticky-end-bearing) staples are
exported, not the full staple set.

- `by_row=True` starts each particle type on a new plate row. It requires every monomer to have
  exactly 24 candidate staples.
- `incl_no_sticky` includes candidate staples that didn't get a sticky end.
- `incl_absent` includes particle types that don't appear in the scene.
- `incl_original_patch_nums` appends the polycube face name (polycube scenes only).

## How it works

For each particle type, the converter finds the best rotation of the DNA monomer onto the
particle. It superimposes patch centroids and directions on strand-group centroids and
directions, trying every assignment of strand groups to patches and keeping the one with the
lowest RMS. Within each patch, it then pairs teeth with strands by minimum total distance. The
scale factor between particle units and nm is chosen so that bonded monomers sit far enough
apart for their sticky ends (spacer + sticky sequence) to reach each other, times `padding`.
`convert()` then extends each bonded tooth pair's staples with complementary sticky ends, and
gives unbonded patches single-stranded sticky ends pointing outward.

## Caveats

This code has had little use since the oxpy energy-model rewrite, and has some open questions:

- The 3'/5' end handling is inconsistent in places: some code paths take the strand's 5' base
  where they mean the 3' end. Check the generated structure in oxView before trusting it.
- Scenes from MGL files probably don't convert (they lack the multidentate source record).
- Scenes from `lorenzo` files aren't covered by tests.
