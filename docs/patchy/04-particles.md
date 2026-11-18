# Particle sets

An ensemble needs to know what particles it's simulating: how many types there are, where each
type's patches are, and which patches bind which. pypatchy's internal representation of this is
a **`PLParticleSet`** (`pypatchy/patchy/pl/plparticleset.py`), a list of `PLPatchyParticle`
types, each with `PLPatch`es, plus an **interaction matrix** saying how strongly each pair of
patch colors binds. ("PL" is a legacy name and no longer stands for anything.)

You can give an ensemble its particles in three ways:

1. a **polycubes rule string** (`"rule"`), the usual case;
2. polycube **cube types as JSON** (`"cube_types"`);
3. a **`PLParticleSet`** built in Python, or loaded from existing particle files
   (`"particle_types"`).

## Polycubes rules

A polycubes rule describes cube-shaped particle types with up to one patch per face. It's the
format the design tools output, and what most lab ensembles use.

```
<cube type 0>_<cube type 1>_..._<cube type N>
```

Cube types are separated by `_`. Each cube type is **six** `|`-separated faces, always in the
order:

| index | face | direction |
|---|---|---|
| 0 | left | −x |
| 1 | right | +x |
| 2 | bottom | −y |
| 3 | top | +y |
| 4 | back | −z |
| 5 | front | +z |

Each face is empty (no patch) or `color[:orientation[:conditional]]`:

- **color** — a nonzero integer. **Color `c` binds `−c`** and nothing else; `1` doesn't bind
  `1`, and `1` doesn't bind `−2`.
- **orientation** — 0 to 3, the patch's rotation about the face normal in 90° steps. In a
  torsional model, two patches only bind fully when their orientations line up once the cubes
  are face to face. Omit it if you don't care.
- **conditional** — an allosteric condition on the other faces' binding state. See
  [Allostery](#allostery).

Examples:

- `1|1|1|1|1|1_-1|||||` — the "3D cross". `CT0` has a `+1` patch on every face, and `CT1` has a
  single `−1` patch on its left face. One `CT0` binds six `CT1`s; `CT0`s can't bind each other.
- `|1||-4||3:1_||||-3:3|-3:2_5||4:1|-1:2||_...` — faces with orientations, and empty faces
  wherever a cube has no patch.

Types generated from a rule are named `CT0`, `CT1`, … in order, and those are the names you use
for [particle counts](05-ensembles.md#particle-counts).

Some details:

- An older dialect separates faces with `#` instead of `|`. pypatchy still reads it, so specs
  like `1:0:##1:0:##1:0:#_-1:0:#-1:1:####` work.
- **Hexadecimal rules** (`040404040404840000000000`), as used by the polycubes web app, are not
  accepted. Convert them in the web app first.
- To check a rule visually, `PolycubesRule(rule_str=...).draw_rule()` returns an SVG of each
  cube type as an unfolded net.

The [polycubes web app](https://akodiat.github.io/polycubes/) is the quickest way to see what
a rule assembles into (in a lattice model, not oxDNA).

### Allostery

A face's third field, or an `@` suffix on a cube type (`@src1,src2>target;...`), makes patches
switch on or off depending on which other patches are bound. This only does anything in the
`josh_flavio` format, simulated with the alloassembly fork of oxDNA. The parsing code is
fragile. Don't combine the inline `color:orientation:conditional` form with the 4-field
`color:orientation:stateVar:activationVar` form in one rule.

### Cube types as JSON

`"cube_types"` accepts a rule in the JSON form the polycubes tools export: a list of cube
types, each with a `patches` list giving `color`, `dir` and `alignDir` vectors, and optionally
allostery state:

```json
"cube_types": [
    {
        "typeName": "CT0",
        "patches": [
            {"color": 1, "dir": {"x": -1, "y": 0, "z": 0}, "alignDir": {"x": 0, "y": 0, "z": 1}},
            {"color": 4, "dir": {"x": 0, "y": -1, "z": 0}, "alignDir": {"x": 0, "y": 0, "z": 1}}
        ]
    }
]
```

## Colors

Rule colors are **shifted by 20** when converted to a particle set: polycube color `1` becomes
`21`, `−1` becomes `−21`, `2` becomes `22`, and so on. This keeps them clear of the Romano
plugin's convention that colors with magnitude below 10 bind to *themselves*. So when you look
at generated `patches.txt` or set interaction strengths in Python, rule color `n` is `20 + n`.

After [multidentate conversion](#multidentate-conversion) with torsion, each tooth gets its own
color, so the mapping is no longer one-to-one with rule colors (see below).

## Interaction matrix and strengths

By default every patch has strength 1, and each color `c` binds `−c` with energy scale
`sqrt(strength_a · strength_b)`. To change specific bond strengths, build the particle set
yourself and edit its interaction matrix before passing it to the ensemble:

```python
from pypatchy.polycubeutil.polycubesRule import PolycubesRule
from pypatchy.patchy.pl.plpatchylib import polycube_rule_to_PL
from pypatchy.patchy.param_val_types import PARTICLE_TYPES_KEY

ptypes = polycube_rule_to_PL(PolycubesRule(rule_str="|1||-4||3:1_||||-3:3|-3:2_..."))
ptypes.interaction_matrix()[(-21, 21)] = 2.0      # rule color 1 binds twice as strongly
ptypes.interaction_matrix()[(-22, 22)] = 2.66

e = build_ensemble(cfg={..., PARTICLE_TYPES_KEY: ptypes, ...}, mdt={...})
```

(From [`examples/setup_sims_programmatic.py`](../../examples/setup_sims_programmatic.py).)
Keys are pairs of colors **after** the +20 shift.

For binding rules that `c ↔ −c` can't express (many-to-many colors, self-binding colors,
angle-dependent strengths), `DetailedPolycubesRule` in `polycubesRule.py` takes an explicit
interaction matrix. Note that multidentate conversion requires a one-to-one matrix.

## Particle sets in Python

To build particle types that aren't cubes:

```python
import numpy as np
from pypatchy.patchy.pl.plparticleset import PLParticleSet
from pypatchy.patchy.pl.plparticle import PLPatchyParticle
from pypatchy.patchy.pl.plpatch import PLPatch

ptypes = PLParticleSet([
    PLPatchyParticle(
        type_id=0, particle_name="A", radius=0.5,
        patches=[
            PLPatch(type_id=0, color=21,
                    relposition=np.array([0.5, 0, 0]),   # position relative to the center
                    a1=np.array([1, 0, 0]),              # direction the patch points
                    a2=np.array([0, 1, 0])),             # orientation; needed for torsion
                                                         # and for multidentate conversion
            PLPatch(type_id=1, color=22, relposition=np.array([-0.5, 0, 0]),
                    a1=np.array([-1, 0, 0]), a2=np.array([0, 1, 0])),
        ]),
    PLPatchyParticle(
        type_id=1, particle_name="B",
        patches=[
            PLPatch(type_id=2, color=-21, relposition=np.array([0.5, 0, 0]),
                    a1=np.array([1, 0, 0]), a2=np.array([0, 1, 0])),
        ]),
])
```

- Patch `type_id`s must be unique across the whole set. Number them consecutively from 0;
  several writers assume patch IDs are list positions.
- If no interaction matrix is given, one is made from the `c ↔ −c` rule.
- `a2` must be perpendicular to `a1`.

Pass it with `PARTICLE_TYPES_KEY`. Particle counts then use the particle names (`A`, `B`), or
`particletype_<id>` for unnamed types.

### Loading existing particle files

In a JSON spec, `particle_types` can point at existing particle files in a known format,
relative to `~/.pypatchy/input/`:

```json
"particle_types": {
    "format": "flavio",
    "patchy_file": "my_design/patches.txt",
    "particle_file": "my_design/particles.txt"
}
```

The other keys are passed to that writer's `read_particle_types`; see
[Models and file formats](03-models-and-formats.md#reading-and-writing-files-directly).

## Multidentate conversion

[Multidentate patches](01-background.md#multidentate-patches) replace each patch with a ring of
`n_teeth` smaller patches. In an ensemble, add an `mdt_convert` parameter. As a constant:

```json
"const_params": {
    "mdt_convert": {
        "type": "mdt_convert",
        "n_teeth": 4,
        "dental_radius": 0.2,
        "torsion": true
    }
}
```

or as an ensemble parameter, to compare settings (each value needs a `name`, used for its
folder):

```json
["mdt_convert", [
    {"type": "mdt_convert", "name": "3teeth", "n_teeth": 3, "dental_radius": 0.5},
    {"type": "mdt_convert", "name": "2teeth", "n_teeth": 2, "dental_radius": 0.5}
]]
```

In Python, convert a particle set (or a whole scene) directly:

```python
from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings

mdt_ptypes = ptypes.to_multidentate(MultidentateConvertSettings(n_teeth=4, dental_radius=0.2))
```

> Don't set a `num_teeth` parameter directly; older specs did, and that now raises an error.
> Use `mdt_convert`.

### Settings

`MultidentateConvertSettings` fields:

| Field | Default | Meaning |
|---|---|---|
| `n_teeth` | *required* | Teeth per original patch. |
| `dental_radius` | *required* | Distance from the original patch position to each tooth. Bigger rings lock rotation more firmly, but teeth can end up off the particle surface or overlapping neighboring patches. |
| `torsion` | `True` | Whether source patches have a meaningful `a2`. If `True` and a patch has no valid `a2`, conversion raises. If `False`, an arbitrary `a2` is made up and a warning is printed. In that case teeth on binding partners won't necessarily line up. |
| `follow_surf` | `False` | `False` offsets teeth in the plane tangent to the patch. `True` moves them along the sphere's surface (more accurate for large radii). |
| `energy_scale_method` | `"linear"` | Each tooth's strength is divided by `n_teeth`, so a fully docked face has the same total energy as the original patch. `"log"` divides by `n_teeth·ln(n_teeth)` (experimental). `"none"` makes a docked face `n_teeth` times stronger. A number is used as a multiplier. |

Constraints:

- The interaction matrix must be one-to-one (every color binds exactly one other color).
- With `torsion=True`, self-binding colors aren't supported yet.
- At least one of `n_teeth > 1` or `dental_radius > 0`.

### Colors after conversion

With `torsion=True`, every tooth gets its own color, arranged so each tooth on one face lines up
with exactly one tooth on the partner face when they dock. Tooth colors are
`±(20 + (|c| − 21)·n_teeth + tooth_index)`. With 4 teeth, rule color 1 (PL color 21) becomes
colors 20–23, and rule color 2 becomes 24–27. With `torsion=False`, teeth keep the original
color.

Don't work tooth colors out by hand. Ask the particle set:

```python
mdt_ptypes.is_multidentate()       # True
mdt_ptypes.mdt_rep(patch_id)       # the teeth that replaced original patch `patch_id`
mdt_ptypes.get_src_map()           # mapping back to the original (unidentate) set
```
