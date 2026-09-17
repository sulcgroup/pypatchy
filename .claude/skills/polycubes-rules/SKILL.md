---
name: polycubes-rules
description: Use when reading, writing, or reasoning about a PolycubesRule string (like "1|1|1|1|1|1_-1|||||") - parsing what a rule means, constructing a rule by hand for a target structure, understanding patch colors/rotation/binding, or the allostery/effects syntax. This is the string format every pypatchy ensemble's "rule" key and every polycube_rule_to_PL() call assumes but never spells out. Not for PL/oxDNA file formats after conversion - see patchy-model-formats for that.
---

# Polycubes rule strings

A `PolycubesRule` string is pypatchy's compact encoding of a combinatorial
assembly rule: a set of cube types, each with up to 6 colored, directional
patches (one per face), that determine how copies of these cubes can snap
together on a 3D grid. It's parsed/emitted by `PolycubesRule`
(`pypatchy/polycubeutil/polycubesRule.py`) and is the value every ensemble's
`"rule"` config key expects (see [[setup-run-ensemble]]), before
`polycube_rule_to_PL()` turns it into a `PLParticleSet` for actual simulation
(see [[patchy-model-formats]] for what happens after that).
An *assembled* structure built from a rule (`PolycubeStructure`, in
`polycubeutil/polycube_structure.py`) is a [[structure-topology]]
`TypedStructure` subclass — homomorphism checks, `.matrix()`, JSON export,
etc. from that skill all apply to it directly.

## The shape

```
<cube_type_0>_<cube_type_1>_..._<cube_type_N>
```
Cube types are separated by `_`. Each cube type is:
```
<left>|<right>|<bottom>|<top>|<back>|<front>
```
— **always exactly 6 fields**, separated by `|`, one per face, in this fixed
order (`RULE_ORDER`/`FACE_NAMES` in `polycubesRule.py`):

| index | face   | direction (x,y,z) |
|---|--------|---|
| 0 | left   | (-1, 0, 0) |
| 1 | right  | (1, 0, 0) |
| 2 | bottom | (0, -1, 0) |
| 3 | top    | (0, 1, 0) |
| 4 | back   | (0, 0, -1) |
| 5 | front  | (0, 0, 1) |

An empty field means no patch on that face. So `"1|1|1|1|1|1_-1|||||"` is two
cube types: **CT0** has a `color=1` patch on all 6 faces; **CT1** has a
single `color=-1` patch on its left face and nothing else (the other 5 `|`s
are still there, just with nothing between them).

## Each face field: `color[:rotation[:conditional]]`

- **`color`** (required, signed int; `0` or empty = no patch): the
  fundamental binding rule is **color `c` binds only color `-c`** — never the
  same sign, never itself. There's no inherent meaning to a color's magnitude
  beyond distinguishing which patches are compatible; `1` and `-1` bind each
  other, `2` and `-2` bind each other, `1` and `2` do not bind, `1` and `-2`
  do not bind. (An ordinary `PolycubesRule` can't express anything richer than
  this 1-to-1 signed convention — see "Escaping the 1-to-1 rule" below if you
  need to.)
- **`rotation`** (optional int, 0-3): the patch's orientation around its face
  normal, in 90° increments (`get_orientation(face_idx, rotation)`). Only
  matters for a rule with torsion: two patches bind only if their colors are
  negatives of each other **and** their orientations line up (relative angle
  0) when the two cubes are docked face-to-face — otherwise two same-color
  faces could bind at any rotation, which is usually not what you want for a
  rule meant to assemble a specific, oriented target structure. Omit it
  (`color` alone) for a rule where face orientation shouldn't matter.
- **`conditional`** (optional, 3rd field): an *allostery* condition — see below.

Don't provide a 3-field (`color:rotation:conditional`) and a 4-field
(`color:rotation:stateVar:activationVar`, the alternate "dynamic-effects"
form) patch on faces within the *same* rule string — the parser's own comment
warns against mixing them, and gives no error, just wrong-but-silent behavior.

## Worked examples

- `"1|1|1|1|1|1_-1|||||"` — the "3D cross": CT0's six `+1` patches each bind
  a copy of CT1's single `-1` "cap" patch (not each other — `+1` doesn't bind
  `+1`). This is the rule used throughout [[setup-run-ensemble]]'s examples.
- `"|1||-4||3:1_||||-3:3|-3:2_5||4:1|-1:2||_2:2||2:2||2:2|_-2:1|-2||-5:1||-5:2"`
  (from `examples/setup_sims_programmatic.py`) — 5 cube types, several with
  rotation values, and empty leading/trailing fields wherever a face has no
  patch (e.g. CT0 has no patch on `left`, a `1` on `right`, none on `bottom`,
  a `-4` on `top`, none on `back`, `3:1` — color 3, rotation 1 — on `front`).

## Allostery / effects (the fragile part)

A cube type can end in `@<effects>`, appended after the 6 `|`-separated
fields: `@src1,src2>target;src3>target2;...` — each `sources>target` clause
means "state variable `target` becomes active exactly when every source is
satisfied" (a positive source id means that variable must be *active*, a
negative one means it must be *inactive*; multiple sources are AND-ed
together, per `DynamicEffect.conditional()` in `effect.py`). A face's 3rd
patch component (`color:rotation:conditional`) is shorthand for the same
concept inlined onto one patch instead of declared separately.

Treat this machinery as legacy/fragile rather than something to reason about
from first principles — the parsing code's own comments call it out directly
("PLEASE for the LOVE of GOD do NOT combine static and dynamic patches",
"I've messed these up so badly omg", "shittiest version"). If a rule you're
given or writing actually uses `@effects`, read `PolycubesRule.__init__`'s
`rule_str` branch and `effect.py` directly rather than trusting a second-hand
summary for anything nontrivial — this doc is enough for the common
color/rotation-only case, not for debugging a broken allostery rule.

## Going from a rule to code, and back

```python
from pypatchy.polycubeutil.polycubesRule import PolycubesRule
rule = PolycubesRule(rule_str="1|1|1|1|1|1_-1|||||")
str(rule)   # round-trips back to the canonical 6-field-per-type form
            # (always all 6 fields, even if some are empty; `@effects`
            # appended only if the cube type actually has effects)
```
- `PolycubesRule(rule_json=[...])` — an alternate constructor from a list of
  dicts (the format used by the web polycubes tool / `Akodiat/polycubes`
  JSON), for when you're handed JSON instead of a string.
- `rule.draw_rule() -> drawsvg.Drawing` — renders the rule as an SVG (each
  cube type as an unfolded net) — useful for a human to sanity-check a rule
  visually rather than parsing the string by eye.
- Cube types default to being named `f"CT{index}"` (`CT0`, `CT1`, ...) unless
  given an explicit name.

## Escaping the 1-to-1 rule: `DetailedPolycubesRule`

A plain `PolycubesRule` can only express "color `c` binds `-c`, at matching
rotation or not at all." `DetailedPolycubesRule` (same module) replaces that
with an explicit `InteractionMatrix` (or `TorsionalInteractionMatrix` for
angle-dependent strengths), so it can express many-to-many colors,
self-interacting colors, or per-angle bond strengths that an ordinary rule
string can't — mirroring what `PLParticleSet.interaction_matrix()` already
lets you do on the PL side (see [[setup-run-ensemble]]'s "Particle types"
section). Built from a plain `rule_str`/`rule_json` with no `interaction_matrix`
given, it behaves identically to an ordinary `PolycubesRule` until you edit
its matrix.
