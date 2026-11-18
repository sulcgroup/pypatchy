# Background: patchy particle models

## What a patchy particle is

In a patchy particle model, each building block of an assembly is coarse-grained into a single
rigid body: usually a sphere, sometimes a small cluster of spheres. The body has a short-range
repulsion that keeps particles from overlapping, and a set of **patches** on its surface.
Patches are the only source of attraction. Two patches attract only if they are close
together, at a compatible relative orientation, and of compatible types.

The model is old (Smith and Nezbeda, 1984, is the earliest version we know of). We use it
because it's the cheapest model that still captures what matters for the self-assembly of
designed DNA nanostructures: each DNA-origami monomer becomes one particle, each set of sticky
ends becomes a patch, and the sticky-end sequences become patch colors. That lets us simulate
the assembly of hundreds to thousands of monomers.

In the lab, patchy particle designs usually start as **polycubes**: cube-shaped particles with
up to one patch per face, whose binding rules come from the SAT-based design tools (see
[Particle sets](04-particles.md)).

## What a patch has

Different oxDNA plugins store slightly different things, but conceptually a patch has:

| Property | Meaning |
|---|---|
| **position** | Location relative to the particle's center, in the particle's own frame. |
| **a1** | The direction the patch points (normally outward, along the position vector). Used for angular modulation. |
| **a2** | A vector perpendicular to `a1` that fixes the patch's rotation about `a1`. Only used by **torsional** models. |
| **color** | An integer type. Which colors bind which is decided by an interaction matrix, or by a color convention (e.g. `c` binds `-c`). |
| **strength** | Multiplies the bond energy. |

Particle orientations are stored in oxDNA's usual way, as two orthonormal vectors `a1` and
`a3` per particle in the configuration file. Patch vectors are rotated by that orientation.

## The interactions we use

All energies, distances, and temperatures are in **simulation units**. The particle diameter
is normally 1, so `PATCHY_radius = 0.5`, and temperature `T` is a reduced temperature
(`k_B T` in units of the bond energy scale). Typical assembly temperatures are around 0.01–0.1
depending on the model and patch strengths. Expect to sweep `T`.

### Romano model (`PatchyShapeInteraction`, `contrib/romano`)

Flavio Romano's model, used with the `flavio` (and allosteric `josh_flavio`) file formats.

**Excluded volume.** A truncated Lennard-Jones repulsion between particle centers. For
`shape = sphere`, with σ = 2R (R = `PATCHY_radius`):

- r < 0.9053σ: V = 4ε[(σ/r)¹² − (σ/r)⁶], with ε = 2
- 0.9053σ < r < 0.99888σ: a quadratic smoothing term, V = ε·b·(r − 0.99888σ)²
- beyond 0.99888σ: 0

`shape = hs` gives a hard sphere instead (MC only).

**Patch attraction.** For two patches a distance r apart, with r below a hard cutoff of 0.18:

```
V_patch(r) = K · [ −1.001 · exp(−½ (r/α)^10) − E_cut ] · (angular factor)
```

where α = `PATCHY_alpha` (default 0.12), K is the patch strength, and E_cut is the value of the
exponential term at the cutoff (about zero for α = 0.12). The `(r/α)^10` power makes the well
almost flat-bottomed out to r ≈ α and then fall off sharply. Smaller α means narrower patches.

> Earlier lab documentation gave this as `−1.001 ^ [−(r/α)^10]`. That was a transcription
> error. The exponential and the factor of ½ above are what the code computes.

**Angular modulation and torsion.** With `use_torsion` on (the default), the attraction is
multiplied by three angular factors:

- f(θ_a), where θ_a is the angle between the first patch's `a1` and the center-to-center
  direction;
- f(θ_b), the same for the second patch;
- f(θ_ab), the angle between the two patches' `a2` vectors. This is the **torsional** term:
  patches only bind fully when their `a2`s are parallel.

`use_torsion = partial` drops the torsional term (marked untested in the code), and
`use_torsion = 0` drops all angular dependence.

f(θ) is a piecewise function (with t = θ):

```
f(t) = 1 − a·t²          for t < t_s
f(t) = b·(t_c − t)²      for t_s < t < t_c
f(t) = 0                 for t > t_c
```

The constants come from **`narrow_type`**, an integer from 0 (most permissive) to 4
(narrowest). The functional form and its motivation are described in Bohlin et al.,
*Designing the Self-Assembly of Arbitrary Shapes Using Minimal Complexity Building Blocks*.
The values in the code are:

| `narrow_type` | t_s | t_c (rad) | a | b |
|---|---|---|---|---|
| 0 | 0.7 | 3.106 | 0.46 | 0.1339 |
| 1 | 0.2555 | 1.3046 | 3.0 | 0.7306 |
| 2 | 0.2555 | 0.7828 | 5.0 | 2.4228 |
| 3 | 0.17555 | 0.4382 | 13.0 | 8.6895 |
| 4 | 0.17555 | 0.3227 | 17.65 | 21.0506 |

So `narrow_type = 0` allows patches to bind up to about 178° off-axis (effectively no angular
constraint), while `narrow_type = 4` cuts off at about 18°.

**Other settings.** `no_multipatch` (default on) stops a patch from binding more than one
other patch at once; `same_type_bonding` (default on) allows particles of the same type to
bind. Colors with magnitude below 10 bind to the *same* color; colors with magnitude ≥ 10 bind
to their negative. pypatchy offsets polycube colors by 20 (see
[Particle sets](04-particles.md#colors)), so you'll normally only see the second rule.

The Romano model is **CPU-only**.

### Rovigatti model (`DetailedPatchySwapInteraction`, `contrib/rovigatti`)

Lorenzo Rovigatti's model, used with the `lorenzo` file format. Its main attraction is that it
**runs on CUDA**. It has **no torsion**: patches are points with no orientation.

**Excluded volume.** A WCA (purely repulsive, shifted Lennard-Jones) potential between centers,
with diameter 1.

**Patch attraction.** For patch–patch distance r below r_c = 1.5 σ_ss:

```
V_patch(r) = ε · A · exp( σ_ss / (r − r_c) ) · ( B / r⁴ − 1 )
```

with A and B chosen so that V(σ_ss) = −ε. σ_ss is `DPS_sigma_ss` (default 0.4; our configs
typically use 0.082). ε comes from the **interaction matrix** file and is indexed by **patch
type**, not color.

**Three-body (swap) term.** If one patch is near two others, an extra
`λ · E_ij · E_ik` is added. It's a penalty that makes "one patch, one bond" the low-energy
state while still letting bonds swap partners smoothly, which speeds up equilibration. λ is
`DPS_lambda` (default 1). `DPS_no_three_body = 1` disables it on CPU. **The CUDA version ignores
`DPS_no_three_body`**; set `DPS_lambda = 0` instead.

`DPS_normalise_patches` (default on) rescales patch positions to lie on the particle surface.
pypatchy's configs normally turn this off and write positions explicitly.

### Raspberry model (`RaspberryPatchyInteraction`, `contrib/evans`)

Josh's model, used with the `raspberry` format. It's for particles that aren't spheres. Instead
of one central repulsive sphere, each particle type has a set of **repulsion sites**: spheres at
arbitrary positions (and with arbitrary radii) relative to the particle center. Together they
make up the particle's shape, like the drupelets of a raspberry. Patches are as in the Romano
model, but **without torsion**. It runs on CPU and **CUDA**.

- **Excluded volume** acts only between repulsion sites (same functional form as Romano,
  scaled by the sum of the two sites' radii). There is no central sphere, so a type with no
  repulsion sites has no excluded volume at all.
- **Patch attraction** uses the same `−1.001 · exp(−½ (r/α)^10)` form, with a cutoff where the
  well has decayed to 10⁻⁴. Here α for a pair of patches is the **sum** of the two patches' α
  values. pypatchy doesn't write per-patch α, so every patch gets `PATCHY_alpha` from the input
  file, and **the effective width is 2 × `PATCHY_alpha`**. Halve whatever you'd use in the
  Romano model.
- **Angular modulation** (`patchy_angmod`, default on) uses the same f(θ) factors on θ_a and
  θ_b, and requires either `narrow_type` or the explicit `PATCHY_theta_*` constants.
- Bond strength between colors comes from an interaction matrix embedded in the topology.
  Without one, colors with |c| ≥ 20 bind their negative and smaller colors bind themselves.
- Each patch can hold at most one bond.

`dt` must be in the input file (it always is when pypatchy writes it).

### Comparing them

| Model | Plugin | pypatchy format | Torsion | GPU | Allostery | Notes |
|---|---|---|---|---|---|---|
| Romano | `romano` (mainline oxDNA) | `flavio` | yes | no | no | The reference model for polycube work. |
| Romano + allostery | `romano` in `sulcgroup/alloassembly` | `josh_flavio` | yes | no | yes | Older oxDNA codebase. Only needed for allosteric patches. |
| Rovigatti DPS | `rovigatti` (mainline oxDNA) | `lorenzo` | no | yes | no | Point patches. Use multidentate patches to get orientational specificity. |
| Raspberry | `evans` (`JeDeveloper/oxDNA`) | `raspberry` | no | yes | no | Arbitrary particle shapes via repulsion sites. |

The `subhajit` format exists in pypatchy but doesn't currently work end-to-end; see
[Models and file formats](03-models-and-formats.md#subhajit).

## Multidentate patches

A single point patch can't hold a bond's orientation. Two particles bonded by one patch each
can spin freely about the bond axis, and nothing stops a face from binding at the wrong
rotation. Torsional models fix this with the `a2` term, but the GPU models don't have one.

A **multidentate** patch replaces one patch with a ring of `n_teeth` smaller patches ("teeth")
spaced around the original patch position, at a distance `dental_radius` from it. When two
faces dock, several teeth bond at once. That pins the relative rotation, and as a bonus it's a
better picture of how a real DNA-origami face binds through several sticky ends at once.
pypatchy converts unidentate particle sets to multidentate automatically; see
[Particle sets](04-particles.md#multidentate-conversion).
