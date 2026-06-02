# Reading the merge-tree plots

A merge-tree plot summarises a whole potential-energy surface (PES) as a single
tree. Each `*_merge_tree.png` produced by `MapMaker_FoS_SHE.py` has two panels
of this tree for one nucleus: **unpruned** (left) and **pruned** (right). This
page explains what the lines, points, and panels mean if you have never seen one.

## The one-sentence version

Pour water into the energy landscape from the bottom up: every separate puddle is
a **basin**, and the tree records the energy at which each puddle spills over a
**saddle** and merges into a deeper neighbour. The plot is that record.

![Map for 228Th](./pes_Z90_N138_c_vs_a4.png)

![Merge tree for 228Th](./pes_Z90_N138_merge_tree.png)

*`228Th` (`pes_Z90_N138_merge_tree.png`): unpruned (left, n=123) vs pruned
(right, n=9). Energy is the vertical axis; basins are ordered left-to-right by
elongation `c`. Refer back to this figure while reading the sections below.*

## The axes

- **Vertical axis — energy `E` (MeV).** Low is down, high is up. This is real
  energy; a point's height is the energy of the basin minimum or the saddle it
  marks.
- **Horizontal axis — ordering only.** Basins are placed left-to-right in order
  of their elongation coordinate `c` (small `c` = compact shape on the left,
  large `c` = stretched, near-scission shapes on the right). The *spacing*
  carries no physical meaning — it only keeps the branches from overlapping. The
  x-axis has no ticks for this reason.

## The lines

- **Vertical line = a basin.** It rises from the basin's own minimum (a marker at
  the bottom) up to the energy where that basin merges into a deeper one. A short
  vertical line is a shallow basin; a tall one is a deep, well-separated basin.
  The height of that line is the basin's **persistence** — the energy you must add
  before it spills into a deeper neighbour (defined precisely in the *Persistence*
  section below).
- **Horizontal line = a saddle (a merge event).** It is drawn at the energy of
  the mountain pass connecting two basins. Below that energy the two basins are
  separate puddles; at that energy they become one. Every horizontal line is the
  moment two branches join.

### The trunk (main line)

The single line that runs all the way to the top of the panel is the **root
basin** — the *globally* deepest basin on the surface. Every other basin merges
into it (directly or through intermediates), so it never merges *away*: it has no
saddle above it and simply continues to the top.

**The trunk does not necessarily lead to the ground state.** The root is whatever
point is lowest in energy *anywhere* on the surface. On a fission surface the
elongated scission / fission-exit region is often far more bound than the compact
ground state (the separating fragments are more stable), so the trunk frequently
descends to the fission-exit basin (white circle), not the green GS marker. In
`300Ubn`, for example, the trunk drops to a fission exit near −50 MeV while the
ground state sits around −6 MeV.

![Map for 300Ubn](./pes_Z120_N180_c_vs_a4.png)

![Merge tree for 300Ubn](./pes_Z120_N180_merge_tree.png)

*`300Ubn` (`pes_Z120_N180_merge_tree.png`): the trunk descends to the
fission-exit basin (white circle) near −50 MeV, far below the green ground state
at ≈ −6 MeV. Here the root is the global minimum, which is **not** the GS.*

The **ground state is identified separately**, not as the deepest basin overall:
it is the deepest basin among *compact* shapes only (small `c`, below the
`GROUND_STATE_C_THRESHOLD` of 1.4), and it is drawn with the green circle wherever
it falls on the tree. So to find the physics, read the **markers**, not the trunk:
green for the GS, squares for the barriers between it and the more elongated
basins.

## Persistence: why points disappear when pruned

**Persistence** measures how real a basin is. For any basin it is

```
persistence = (energy of the saddle where it merges) − (its own minimum energy)
```

i.e. how far you would have to raise the water before that puddle spills into a
deeper one. A deep, distinct valley has large persistence; a shallow dimple
sitting on the side of a bigger valley has tiny persistence. (The root basin
never merges, so its persistence is infinite.)

The **right panel is the same tree with the shallow basins removed.** Pruning
deletes every basin whose persistence is below a threshold
(`MERGE_TREE_PRUNE`, 0.4 MeV) and folds it into its deeper neighbour across the
shallow saddle. Those basins are almost always artefacts — discretisation noise
on a dense grid, not genuine physical wells — so removing them is exactly what
you want. **Named critical points are exempt:** the ground state, secondary
minimum, third minimum, fission exit, and their saddles (the labelled points
defined in *The points* below) are kept in the right panel even when their
persistence falls below the threshold, so the labelled story is never pruned away
(this is why the shallow 230Th fission exit discussed below survives).

Two things happen from left to right:

1. **Spurious branches vanish.** The "hairy" cluster of tiny twigs on the left
   collapses, because each of those was a sub-0.4-MeV dimple. The panel title
   counts them: e.g. `Unpruned (n=123)` → `Pruned ≥0.4 MeV (n=9)`.
2. **Survivors spread out.** X position is re-ranked over only the surviving
   basins, so the handful that remain use the full width and become easy to
   read. The *shape* of the surviving tree (who merges into whom, at what energy)
   is unchanged — only the clutter is gone.

If a basin you expected is missing from the right panel, it was shallower than
`MERGE_TREE_PRUNE` (0.4 MeV). Lower `MERGE_TREE_PRUNE` to keep more, raise it to
keep only the deepest valleys.

### One threshold, several names

`MERGE_TREE_PRUNE`, `SM_PERSISTENCE`, and `THIRD_MIN_PERSISTENCE` are distinct
constants that all happen to equal 0.4 MeV, because they all measure the same
thing: whether a basin is *valid* — a genuine physical well rather than
discretisation noise. The 0.4 MeV value was chosen by eye. The persistence
histogram is what will standardise it: real basins and noise dimples separate
into two populations, and the threshold belongs in the gap between them.

![Persistence histogram for 228Th](./pes_Z90_N138_persistence_hist.png)

*`228Th` (`pes_Z90_N138_persistence_hist.png`): basin count vs persistence, log
`y`. The ~114 noise dimples pile up below ≈ 0.25 MeV; the eight genuine basins
spread out as a sparse tail to ≈ 6 MeV. The two populations separate cleanly,
with an empty bin between them — 0.4 MeV falls in that gap, which is why the
by-eye cut works here. The count (n=122) is 228Th's basins with finite
persistence: one fewer than its unpruned merge tree (n=123, the figure at the top
of this page), because the root basin has infinite persistence and is dropped from
the histogram.*

## The points

Markers sit at basin minima (circles) and at the named saddles (squares). Colours
match the contour maps (`*_c_vs_a4.png`) so the two figures agree:

| Marker | Meaning |
| --- | --- |
| green circle | Ground state (GS) |
| red circle | Secondary minimum (SM, the fission isomer) |
| orange circle | Third minimum |
| white circle | Fission exit (the far, scission-side basin) |
| green square | Inner saddle (barrier GS → SM) |
| red square | Outer saddle |
| orange square | Third saddle |
| small grey circle | any other basin above the marker noise floor |

Each marker carries a small text label with its coordinates and energy, e.g.
`c=1.280  a4=0.130` / `E=0.733`.

Only basins that matter get a marker: a basin is labelled if it is one of the
named critical points, or if its **persistence** (see above) clears a small floor
(`MERGE_TREE_MARKER_MIN`, 0.1 MeV). The countless shallow dimples from numerical
noise still draw their faint branch line but get no marker — that is the "hair"
you see along the trunk on the left panel.

### How each named point is chosen

The named points are not picked independently. `select_fos_critical_points` finds
them in sequence, each built on the last. The rules:

- **Ground state (green circle)** — the lowest-energy basin with a confirmed
  minimum among *compact* shapes (`c < GROUND_STATE_C_THRESHOLD`, 1.4). Selected
  by depth, not persistence.
- **Secondary minimum (red circle)** — the lowest-energy *interior* basin more
  elongated than the GS. A candidate needs a confirmed minimum, `c > c(GS)`,
  finite persistence above `SM_PERSISTENCE` (0.4 MeV, a noise floor so a dimple
  beside the GS cannot win), no cell on the `c`-max or `a4`-max grid wall (the
  interior test excludes the scission valley the box clips at its corner), and a
  real **barrier outward toward scission** — its easiest outward saddle must sit
  at higher `c` than its own floor. The outward-barrier test rejects a deep well
  already on the scission slope, whose outward saddle is level in `c` with its
  minimum (256Fm, below). Chosen by depth, which stays robust when the isomer is
  shallow (light Th).
- **Inner saddle (green square)** — the highest saddle on the tree path from GS to
  SM, i.e. the GS → SM barrier.
- **Fission exit (white circle) + outer saddle** — an *easiest-barrier* search
  outward from the SM. Among basins with larger `c` whose own minimum sits on the
  `a4`-max wall (the surface still runs downhill there, marking a real scission
  exit rather than a basin that merely touches the wall), pick the one whose tree
  path from the SM crosses the **lowest** maximum saddle. That barrier saddle is
  the outer saddle.
- **Third minimum (orange circle)** — optional. The most *persistent* basin
  between SM and exit in elongation (`c(SM) < c < c(exit)`), with persistence above
  `THIRD_MIN_PERSISTENCE` (0.4 MeV), minimum energy below the outer saddle, an
  interior minimum (off both max walls), the same outward-barrier test as the SM
  (a real barrier at higher `c` toward scission, so the deep scission-slope basin
  cannot slip into this slot), and lying **past the SM's outer barrier** — it must
  reach the fission exit over a saddle *distinct* from the SM's controlling barrier.
  A second pocket at the floor of the secondary well that escapes over the *same*
  outer saddle as the SM is not a third minimum (256Fm, below). It is never sought
  directly; it emerges from these predicates.
- **Third saddle (orange square)** — appears only when a third minimum exists. It
  splits the SM → exit barrier in two: the **outer saddle** is recomputed as the
  highest saddle on SM → 3rd, and the **third saddle** is the highest saddle on
  3rd → exit.

**No-isomer fallback.** A nucleus with no fission isomer still gets a ground
state, a barrier, and an exit. If no basin clears the secondary-minimum test, the
easiest-barrier search runs outward from the **GS** instead of the SM. The
fission exit (white circle) is reported as usual, and its single controlling
saddle — the lone **fission barrier**, with no inner/outer split to make — is
drawn as the **inner saddle** (green square). The secondary minimum, third
minimum, outer and third saddles stay unset. So the minimal legible tree is
GS → fission saddle → exit.

### A worked reading (228Th)

Reading the `228Th` figure at the top of this page bottom to top:

- The **green** GS marker sits near `c≈1.23` at low energy — the deepest *compact*
  basin. (The trunk itself may run to this basin or to a deeper elongated one,
  depending on the nucleus; locate the GS by its green marker, not the trunk.)
- A **red** SM marker sits at larger `c` and higher energy — the fission isomer —
  joined to the GS branch by a **green-square inner saddle** (the first barrier).
- Continuing to larger `c`, a **red-square outer saddle** and an **orange** third
  minimum appear, then a **white** fission-exit basin far to the right.
- The left panel shows ~120 extra grey twigs along the way; the right panel keeps
  only the ~9 basins that are deeper than 0.4 MeV, so the GS → SM → exit story is
  immediately legible.

The barriers you care about for fission are simply the heights of the square
saddle markers above the GS: read them straight off the energy axis.

## Branches are not isolated tunnels

A common misreading is to treat each branch as a dead-end corridor whose only
exit is back over the saddle you entered by. **The tree is a topological
summary of which basins connect and at what energy, not a spatial map of paths.**
The horizontal axis is ordering, not a reaction coordinate, and a basin connects
to the rest of the surface through *every* saddle on its branch, not just the
lowest one near the ground state.

To find the barrier between any two basins, trace the path **along the tree**
from one, up to the first junction they share, and back down to the other; the
**highest saddle on that path** is the barrier you must cross — once. You do not
pass back through unrelated saddles on other branches.

**Worked example — `pes_Z90_N140_merge_tree.png` (230Th).** The ground state
reaches the secondary minimum over the **inner saddle** (`E=2.127` MeV). It is
tempting to conclude that the secondary minimum is a pocket hanging off that
saddle, so that going on to the third minimum means climbing back out over the
inner saddle the way you came. It does not. The secondary-minimum branch
continues *upward* to the **outer saddle** (`E=4.001` MeV) at larger elongation,
and that is where it joins the third-minimum and fission-exit side of the tree.
So the route secondary minimum → third minimum runs **forward, over the outer
saddle, toward greater `c`** — never backward over the inner saddle. The inner
saddle lies only on the ground-state ↔ secondary-minimum path and plays no part
in the later steps toward scission.

![Map for 230Th](./pes_Z90_N140_c_vs_a4.png)

![Merge tree for 230Th](./pes_Z90_N140_merge_tree.png)

*`230Th` (`pes_Z90_N140_merge_tree.png`): the secondary-minimum branch (red
circle, `E=0.649`) does not dead-end at the inner saddle (green square,
`E=2.127`); it continues up to the outer saddle (red square, `E=4.001`), where it
joins the third minimum (orange) and the fission exit — forward in `c`, not back
over the inner saddle.*

In short: each horizontal connector is a real mountain pass linking the two
regions it joins, and reaching one basin from another never requires re-crossing
a saddle that sits on a different branch.

## A shallow fission exit merges with its neighbour, not the trunk

Where a basin attaches to the tree is set by saddle energies, not by how
"important" the basin is — and two neighbouring isotopes show this cleanly.
Compare how the **white fission-exit basin** reaches the trunk in `230Th`
(`pes_Z90_N140`) and `228Th` (`pes_Z90_N138`).

In **230Th** the fission exit (`c=1.680`, `E=3.117`) is very shallow and sits
almost on top of the **third minimum** (orange circle, `c=1.610`, `E=2.569`) —
only `Δc=0.07` away. The low **third saddle** (orange square, `c=1.660`,
`E=3.142`) bounding the exit lies just `0.025` MeV above it, so the exit is barely
a basin at all. The two merge with *each other* first: the fission exit hangs off
the third-minimum branch and reaches the trunk **indirectly, through the third
minimum** — exit → third saddle → third minimum → trunk. (Its persistence is far
below the prune threshold; it survives the right panel only because named critical
points are exempt from pruning — see the *Persistence* section.)

In **228Th** the fission exit lies farther out (`c=2.000`, `E=0.118`) with no
shallow neighbour to pair with. Its lowest bounding saddle leads straight to the
root, so its branch joins the **trunk directly**, with no third-minimum
intermediate in between.

The rule behind both: a basin attaches wherever its lowest bounding saddle leads.
A shallow exit beside a deeper neighbour spills into that neighbour before the
water rises far enough to reach the trunk; an isolated exit holds out until it
merges with the root.

## A high basin is not a third minimum, even when persistence and position look right (236U)

`236U` has no third minimum, but it has a basin that looks like one until you
check its energy. Its named points are GS (`c=1.20`, `E=-2.70`), SM (`c=1.42`,
`E=-0.26`), outer saddle (`c=1.56`, `E=2.96`) and fission exit (`c=1.72`,
`E=0.66`) — no orange marker appears anywhere.

The tempting candidate is the grey circle high on the tree at `c=1.730`,
`a4=0.240`, `E=6.483`. From the basin list alone it has the two properties you
would screen for: it sits at large elongation (past the SM) and its persistence is
`0.692` MeV — well above the `THIRD_MIN_PERSISTENCE` floor (`0.4` MeV), and deeper than several basins that
*do* get marked. By location and persistence it reads as a third minimum.

Its **raw energy disqualifies it.** At `E=6.48` it sits ~3.5 MeV *above* the outer
saddle (`E=2.96`), and the third-minimum rule requires the well to lie *below* the
outer saddle. The reason is physical. A genuine third minimum is a pocket on the
fission-valley floor, lower than the barrier you cross to enter it, so the system
rests there on the way to scission. This basin is the opposite — a high side
pocket. To reach it the system would climb over the outer saddle and then a
further 3.5 MeV uphill, only to drop all the way back down to the fission exit at
`E=0.66`. At the low excitation energies of fission that detour is never taken;
the system passes underneath it. Persistence measures how *distinct* a basin is,
not how *low* — so a basin can be both persistent and well-placed in `c` yet still
sit far too high to be a valley-floor well.

![Map for 236U](./pes_Z92_N144_c_vs_a4.png)

![Merge tree for 236U](./pes_Z92_N144_merge_tree.png)

*`236U` (`pes_Z92_N144_merge_tree.png`): the persistent grey circle at `c=1.730`,
`E=6.483` is **not** drawn orange. It clears the persistence floor (`0.692` MeV)
and survives pruning, but its energy is ~3.5 MeV above the outer saddle (red
square, `E=2.956`), so the third-minimum energy ceiling rejects it. The GS → SM →
exit story carries no third minimum.*

## The deepest elongated well is not the secondary minimum when it sits on the scission slope (256Fm)

`256Fm`'s deepest interior well past the ground state lies at `c=1.830`,
`a4=0.320`, `E=-6.941` — deeper even than the GS itself (`E=-4.366`). By depth
alone it would be named the secondary minimum, which is how an earlier version of
the rule labelled it.

**It is not an isomer; it is the fission-valley floor, already on the scission
slope.** The tell is geometric: its easiest barrier *outward* toward the
`a4`-edge fission exit sits at the **same elongation**, `c=1.830` — there is no
saddle at higher `c` standing between the well and scission. A genuine fission
isomer must climb in `c` to escape; this basin does not. The **outward-barrier
test** encodes exactly that — a secondary-minimum (and third-minimum) candidate is
rejected unless its easiest outward saddle sits at higher `c` than its own floor —
and it removes the `c=1.830` basin from both slots.

The real secondary minimum is then `c=1.400`, `a4=0.150`, `E=-2.977`.

**256Fm has no third minimum, despite a second deep pocket nearby.** A basin at
`c=1.460`, `a4=0.270`, `E=-3.307` sits just beyond the SM and reads like a third
minimum by depth and position. It is not. Both the secondary minimum and this
pocket **connect to the trunk by the same saddle** — the `c=1.570`, `E=-1.359`
pass. The pocket sits on the SM's side of that single barrier, not beyond it:
crossing into it buys no progress toward scission, because the *same* saddle still
stands between the merged region and the exit. A genuine third minimum (the Th
chain) lies *past* the outer barrier, so its onward barrier to the exit is a
**different, lower** saddle. The **past-outer-barrier test** rejects the `c=1.460`
pocket: the `c=1.570` saddle is therefore the SM's **outer barrier**, and there is
no third minimum or third saddle.

![Map for 256Fm](./pes_Z100_N156_c_vs_a4.png)

![Merge tree for 256Fm](./pes_Z100_N156_merge_tree.png)

*`256Fm` (`pes_Z100_N156_merge_tree.png`): two rejections leave a clean GS → SM →
exit story. The deep well at `c=1.830`, `E=-6.941` (deeper than the GS) is **not**
the secondary minimum — its outward barrier sits at the same elongation `c=1.830`,
so the outward-barrier test marks it scission-slope floor, not an isomer. The
`c=1.460` pocket is **not** a third minimum — it reaches the exit over the same
`c=1.570` saddle as the SM (red square, the outer barrier), so the
past-outer-barrier test rejects it. The secondary minimum is `c=1.400` (red
circle).*
