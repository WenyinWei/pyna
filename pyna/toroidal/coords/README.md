# Toroidal coordinate systems

## PEST terminology

**PEST** is a proper academic name and is always written in all capitals in
scientific prose, documentation, plot labels, and report titles.  Lower-case
`pest` is reserved for Python identifiers, command-line options, filenames,
and literal paths.  The coordinate system is therefore a **PEST coordinate
system**, never a prose-level “Pest” or “pest” coordinate system.

## Module boundary

`pyna.toroidal.coords` owns reusable PEST, Boozer, Hamada, equal-arc, VMEC
surface, and magnetic-axis coordinate construction.  Downstream projects
should use these builders instead of maintaining a second VMEC-angle to PEST-
angle inversion.

Current-line acceptance is intentionally outside this coordinate package.
TopoQuest applies its route-neutral first-poloidal-return gate after building
PEST coordinates: the first complete PEST poloidal winding must occur within
`2*pi/(3*Nfp)` of toroidal excursion.  Route A, Route B, and Route C bind their
own fields and evidence to that shared gate rather than changing coordinate
construction here.
