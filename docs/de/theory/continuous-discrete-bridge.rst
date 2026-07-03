Kontinuierliche und diskrete Geometrie
======================================

pyna verwendet getrennte Objektfamilien für kontinuierliche und diskrete
dynamische Systeme.

Kontinuierliche Seite:

- ``Trajectory`` ist abgetastete endliche Geometrie.
- ``Cycle`` ist ein periodischer Orbit eines Flusses.
- ``Tube`` ist eine Resonanzzone um einen elliptischen Zyklus, möglicherweise
  durch hyperbolische Zyklen begrenzt.
- ``TubeChain`` gruppiert Tubes, die zu einer Resonanz gehören.

Diskrete Seite:

- ``Orbit`` ist abgetastete Geometrie von Karteniterationen.
- ``PeriodicOrbit`` ist ein geschlossener Orbit einer Abbildung.
- ``Island`` ist eine reduzierte Resonanzinsel auf einem Schnitt.
- ``IslandChain`` ist die Inselkette auf Schnittebene.

Die Brücke zwischen beiden Seiten ist ein Schnitt.  Das Schneiden eines
``Cycle`` mit einem Poincaré-Schnitt erzeugt einen ``PeriodicOrbit`` der
Return Map.  Das Schneiden eines ``Tube`` erzeugt eine ``IslandChain``.  Das
Schneiden einer ``TubeChain`` führt die Inselketten ihrer Tubes zusammen.

Diese Trennung ist beabsichtigt.  Eine numerische Trajektorie kann nützliche
Geometrie sein, ohne Invarianz zu beweisen.  Builder und Adapter machen
Hochstufung daher explizit: Benutzer können Schließprüfungen verlangen, bevor
eine abgetastete Trajektorie zu einem ``Cycle`` wird oder bevor Kartenstichproben
zu einem ``PeriodicOrbit`` werden.

Dasselbe Vokabular wird von generischen endlichdimensionalen Systemen und von
der toroidalen Spezialisierung magnetischer Feldlinien geteilt.  Generische
Wurzeln sind als ``pyna.topo.CoreTube`` und verwandte Namen verfügbar;
toroidale Defaults bleiben als ``pyna.topo.Tube``, ``pyna.topo.Cycle`` und
``pyna.topo.IslandChain`` verfügbar.

Siehe auch
----------

- :doc:`/de/mini-cases`
- :doc:`/notebooks/i18n/de/tutorials/RMP_resonance_analysis`
- :doc:`/notebooks/i18n/de/tutorials/monodromy_xcycle_analytic`
- :doc:`/notebooks/i18n/de/tutorials/island_jacobian_analysis`
