Tutoriels et exemples
=====================

Les notebooks publics sont regroupes par workflow. Le build de documentation
copie ``notebooks/`` dans l'arborescence source Sphinx ; les chemins ci-dessous
refletent donc l'organisation du dépôt.

Parcours d'apprentissage recommande
-----------------------------------

Commencez par :doc:`/fr/quickstart`, puis parcourez le workflow de géométrie
generique, les modeles stochastiques, puis les exemples de monodromie/RMP
toroidaux :

1. :doc:`/fr/mini-cases`
2. :doc:`sde-monte-carlo`
3. :doc:`/notebooks/tutorials/RMP_resonance_analysis`
4. :doc:`/notebooks/tutorials/monodromy_xcycle_analytic`
5. :doc:`/notebooks/tutorials/island_jacobian_analysis`
6. :doc:`/notebooks/tutorials/RMP_island_validation_solovev`

Systemes dynamiques generaux
----------------------------

.. toctree::
   :maxdepth: 1

   sde-monte-carlo

Le workflow de géométrie generique et le workflow de points fixes de stellarator
analytique sont maintenant integres au tutoriel de résonance RMP au lieu d'etre
publies comme notebooks autonomes uniquement textuels. Ce tutoriel montre la
même chaine de promotion : croisements échantillonnés -> géométrie de points
fixes -> classification X/O -> superpositions de variétés et de grilles de
coordonnées.

Équations différentielles stochastiques
---------------------------------------

Le tutoriel SDE est préexécuté localement parce que l'estimation de distribution
utilise souvent des dizaines ou centaines de milliers de chemins Monte Carlo.
GitHub Pages rend les sorties sauvegardees au lieu de depenser du temps CI sur
les cellules d'échantillonnage lourdes.

.. toctree::
   :maxdepth: 1
   :hidden:

   /notebooks/tutorials/sde_monte_carlo_distribution

Coordonnées magnétiques et équilibres
-------------------------------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/magnetic_coordinates_comparison

RMP, îlots et analyse de Poincaré
---------------------------------

Commencez par le notebook d'analyse de résonance lorsque vous étudiez la
topologie magnétique. Il couvre maintenant les gabarits RMP sans divergence, la
branche importante ``m=1``, la validation de points fixes ``cyna``, les atlas de
spectre magnétique ``B^r`` contravariant multi-composants, les cartes de
résonance modulaires ``q``/``m/n`` avec projections de Poincaré et superpositions
d'îlots facultatives, les spectres mixtes RMP/nRMP, la reponse nRMP totale de
tous les modes non resonants, la modulation de vitesse des lignes de champ et
les controles d'ordre de perturbation.

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/RMP_resonance_analysis
   /notebooks/tutorials/RMP_island_validation_solovev
   /notebooks/tutorials/island_jacobian_analysis

``RMP_resonance_exec.ipynb`` est conserve dans le dépôt comme variante
d'exécution/cache du workflow d'analyse de résonance, mais la documentation
publique pointe vers la version explicative ci-dessus.

Monodromie et variétés
----------------------

.. toctree::
   :maxdepth: 1

   /notebooks/tutorials/monodromy_mobius_saddle
   /notebooks/tutorials/monodromy_xcycle_analytic

Systemes dynamiques classiques et generaux
------------------------------------------

Le dépôt inclut aussi des notebooks legers sous ``notebooks/examples`` :
``Lorenz_attractor.ipynb``, ``resonance_1_1_map.ipynb``,
``Mobiusian_saddle_cycle.ipynb``, ``Xcycle_construction.ipynb`` et
``FPT_DX_to_DP_sympy.ipynb``. Ils sont conserves comme exemples sources plutot
que pages de documentation exécutées, car plusieurs sont des notebooks de type
brouillon sans titres de sections.

Figures statiques de tutoriels
------------------------------

Plusieurs workflows plus longs sont representes dans le dépôt par des figures
statiques et des sorties générées sous ``notebooks/tutorials``. Ils couvrent les
diagnostics de profil q, les coordonnées PEST/Boozer/Hamada/à arc égal, les
balayages de suppression d'îlots, le contrôle de phase, les variétés de
Poincaré et les exemples Solov'ev a point nul unique.
