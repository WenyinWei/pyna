Tutoriels et exemples
=====================

Les notebooks publics sont regroupés par flux de travail. La génération de la
documentation copie ``notebooks/`` dans l'arborescence source Sphinx ; les
chemins ci-dessous reflètent donc l'organisation du dépôt.

Parcours d'apprentissage recommandé
-----------------------------------

Commencez par :doc:`/fr/quickstart`, puis parcourez le flux de géométrie
générique, les modèles stochastiques, puis les exemples de monodromie/RMP
toroïdaux :

1. :doc:`/fr/mini-cases`
2. :doc:`sde-monte-carlo`
3. :doc:`/notebooks/i18n/fr/tutorials/RMP_resonance_analysis`
4. :doc:`/notebooks/i18n/fr/tutorials/monodromy_xcycle_analytic`
5. :doc:`/notebooks/i18n/fr/tutorials/island_jacobian_analysis`
6. :doc:`/notebooks/i18n/fr/tutorials/RMP_island_validation_solovev`

Systèmes dynamiques généraux
----------------------------

.. toctree::
   :maxdepth: 1

   sde-monte-carlo

Le flux de géométrie générique et le flux de points fixes de stellarator
analytique sont maintenant intégrés au tutoriel de résonance RMP au lieu d'être
publiés comme notebooks autonomes uniquement textuels. Ce tutoriel montre la
même chaîne de promotion : croisements échantillonnés -> géométrie de points
fixes -> classification X/O -> superpositions de variétés et de grilles de
coordonnées.

Équations différentielles stochastiques
---------------------------------------

Le tutoriel SDE est préexécuté localement parce que l'estimation de distribution
utilise souvent des dizaines ou centaines de milliers de chemins Monte Carlo.
GitHub Pages rend les sorties sauvegardées au lieu de dépenser du temps CI sur
les cellules d'échantillonnage lourdes.

.. toctree::
   :maxdepth: 1
   :hidden:

   /notebooks/i18n/fr/tutorials/sde_monte_carlo_distribution

Coordonnées magnétiques et équilibres
-------------------------------------

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/fr/tutorials/magnetic_coordinates_comparison

RMP, îlots et analyse de Poincaré
---------------------------------

Commencez par le notebook d'analyse de résonance lorsque vous étudiez la
topologie magnétique. Il couvre maintenant les gabarits RMP sans divergence, la
branche importante ``m=1``, la validation de points fixes ``cyna``, les atlas de
spectre de Fourier du ``B^r`` contravariant multi-composant, les cartes de
résonance modulaires ``q``/``m/n`` avec projections de Poincaré et superpositions
d'îlots facultatives, les spectres mixtes RMP/nRMP, la réponse nRMP totale de
tous les modes non résonants, la modulation de vitesse des lignes de champ et
les contrôles d'ordre de perturbation.

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/fr/tutorials/RMP_resonance_analysis
   /notebooks/i18n/fr/tutorials/RMP_island_validation_solovev
   /notebooks/i18n/fr/tutorials/island_jacobian_analysis

``RMP_resonance_exec.ipynb`` est conservé dans le dépôt comme variante
d'exécution/cache du flux d'analyse de résonance, mais la documentation
publique pointe vers la version explicative ci-dessus.

Monodromie et variétés
----------------------

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/fr/tutorials/monodromy_mobius_saddle
   /notebooks/i18n/fr/tutorials/monodromy_xcycle_analytic

Systèmes dynamiques classiques et généraux
------------------------------------------

Le dépôt inclut aussi des notebooks légers sous ``notebooks/examples`` :
``Lorenz_attractor.ipynb``, ``resonance_1_1_map.ipynb``,
``Mobiusian_saddle_cycle.ipynb``, ``Xcycle_construction.ipynb`` et
``FPT_DX_to_DP_sympy.ipynb``. Ils sont conservés comme exemples sources plutôt
que pages de documentation exécutées, car plusieurs sont des notebooks de type
brouillon sans titres de sections.

Figures statiques de tutoriels
------------------------------

Plusieurs flux plus longs sont représentés dans le dépôt par des figures
statiques et des sorties générées sous ``notebooks/tutorials``. Ils couvrent les
diagnostics de profil q, les coordonnées PEST/Boozer/Hamada/à arc égal, les
balayages de suppression d'îlots, le contrôle de phase, les variétés de
Poincaré et les exemples Solov'ev à point nul unique.
