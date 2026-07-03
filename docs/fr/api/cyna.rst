Couche d'accélération cyna
==========================

``cyna`` est la couche d'accélération C++ livree avec pyna. Elle est utilisée
la ou les boucles chaudes Python ne sont pas acceptables : traçage de lignes de
champ, lots de Poincaré, balayages de points fixes, longueur de connexion /
impacts sur paroi, champs de bobines et noyaux de théorie de perturbation
fonctionnelle.

Contrat de build
----------------

``pyna._cyna`` attend un binaire ``_cyna_ext`` compile dans le paquet. Les
installations depuis les sources le construisent via xmake ; les roues PyPI
l'incluent. Voir :doc:`../installation` pour la configuration de plateforme et
les drapeaux CUDA.

L'ordre canonique du cache de champ cylindrique est :

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

Utilisez :func:`pyna._cyna.prepare_field_cache` pour convertir un
``pyna.fields.VectorFieldCylind`` ou un ancien dict en tableaux C-contigus.

API de haut niveau et bas niveau
--------------------------------

Preferez les wrappers de haut niveau pour le code applicatif :

- ``pyna.flt`` et ``pyna.toroidal.flt`` pour le traçage
- ``pyna.topo`` pour les cartes de Poincaré, cycles, îlots, variétés et
  reponse FPT
- ``pyna.toroidal.coils`` pour la construction de champs de bobines

N'utilisez ``pyna._cyna`` directement qu'aux frontières de bridge, pour les
diagnostics ou lors de l'ecriture d'un nouveau wrapper de haut niveau.

Référence du wrapper Python
---------------------------

.. automodule:: pyna._cyna
   :no-index:
   :members:
   :show-inheritance:

Auxiliaires utilitaires
----------------------

.. automodule:: pyna._cyna.utils
   :no-index:
   :members:
   :show-inheritance:
