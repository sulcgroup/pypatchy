``polycubeutil``
-------------------
This module contains functions for reading, writing, and manipulating polycube data.

.. contents::
    :local:

``polycube_structure.py``
..............................

.. autoclass:: pypatchy.polycubeutil.polycube_structure.PolycubeStructure

.. autofunction:: pypatchy.polycubeutil.polycube_structure.load_polycube

``polycubesRule.py``
...............................

.. autofunction:: pypatchy.polycubeutil.polycubesRule.diridx

.. autofunction:: pypatchy.polycubeutil.polycubesRule.rdir

.. autofunction:: pypatchy.polycubeutil.polycubesRule.get_orientation

.. autofunction:: pypatchy.polycubeutil.polycubesRule.getRotationMap

.. autoclass:: pypatchy.polycubeutil.polycubesRule.PolycubesPatch
    :members:

.. autoclass:: pypatchy.polycubeutil.polycubesRule.PolycubeRuleCubeType
    :members:

.. autoclass:: pypatchy.polycubeutil.polycubesRule.PolycubesRule
    :members:

``polycube_util.py``
.............................
This module contains high-level utility functions for polyccubes that don't require a polycubes binary but do require other ``polycubeutil`` modules.
Specifically, it requires the classes ``PolycubesStructure``, ``PolycubeStructureCube``, and ``PolycubesRule`` from ``polycube_structure.py``, ``polycubesRule.py``, and ``cube.py`` respectively.

.. autofunction:: pypatchy.polycubeutil.polycube_util.make_colors_consecutive

.. autofunction:: pypatchy.polycubeutil.polycube_util.get_fully_addressable_rule