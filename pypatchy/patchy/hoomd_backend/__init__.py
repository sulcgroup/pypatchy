"""
pypatchy's HOOMD-blue backend: the rigid-body patchy model (model), parameter handling (params), conversion between
scenes and GSD frames (gsdio) and the stage runner (run_stage). The ensemble-facing part is
pypatchy.patchy.engines.hoomd_engine.HOOMDEngine.

Importing this package doesn't import hoomd; only running a stage does (and reading or writing GSD files needs gsd).
"""
