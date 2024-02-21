"""
This file contains code for testing the read/write capabilities of pypatchy
specifically with regard to
"""
from ..patchy.pl.patchyio import get_writer


def test_fwriter():
    """
    tests flavio-format writer
    """
    writer = get_writer("flavio")

def test_lwriter():
    """
    tests lorenzo-format writer
    """
    writer = get_writer("lorenzo")

def test_swriter():
    """tests subhajit-format writer"""
    writer = get_writer("subhajit")