from pypatchy.structure import Structure


def test_bindings_list_self_loop_round_trips():
    """
    A vertex bound to itself (e.g. a MOSEA-reduced representative bonding to a rotated copy
    of itself) must round-trip through bindings_list with its two distinct directions intact,
    not collapsed into (u, du, u, du) -- see the fix in Structure.bindings_list.
    """
    # +x (1) binds to -x (0) on the same vertex; -y (2) binds to +y (3) on the same vertex
    s = Structure(bindings=[(1, 1, 1, 0), (1, 2, 1, 3)])
    bindings = s.bindings_list

    assert (1, 1, 1, 0) in bindings
    assert (1, 2, 1, 3) in bindings
    assert len(bindings) == 2
    # every direction must appear on exactly one side of exactly one binding
    seen_dirs = [d for u, d, v, dv in bindings for d in (d, dv)]
    assert sorted(seen_dirs) == [0, 1, 2, 3]


def test_bindings_list_self_loop_mixed_with_normal_binding():
    """A self-loop binding coexisting with an ordinary two-vertex binding shouldn't disturb
    either's reconstruction."""
    s = Structure(bindings=[(1, 1, 2, 0), (2, 2, 2, 3)])
    bindings = s.bindings_list

    assert (1, 1, 2, 0) in bindings
    assert (2, 2, 2, 3) in bindings
    assert len(bindings) == 2
