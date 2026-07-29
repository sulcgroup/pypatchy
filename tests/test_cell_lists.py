import numpy as np
import pytest

from pypatchy.cell_lists import CellLists
from pypatchy.patchy_base_particle import PatchyBaseParticle


class DummyParticle(PatchyBaseParticle):
    """
    Minimal concrete PatchyBaseParticle for exercising CellLists without
    depending on a specific particle implementation (mgl, polycube, etc.)
    """
    def rotation(self):
        return np.eye(3)

    def rotate(self, rotation):
        pass

    def patches(self):
        return []

    def patch(self, idx):
        raise IndexError

    def num_patches(self):
        return 0


def make_particle(uid: int, position) -> DummyParticle:
    return DummyParticle(uid=uid, type_id=0, position=np.array(position, dtype=float))


@pytest.fixture
def cell_lists() -> CellLists:
    cl = CellLists()
    cl.set_box_size([4.0, 4.0, 4.0])
    cl.compute_cell_size(cell_size=1.0)
    cl.apportion_cells()
    return cl


def test_compute_cell_size_direct():
    cl = CellLists()
    assert cl.compute_cell_size(cell_size=2.5) == 2.5
    assert cl.get_cell_size() == 2.5


def test_compute_cell_size_from_n_cells():
    cl = CellLists()
    cl.set_box_size([8.0, 8.0, 8.0])
    assert cl.compute_cell_size(n_cells=4) == 2.0


def test_compute_cell_size_from_n_particles():
    cl = CellLists()
    cl.set_box_size([8.0, 8.0, 8.0])
    # n_cells = ceil((16/2)**(1/3)/2) = ceil(1) = 1 -> cell_size = box max / 1
    assert cl.compute_cell_size(n_particles=16) == 8.0


def test_compute_cell_size_requires_a_param():
    cl = CellLists()
    with pytest.raises(ValueError):
        cl.compute_cell_size()


def test_apportion_cells_grid_shape(cell_lists: CellLists):
    assert list(cell_lists.cells_matrix_dimensions) == [4, 4, 4]
    assert len(cell_lists.cells) == 64
    assert len(cell_lists.cells) == np.prod(cell_lists.cells_matrix_dimensions)


def test_apportion_cells_bounds(cell_lists: CellLists):
    cell = cell_lists.get_cell((0, 0, 0))
    assert np.array_equal(cell.startcoords, np.array([0.0, 0.0, 0.0]))
    assert np.array_equal(cell.endcoords, np.array([1.0, 1.0, 1.0]))

    last_cell = cell_lists.get_cell((3, 3, 3))
    # last cell along each axis should end exactly at the box boundary
    assert np.array_equal(last_cell.endcoords, cell_lists.box_size())


def test_set_box_size_reapportions_cells():
    cl = CellLists()
    cl.set_box_size([4.0, 4.0, 4.0])
    cl.compute_cell_size(cell_size=1.0)
    cl.apportion_cells()
    assert len(cl.cells) == 64

    cl.set_box_size([2.0, 2.0, 2.0])
    # changing box size with a cell size already set should re-apportion automatically
    assert len(cl.cells) == 8


def test_get_cell_by_position(cell_lists: CellLists):
    cell = cell_lists.get_cell(np.array([1.5, 2.5, 0.5]))
    assert tuple(cell.idxs) == (1, 2, 0)


def test_apportion_cell_particles_and_lookup(cell_lists: CellLists):
    particles = [
        make_particle(0, [0.5, 0.5, 0.5]),
        make_particle(1, [1.5, 0.5, 0.5]),
        make_particle(2, [3.9, 3.9, 3.9]),
    ]
    cell_lists.apportion_cell_particles(particles)

    cell0 = cell_lists.get_cell(0)
    assert particles[0] in cell0.particles
    assert tuple(cell0.idxs) == (0, 0, 0)

    cell1 = cell_lists.get_cell(particles[1])
    assert tuple(cell1.idxs) == (1, 0, 0)

    cell2 = cell_lists.get_cell(2)
    assert tuple(cell2.idxs) == (3, 3, 3)


def test_apportion_cell_particles_rejects_out_of_box(cell_lists: CellLists):
    particles = [make_particle(0, [-0.1, 0.5, 0.5])]
    with pytest.raises(ValueError):
        cell_lists.apportion_cell_particles(particles)


def test_interaction_cells_covers_27_neighbors_with_wraparound(cell_lists: CellLists):
    corner_cell = cell_lists.get_cell((0, 0, 0))
    neighbors = list(cell_lists.interaction_cells(corner_cell))
    assert len(neighbors) == 27

    neighbor_idxs = {tuple(int(i) for i in c.idxs) for c in neighbors}
    assert len(neighbor_idxs) == 27
    # periodic boundary conditions should wrap -1 around to the far edge (3)
    assert (3, 3, 3) in neighbor_idxs
    assert (0, 0, 0) in neighbor_idxs


def test_interaction_particles_excludes_self_and_finds_neighbors(cell_lists: CellLists):
    particles = [
        make_particle(0, [0.5, 0.5, 0.5]),  # cell (0,0,0)
        make_particle(1, [1.5, 0.5, 0.5]),  # cell (1,0,0), adjacent
        make_particle(2, [3.5, 0.5, 0.5]),  # cell (3,0,0), adjacent via wraparound
        make_particle(3, [2.5, 2.5, 2.5]),  # cell (2,2,2), not adjacent
    ]
    cell_lists.apportion_cell_particles(particles)

    neighbors = list(cell_lists.interaction_particles(particles[0]))
    neighbor_uids = {p.get_uid() for p in neighbors}

    assert 0 not in neighbor_uids  # never includes itself
    assert {1, 2}.issubset(neighbor_uids)
    assert 3 not in neighbor_uids