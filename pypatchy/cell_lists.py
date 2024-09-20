import math
from dataclasses import dataclass, field
from typing import Union, Callable, Generator

import numpy as np
from .patchy_base_particle import PatchyBaseParticle


@dataclass
class Cell:
    idxs: np.ndarray = field()
    startcoords: np.ndarray = field()
    endcoords: np.ndarray = field()
    particles: list[PatchyBaseParticle] = field(default_factory=list)

    def __post_init__(self):
        assert all(self.endcoords >= self.startcoords)


class CellLists:
    # TODO: consider making this part of base Scene class
    # TODO: togglable periodic boundry conditions

    # todo: more with this
    cells_shape: tuple[int, int, int]  # n x cells, n y cells, n z cells
    cells: dict[tuple[int, int, int], Cell]
    particle_cells: dict[int, Cell]  # maps each particle idx to its cell
    cell_size: float  # cells should be cubic
    cells_matrix_dimensions: np.ndarray  # num x, num y, num z
    _box_size: np.ndarray  # make sure to keep consistant with superclass or whatever

    def __init__(self):
        # TODO: more specificity
        self.cell_size = 0
        self.cells = None
        self.particle_cells = None
        self._box_size = None
        self.cell_size = None

    def get_cell(self,
                 item: Union[tuple, np.ndarray, int, PatchyBaseParticle]) -> Cell:
        if isinstance(item, np.ndarray):
            cell_idxs = np.floor(item / self.cell_size)
            cell = self.cells[tuple(cell_idxs.astype(int))]
            assert np.all((cell.startcoords <= item) & (item < cell.endcoords))
            return cell
        elif isinstance(item, int):
            return self.particle_cells[item]
        elif isinstance(item, PatchyBaseParticle):
            return self.particle_cells[item.get_uid()]
        else:
            assert isinstance(item, tuple)
            return self.cells[item]

    def interaction_cells(self, cell: Cell) -> Generator[Cell, None, None]:
        """
        iterates cells that can interact with particles in the cell passed as an arg, in no particular order
        incl. cell
        """
        # TODO: better efficiency
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    idxs = (np.array([x, y, z]) + cell.idxs) % self.cells_matrix_dimensions
                    yield self.get_cell(tuple(idxs))

    def interaction_particles(self,
                              p: Union[np.ndarray,
                                       PatchyBaseParticle]) -> Generator[PatchyBaseParticle,
                                                                         None,
                                                                         None]:
        """
        iterates particles that can interact with this particle, in no particlar order
        """
        i = 0
        for cell in self.interaction_cells(self.get_cell(p)):
            for particle in cell.particles:
                if isinstance(p, np.ndarray) or particle.get_uid() != p.get_uid():
                    yield particle
            i += 1
        assert i == 27

    def box_size(self) -> np.ndarray:
        return self._box_size

    def set_box_size(self, box: Union[np.ndarray, list]):
        assert len(box) == 3
        oldbox = self.box_size()
        self._box_size = np.array(box)
        # clear cells
        if not np.array_equal(oldbox, self.box_size()) and self.cell_size is not None:
            self.apportion_cells()

    def compute_cell_size(self,
                          cell_size: float = None,
                          n_particles: int = None,
                          n_cells: int = None
                          ) -> float:
        # TODO: incl. warnings if cells are too small
        if n_particles is not None:
            n_cells = math.ceil((float(n_particles) / 2) ** (1 / 3) / 2)
            self.compute_cell_size(n_cells=n_cells)
        elif n_cells is not None:
            self.compute_cell_size(cell_size=self.box_size().max() / float(n_cells))
        elif cell_size is not None:
            self.cell_size = cell_size
        else:
            raise Exception("No parameter provided to use to compute cell size!")
        assert self.cell_size > 0
        return self.cell_size

    def get_cell_size(self) -> float:
        return self.cell_size

    def apportion_cells(self):
        """
        Resets cell dimeisons
        """
        assert self.cell_size is not None

        xs = np.arange(stop=self._box_size[0], step=self.get_cell_size())
        ys = np.arange(stop=self._box_size[1], step=self.get_cell_size())
        zs = np.arange(stop=self._box_size[2], step=self.get_cell_size())
        self.cells_matrix_dimensions = np.array([xs.size, ys.size, zs.size])
        self.cells = dict()
        self.particle_cells = dict()
        for xidx, x in enumerate(xs):
            for yidx, y in enumerate(ys):
                for zidx, z in enumerate(zs):
                    cell = Cell(
                        np.array([xidx, yidx, zidx]),
                        np.array([x, y, z]),
                        np.array([xs[xidx + 1] if xidx + 1 < xs.size else self.box_size()[0],
                                  ys[yidx + 1] if yidx + 1 < ys.size else self.box_size()[1],
                                  zs[zidx + 1] if zidx + 1 < zs.size else self.box_size()[2]
                                  ])
                    )
                    self.cells[(xidx, yidx, zidx)] = cell
        assert len(self.cells) == np.prod(self.cells_matrix_dimensions)
        # assert all([cell is not None for cell in self.particle_cells]), "Failed to place all particles in cells!"

    def apportion_cell_particles(self, particles: list[PatchyBaseParticle]):
        for particle in particles:
            assert (particle.position() >= 0).all(), "Conf is not inboxed!!"
            cell = self.get_cell(particle.position())
            # if cell.startcoords <= particle.position() < cell.endcoords:
            cell.particles.append(particle)
            self.particle_cells[particle.get_uid()] = cell
