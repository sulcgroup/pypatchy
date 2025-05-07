from typing import Union

from crisscross.core_functions.megastructures import Megastructure
from .patchy_slat import SlatsStructure, PatchySlat

from ..origami_deabstractor import OrigamiDeabstractor
from ..patchy.dna_particle import DNAParticle


class SlatOrigamiConverter(OrigamiDeabstractor):
    """
    similar system to patchy_origami_convert but for slats from William Shih's group
    ideally api should be very similar to PatchyOrigamiConverter
    """
    layer_distance: float
    ms: Megastructure

    def __init__(self,
                 structure: Megastructure,
                 sticky_length: Union[int, None] = None,
                 spacer_length: Union[int, None] = None,
                 expected_num_edges: Union[int, None] = None):
        super().__init__(sticky_length, spacer_length, expected_num_edges)
        self.ms = structure
        # todo incorporate scene api
        self.scene = SlatsStructure(structure)

    def link_patchy_particle(self, p: PatchySlat, dna: DNAParticle):
        pass

    def convert(self, unbound_stickies: bool = True):
        # unbound stickies will generally not do anything since slats structures are fully addressable
        slat_positions = self.ms.slat_array.slat_array
        """
        Converts a scene containing joined MGL particles to an oxDNA model consisting of
        DNA origamis joined by sticky end handles.
        """
        if self.is_track_conf_stickies() and len(self.sticky_book) > 0:
            print("WARNING: attempting call of convert() on a system where some sticky ends have already been "
                  "assigned! Program is probably about to crash.")
        assert len(self.scene.potentials) > 0, "You need to provide a patchy potential to use to identify patchy bonds"
        print("binding particles using 3p patches")
        self.bind_particles3p()
        assert self.expected_num_edges is None or self.expected_num_edges == -1 or self.bondcount == self.expected_num_edges, \
            f"Wrong number of bonds created! Expected {self.expected_num_edges} bonds, got {self.bondcount}."
        if self.is_track_conf_stickies() and unbound_stickies:
            self.add_unbound_stickies()
        print("Done!")