from __future__ import annotations

from typing import Union

import networkx as nx


class Joint:
    """
    This class is designed to be immutable! Please respect!
    """
    # particle id for self. whether this is a unique id or a type id depends on the context
    _particle: int
    # particle id for previous neighbor. whether this is a unique id or a type id depends on the context
    _prev_neighbor: int
    # particle id for next neighbor. whether this is a unique id or a type id depends on the context
    _next_neighbor: int
    # tuple where the first value is the patch ID on self that connects to _prev_neighbor
    # and the second value is the patch ID on _prev_neigbor that connects to self
    _prev_top_edge: tuple[int, int]
    # tuple where the first value is the patch ID on self that connects to _next_neighbor
    # and the second value is the patch ID on _next_neigbor that connects to self
    _next_top_edge: tuple[int, int]

    def __init__(self,
                 particleid: int,
                 prevneighbor: int,
                 nextneighbor: int,
                 prevtopedge: tuple[int, int],
                 nexttopedge: tuple[int, int]):
        self._particle = particleid
        # assert prevneighbor != nextneighbor, "U turns are not allowed!"
        self._prev_neighbor = prevneighbor
        self._next_neighbor = nextneighbor
        # assert prevtopedge[1] != nexttopedge[0], "Invalid patch layout! Cannot have two patches with same ID"
        self._prev_top_edge = prevtopedge
        self._next_top_edge = nexttopedge

    def unique_tuple(self) -> tuple[int, int, int]:
        """
        WARNING: DON'T USE THIS FUNCTION FOR GENERIC JOINTS unless you want it to do what it's actually going to do
        in which case go ahead
        Returns:
            a simplified version of this behavior that - if ids are unique - will nonetheless act as a unique identifier within this structure

        """
        return (
            self._prev_neighbor, self._particle, self._next_neighbor
        )

    def particle(self) -> int:
        return self._particle

    def next_neighbor(self) -> int:
        return self._next_neighbor

    def prev_neighbor(self) -> int:
        return self._prev_neighbor

    def prev_edge(self) -> tuple[int, int]:
        return self._prev_top_edge

    def next_edge(self) -> tuple[int, int]:
        return self._next_top_edge

    def particle_prev_patch(self) -> int:
        return self.prev_edge()[0]

    def particle_next_patch(self) -> int:
        return self.next_edge()[0]

    def inner(self) -> tuple[int, int, int]:
        return self.particle_prev_patch(), self.particle(), self.particle_next_patch()

    def inner_equal(self, joint: Joint) -> bool:
        return self.inner() == joint.inner()

    def __hash__(self):
        """
        This can act as a unique identifier for a behavior within a structure or alternateively
        can be used to compare two behaviors

        Returns:
            an int that acts as a unique identifier for this behavior
        """
        return hash((
            self.unique_tuple(),
            self._prev_top_edge,
            self._next_top_edge
        ))

    def __invert__(self) -> Joint:
        """
        Returns:
            a behavior that is this behavior but in reverse

        """
        return Joint(self._particle,
                     self._next_neighbor, self._prev_neighbor,
                     self._next_top_edge, self._prev_top_edge)

    def __repr__(self):
        return f"({self.prev_neighbor()}){self.prev_edge()[1]}->{self.prev_edge()[0]}({self.particle()}){self.next_edge()[0]}->{self.next_edge()[1]}({self.next_neighbor()})"


class Pathway:
    """
    immutable.
    represents a path through a structure as a list of Joints
    mostly a wrapper for a tuple of behaviors
    this seemed like a good idea at some point
    """
    __inner: tuple[Joint]
    __cycle_id: int

    def __init__(self, *args: Joint):
        self.__inner = args
        # unique integer identifier that should be the same for all cycles containing the same set of joints
        # todo: prove that no cycle exists which contains the same set of joints but in a different order?
        self.__cycle_id = hash(frozenset(self.__inner))

    def particles(self) -> list[int]:
        return [self[0].prev_neighbor(), *[j.particle() for j in self], self[-1].next_neighbor()]

    def __len__(self):
        return len(self.__inner)

    def __iter__(self):
        return iter(self.__inner)

    def __reversed__(self):
        reverse_path = (~b for b in reversed(self.__inner))
        return Pathway(*reverse_path)

    def __hash__(self) -> int:
        return self.__cycle_id

    def __getitem__(self, item: Union[int, slice]) -> Union[Joint, Pathway]:
        if isinstance(item, int):
            if item < 0:
                return self.__inner[len(self) + item]
            else:
                return self.__inner[item]
        elif isinstance(item, slice):
            # slicing linear pathways is easy!
            if not self.is_cycle():
                return Pathway(*self.__inner[item])
            # slicing cyclic pathways is... not easy
            else:
                # make slice explicit
                if item.start is None:
                    item = slice(0, item.stop, item.step)
                elif item.start < 0:
                    item = slice(len(self) + item.start, item.stop, item.step)
                if item.stop is None:
                    item = slice(item.start, len(self), item.step)
                elif item.stop < 0:
                    item = slice(item.start, len(self) + item.stop, item.step)
                if item.step is None:
                    item = slice(item.start, item.stop, 1)
                assert item.step
                # base cases: complete slice 0:0:1 and 0:0:-1
                if item.start == item.stop:
                    sliced = self
                elif item.start < item.stop:
                    sliced = Pathway(*self.__inner[item])
                # if start is after stop and step is positive
                elif item.start >= item.stop:
                    # lshift the path so that the start point lines up with the path origin then return a slice from the shifted path
                    sliced = (self << item.start)[:(item.stop - item.start):abs(item.step)]
                else:
                    assert (item.start < item.stop) == (item.step > 0), "mismatch!!!!"
                    sliced = Pathway(*self.__inner[item])
                if item.step < 0:
                    sliced = reversed(sliced)
                return sliced

        else:
            raise Exception(f"Invalid indexer {type(item)}")

    def __repr__(self):
        if len(self) == 0:
            return "Empty pathway"
        elif len(self) == 1:
            return f"[{repr(self[0])}]"
        else:
            return "[" + repr(self[0]) + "".join(repr(x)[repr(x).find(f"({x.particle()})") + len(f"({x.particle()}")+1:] for x in self) + "]"
        # return repr(self.__inner)

    def index(self, item):
        if isinstance(item, Joint):
            return self.__inner.index(item)
        elif isinstance(item, int):
            return [j.particle() for j in self].index(item)
        else:
            raise ValueError(f"Invalid indexer {item}")

    def in_main(self, item: Union[int, tuple]):
        if isinstance(item, tuple):
            return any([j1.particle() in item and j2.particle() in item for j1, j2 in zip(self, self[1:])])
        elif isinstance(item, int):
            return any([j.particle() == item for j in self])

    def __contains__(self, item):
        """
        WILL RETURN TRUE FOR PARTICLES / EDGES IN ENDMEMBERS!!!
        Args:
            item:

        Returns:

        """
        if isinstance(item, Joint):
            try:
                self.index(item)
                return True
            except ValueError:
                return False
        elif isinstance(item, int):
            return any([j.particle() == item for j in self]) or self[0].prev_neighbor() == item or self[-1].next_neighbor() == item
        elif isinstance(item, tuple):
            # edge checker
            assert len(item) == 2
            return any([sum([j.prev_neighbor() in item, j.next_neighbor() in item, j.particle() in item]) == 2 for j in self])
        elif isinstance(item, Pathway):
            return self.is_subpathway(item)
        else:
            raise TypeError(f"Invalid arg to Joint::__contains__! Arg type: {type(item)}")

    def is_subpathway(self, other: Pathway) -> bool:
        """
        Returns true if the provided arguement is a subpathway of this pathway, false otherwise
        This method cares about directionality (e.g. a,b,c != c,b,a)
        but not about origin point if the path is cyclic (e.g. a,b,c == c,a,b)
        """
        g1 = self.graph()
        g2 = other.graph()
        def nodes_match(a, b):
            return a["pid"] == b["pid"]
        def edges_match(a, b):
            return a["patch"] == b["patch"]
        matcher = nx.isomorphism.GraphMatcher(g1, g2, node_match=nodes_match, edge_match=edges_match)
        match = matcher.subgraph_is_isomorphic()
        return match


    def is_cycle(self) -> bool:
        return self[0].prev_neighbor() == self[-1].particle() and self[-1].next_neighbor() == self[0].particle()

    def cycle_id(self) -> int:
        return self.__cycle_id

    def cycle_equiv(self, other: Pathway) -> bool:
        """

        given that both self and other are cycles,

        Args:
            other: another cycle

        Returns:
            true if there is a way to shift this cycle so that it's equal to other. false otherwise.
        """
        assert self.is_cycle() and other.is_cycle()
        return self.cycle_id() == other.cycle_id()
        # return len(self) == len(other) and any([self >> i == other for i in range(len(self))])

    def __add__(self, other: Joint):
        """
        Returns this pathway with a joint added to the end
        Args:
            other:

        Returns:

        """
        assert other is not None
        assert other.particle() == self[-1].next_neighbor()
        assert other.prev_neighbor() == self[-1].particle()
        return Pathway(*self.__inner, other)

    def __lshift__(self, n: int) -> Pathway:
        """
        Shifts the cycle n iterations to the left
        Args:
            n: number of items to shift

        Returns:
            self shifted n items to the left

        """
        assert self.is_cycle()
        if n == 0:
            return self
        shifted = Pathway(*self[-n:], *self[:-n])
        assert len(shifted) == len(self)
        return shifted

    def __rshift__(self, n: int) -> Pathway:
        """
        Shifts the cycle n iterations to the right
        Args:
            n: number of spaces to shift item to the right

        Returns:
            self, shifted n items to the right
        """
        assert self.is_cycle()
        if n == 0:
            return self
        shifted = Pathway(*self[n:], *self[:n])
        assert len(shifted) == len(self)
        return shifted

    def graph(self) -> nx.DiGraph:
        """
        Returns this design path as a networkX digraph, mainly for the purposes of drawing
        """
        g = nx.DiGraph()
        for p in self.particles():
            g.add_node(p, pid=p)
        for joint in self:
            g.add_edge(joint.particle(), joint.prev_neighbor(), patch=joint.particle_prev_patch())
            g.add_edge(joint.particle(), joint.next_neighbor(), patch=joint.particle_next_patch())
        return g

    def path_graph(self) -> nx.DiGraph:
        """
                Returns this design path as a networkX digraph, mainly for the purposes of drawing
                """
        g = nx.DiGraph()
        for p in self.particles():
            g.add_node(p, pid=p)
        g.add_edge(self[0].prev_neighbor(), self[0].particle(), patch=self[0].prev_edge()[1])
        for joint in self:
            # g.add_edge(joint.particle(), joint.prev_neighbor(), patch=joint.particle_prev_patch())
            g.add_edge(joint.particle(), joint.next_neighbor(), patch=joint.particle_next_patch())
        g.add_edge(self[-1].particle(), self[-1].next_neighbor(), patch=self[0].next_edge()[0])

        return g
