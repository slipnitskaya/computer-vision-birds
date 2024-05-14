import abc

from typing import List, Tuple, Iterator


SEED: int = 42


class AbstractSimulation(abc.ABC):

    @abc.abstractmethod
    def step(self) -> Tuple[int, List['AbstractState']]:
        pass

    @abc.abstractmethod
    def run(self) -> Iterator[Tuple[int, List['AbstractState']]]:
        pass


class AbstractState(abc.ABC):

    def __init__(self, state_machine: AbstractSimulation):
        super().__init__()
        self.state_machine = state_machine

    def __eq__(self, other):
        return self.__class__ is other.__class__

    def __repr__(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def transit(self) -> 'AbstractState':
        pass


class AbstractTarget(int, abc.ABC):

    @property
    @abc.abstractmethod
    def health(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def damage(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def frequency(self) -> float:
        pass
