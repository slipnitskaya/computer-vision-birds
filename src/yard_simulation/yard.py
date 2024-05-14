import abc

from collections import Counter

from src.yard_simulation.target import Target
from src.yard_simulation.base import AbstractState, AbstractSimulation


class AbstractYardSimulation(AbstractSimulation, abc.ABC):

    current_state: 'AbstractYardState'
    intruder_class: Target
    detected_class: Target
    hit_successfully: bool
    spoiled: bool
    stayed_steps: int
    water_consumption: Counter
    lawn_damage: Counter

    @property
    @abc.abstractmethod
    def bird_present(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def predicted_bird(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def max_stay_reached(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def max_steps_reached(self) -> bool:
        pass

    @abc.abstractmethod
    def reset(self) -> 'AbstractYardSimulation':
        pass

    @abc.abstractmethod
    def simulate_intrusion(self) -> Target:
        pass

    @abc.abstractmethod
    def simulate_detection(self) -> Target:
        pass

    @abc.abstractmethod
    def simulate_sprinkling(self) -> bool:
        pass

    @abc.abstractmethod
    def simulate_spoiling(self) -> bool:
        pass


class AbstractYardState(AbstractState, abc.ABC):
    state_machine: AbstractYardSimulation
