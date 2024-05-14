import numpy as np

from collections import Counter
from typing import Dict, List, Tuple, Iterator, Optional

from src.yard_simulation.target import Target
from src.yard_simulation.state import Start, Spawn, End
from src.yard_simulation.yard import AbstractYardSimulation, AbstractYardState


class YardSimulation(AbstractYardSimulation):

    def __init__(
        self,
        detector_matrix: np.ndarray,
        hit_proba: float = 0.80,
        spoil_proba: float = 0.50,
        max_stay: int = 5,
        cleanup_penalty: float = 9.0,
        num_steps: int = 1,
        seed: Optional[int] = None
    ):
        super().__init__()

        self.detector_matrix: List[Tuple[float, ...]] = [tuple(dp) for dp in detector_matrix]
        self.hit_proba: float = hit_proba
        self.spoil_proba: float = spoil_proba
        self.max_stay: int = max_stay
        self.cleanup_penalty: float = cleanup_penalty
        self.num_steps: int = num_steps

        self.step_idx: int = 0
        self.reset()

        self.water_consumption = Counter()
        self.lawn_damage = Counter()

        spawn_freqs = np.array([t.frequency for t in Target])
        self.spawn_probas: Tuple[float, ...] = tuple(spawn_freqs / np.sum(spawn_freqs))

        self.rng = np.random.default_rng(seed)
        self.choices: Dict[int, List[Target]] = {
            t: [Target(c) for c in self.rng.choice(a=len(Target), size=self.num_steps, p=p)]
            for t, p in enumerate(self.detector_matrix)
        }
        self.choices[-1] = [Target(c) for c in self.rng.choice(a=len(Target), size=self.num_steps, p=self.spawn_probas)]

    def step(self) -> Tuple[int, List[AbstractYardState]]:
        self.step_idx += 1

        transitions = list()
        while True:
            next_state = self.current_state.transit()
            transitions.append(next_state)
            self.current_state = next_state

            if self.current_state in (Spawn(self), End(self)):
                break

        return self.step_idx, transitions

    def run(self) -> Iterator[Tuple[int, List[AbstractYardState]]]:
        while self.current_state != End(self):
            yield self.step()

    def reset(self) -> 'YardSimulation':
        self.current_state = Start(self)
        self.intruder_class = Target.EMPTY
        self.detected_class = Target.EMPTY
        self.hit_successfully = False
        self.spoiled = False
        self.stayed_steps = 0

        return self

    @property
    def bird_present(self) -> bool:
        return self.intruder_class != Target.EMPTY

    @property
    def predicted_bird(self) -> bool:
        return self.detected_class != Target.EMPTY

    @property
    def target_vulnerable(self):
        success: bool = self.target_identified_correctly

        if self.intruder_class == Target.CHICKEN:
            success = success or self.detected_class == Target.TURKEY

        return success

    @property
    def target_identified_correctly(self) -> bool:
        return self.detected_class == self.intruder_class

    @property
    def max_stay_reached(self) -> bool:
        return self.stayed_steps >= self.max_stay

    @property
    def max_steps_reached(self) -> bool:
        return self.step_idx >= self.num_steps

    def get_random_target(self, kind: int) -> Target:
        return self.choices[kind].pop()

    def spawn_target(self) -> Target:
        return self.get_random_target(-1)

    def simulate_intrusion(self) -> Target:
        if not self.bird_present:
            self.intruder_class = self.spawn_target()

        return self.intruder_class

    def simulate_detection(self) -> Target:
        self.detected_class = self.get_random_target(self.intruder_class)

        return self.detected_class

    def simulate_sprinkling(self) -> bool:
        self.water_consumption[self.detected_class] += self.detected_class.health
        self.hit_successfully = self.bird_present and (self.rng.uniform() <= self.hit_proba) and self.target_vulnerable

        return self.hit_successfully

    def simulate_spoiling(self) -> bool:
        self.spoiled = self.bird_present and (self.rng.uniform() <= self.spoil_proba)
        if self.spoiled:
            self.lawn_damage[self.intruder_class] += self.intruder_class.damage

        return self.spoiled
