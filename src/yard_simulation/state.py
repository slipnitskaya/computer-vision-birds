from typing import Union

from src.yard_simulation.target import Target
from src.yard_simulation.yard import AbstractYardState


class Start(AbstractYardState):

    def transit(self) -> 'Spawn':
        return Spawn(self.state_machine)


class Spawn(AbstractYardState):

    def transit(self) -> Union['Intrusion', 'Empty', 'End']:
        self.state_machine.stayed_steps += 1

        self.state_machine.simulate_intrusion()

        next_state: Union['Intrusion', 'Empty', 'End']
        if self.state_machine.max_steps_reached:
            next_state = End(self.state_machine)
        elif self.state_machine.bird_present:
            next_state = Intrusion(self.state_machine)
        else:
            next_state = Empty(self.state_machine)

        return next_state


class IntrusionStatus(AbstractYardState):

    intruder_class: Target

    def transit(self) -> Union['Detected', 'NotDetected']:
        self.state_machine.simulate_detection()
        self.intruder_class = self.state_machine.intruder_class

        next_state: Union['Detected', 'NotDetected']
        if self.state_machine.predicted_bird:
            next_state = Detected(self.state_machine)
        else:
            next_state = NotDetected(self.state_machine)

        return next_state


class Intrusion(IntrusionStatus):
    pass


class Empty(IntrusionStatus):
    pass


class DetectionStatus(AbstractYardState):

    detected_class: Target

    def transit(self) -> 'DetectionStatus':
        self.detected_class = self.state_machine.detected_class

        return self


class Detected(DetectionStatus):

    def transit(self) -> 'Sprinkling':
        super().transit()

        return Sprinkling(self.state_machine)


class NotDetected(DetectionStatus):

    def transit(self) -> Union['Attacking', 'NotAttacked']:
        super().transit()

        next_state: Union['Attacking', 'NotAttacked']
        if self.state_machine.bird_present:
            next_state = Attacking(self.state_machine)
        else:
            next_state = NotAttacked(self.state_machine)

        return next_state


class Sprinkling(AbstractYardState):

    def transit(self) -> Union['Hit', 'Miss']:
        self.state_machine.simulate_sprinkling()

        next_state: Union['Hit', 'Miss']
        if self.state_machine.hit_successfully:
            next_state = Hit(self.state_machine)
        else:
            next_state = Miss(self.state_machine)

        return next_state


class Hit(AbstractYardState):

    def transit(self) -> 'Leaving':
        return Leaving(self.state_machine)


class Miss(AbstractYardState):

    def transit(self) -> Union['Attacking', 'Spawn']:
        next_state: Union['Attacking', 'Spawn']
        if self.state_machine.bird_present:
            next_state = Attacking(self.state_machine)
        else:
            next_state = Spawn(self.state_machine)

        return next_state


class Attacking(AbstractYardState):

    def transit(self) -> Union['Attacked', 'NotAttacked']:
        self.state_machine.simulate_spoiling()

        next_state: Union['Attacked', 'NotAttacked']
        if self.state_machine.spoiled:
            next_state = Attacked(self.state_machine)
        else:
            next_state = NotAttacked(self.state_machine)

        return next_state


class AfterAttacking(AbstractYardState):

    def transit(self) -> Union['Leaving', 'Spawn']:
        next_state: Union['Leaving', 'Spawn']
        if self.state_machine.max_stay_reached:
            next_state = Leaving(self.state_machine)
        else:
            next_state = Spawn(self.state_machine)

        return next_state


class Attacked(AfterAttacking):

    def transit(self) -> Union['Leaving', 'Spawn']:
        return super().transit()


class NotAttacked(AfterAttacking):

    def transit(self) -> Union['Leaving', 'Spawn']:
        return super().transit()


class Leaving(AbstractYardState):

    def transit(self) -> 'Spawn':
        return Spawn(self.state_machine.reset())


class End(AbstractYardState):

    def transit(self) -> 'End':
        return self
