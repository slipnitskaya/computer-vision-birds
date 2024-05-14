import enum

from typing import cast

from src.yard_simulation.base import AbstractTarget


class Chicken(AbstractTarget):

    @property
    def health(self) -> float:
        return 4

    @property
    def damage(self) -> float:
        return 10

    @property
    def frequency(self) -> float:
        return 9


class Turkey(AbstractTarget):

    @property
    def health(self) -> float:
        return 7

    @property
    def damage(self) -> float:
        return 17

    @property
    def frequency(self) -> float:
        return 5


class Empty(AbstractTarget):

    @property
    def health(self) -> float:
        return 0

    @property
    def damage(self) -> float:
        return 0

    @property
    def frequency(self) -> float:
        return 7


class Target(enum.IntEnum):

    CHICKEN = Chicken()
    TURKEY = Turkey()
    EMPTY = Empty()

    health: float
    damage: float
    frequency: float

    def __new__(cls, target: AbstractTarget) -> 'Target':
        obj = AbstractTarget.__new__(cls, len(cls))

        obj._name_ = target.__class__.__name__.upper()
        obj._value_ = int(obj)
        obj.health = cast(float, target.health)
        obj.damage = cast(float, target.damage)
        obj.frequency = cast(float, target.frequency)

        return obj
