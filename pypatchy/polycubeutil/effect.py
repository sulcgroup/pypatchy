from __future__ import annotations

from typing import Union


class Effect:
    def __init__(self, target: int):
        self._target = target

    def target(self) -> int:
        return self._target

    def toJSON(self) -> dict:
        return {
            "target": self.target()
        }


class StringConditionalEffect(Effect):
    def __init__(self, conditional, target):
        super().__init__(target)
        self._conditional = conditional

    def conditional(self) -> str:
        return self._conditional

    def setStr(self, newStr: str):
        self._conditional = newStr

    def toJSON(self) -> dict:
        return {
            **super(StringConditionalEffect, self).toJSON(),
            "conditional": self.target()
        }


class DynamicEffect(Effect):
    def __init__(self, source_variables: list[int], target: int):
        super().__init__(target)
        self._vars = source_variables

    def conditional(self) -> str:
        if len(self._vars) > 1:
            return "(" + "&".join([f"{v}" if v > 0 else f"!{-v}" for v in self._vars]) + ")"
        else:
            v = self._vars[0]
            return f"{v}" if v > 0 else f"!{-v}"

    def sources(self) -> list[int]:
        return self._vars

    def __str__(self) -> str:
        return f"[{','.join([str(s) for s in self.sources()])}]>{str(self.target())}"

    def toJSON(self) -> dict:
        return {
            **super(DynamicEffect, self).toJSON(),
            "sources": self._vars
        }


class EnvironmentalEffect(Effect):
    pass  # TODO


EFFECT_CLASSES = Union[StringConditionalEffect, DynamicEffect, EnvironmentalEffect]
