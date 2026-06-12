from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .train_recognizer_orchestration import TrainRecognizerOrchestration, TrainRecognizerUseCasePort

__all__ = [
    "TrainRecognizerUseCasePort",
    "TrainRecognizerOrchestration",
    "run_train_recognizer",
]


def __getattr__(name: str) -> Any:
    if name in {"TrainRecognizerUseCasePort", "TrainRecognizerOrchestration", "run_train_recognizer"}:
        from .train_recognizer_orchestration import (
            TrainRecognizerOrchestration,
            TrainRecognizerUseCasePort,
            run_train_recognizer,
        )
        return {
            "TrainRecognizerUseCasePort": TrainRecognizerUseCasePort,
            "TrainRecognizerOrchestration": TrainRecognizerOrchestration,
            "run_train_recognizer": run_train_recognizer,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
