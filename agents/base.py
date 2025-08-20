from abc import ABC, abstractmethod
from typing import Any, Dict


class Agent(ABC):
    """Base Agent template defining the run interface."""

    @abstractmethod
    def run(self, payload: Any, context: Dict[str, Any]):
        """Execute the agent with provided payload and context."""
        raise NotImplementedError
