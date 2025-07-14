from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class OperationResult:
    operation: str
    computer_name: str
    success_count: int
    failure_count: int
    errors: list[str]
    duration: float
    timestamp: str

class PlatformOperations(ABC):
    def __init__(self, config: Dict[str, Any], logger=None, simulation_mode: bool = False):
        self.config = config
        self.logger = logger
        self.simulation_mode = simulation_mode

    @abstractmethod
    def delete_files(self, computer: Dict[str, Any]) -> OperationResult:
        pass

    @abstractmethod
    def copy_files(self, computer: Dict[str, Any]) -> OperationResult:
        pass

    @abstractmethod
    def start_applications(self, computer: Dict[str, Any]) -> OperationResult:
        pass

    @abstractmethod
    def kill_applications(self, computer: Dict[str, Any]) -> OperationResult:
        pass

    @abstractmethod
    def restart_applications(self, computer: Dict[str, Any]) -> OperationResult:
        pass
