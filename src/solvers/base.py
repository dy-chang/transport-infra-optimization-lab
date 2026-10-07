"""
Abstract base solver interface for facility location optimization.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np

from src.models import CandidateSite, DemandNode, OptimizationSolution, ScenarioConfig


class BaseSolver(ABC):
    """
    Abstract Base Class for transportation facility location solvers.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def solve(
        self,
        scenario: ScenarioConfig,
        candidates: List[CandidateSite],
        demand_nodes: List[DemandNode],
        dist_cand_to_dem: np.ndarray,
        dist_cand_to_cand: Optional[np.ndarray] = None,
    ) -> OptimizationSolution:
        """
        Solve facility location problem and return an OptimizationSolution object.
        """
        pass
