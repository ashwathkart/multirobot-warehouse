from abc import ABC, abstractmethod
from typing import Optional, List


class Planner(ABC):
    """Base class for path planning algorithms"""

    @abstractmethod
    def plan(self, start: tuple, goal: tuple, sim, occupied: set) -> Optional[List[tuple]]:
        """
        Plan a path from start to goal.

        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            sim: Simulation world object with obstacles and dimensions
            occupied: Set of occupied positions (other robots)

        Returns:
            List of positions from start to goal (inclusive), or None if unreachable
        """
        pass

    def manhattan_distance(self, p1: tuple, p2: tuple) -> int:
        """Calculate Manhattan distance between two points"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def get_neighbors(self, pos: tuple, sim) -> list:
        """Get valid neighboring positions (4-directional movement)"""
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left
        for dx, dy in directions:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= new_pos[0] < sim.width and
                0 <= new_pos[1] < sim.height and
                not sim.obstacles[new_pos[0]][new_pos[1]]):
                neighbors.append(new_pos)
        return neighbors
