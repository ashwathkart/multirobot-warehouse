from typing import Dict, Set, Tuple, Optional
import random


class ObstacleManager:
    """
    Manages dynamic obstacles in the warehouse.
    Obstacles can appear and disappear; robots share obstacle knowledge.
    """

    def __init__(self):
        self.static_obstacles: Set[Tuple[int, int]] = set()
        self.dynamic_obstacles: Dict[Tuple[int, int], Optional[float]] = {}
        self.fleet_known_obstacles: Set[Tuple[int, int]] = set()
        self.robot_discoveries: Dict[int, Set[Tuple[int, int]]] = {}

    def initialize_from_world(self, sim):
        """Initialize static obstacles from world"""
        self.static_obstacles.clear()
        for x in range(sim.width):
            for y in range(sim.height):
                if sim.obstacles[x][y]:
                    self.static_obstacles.add((x, y))
        self.fleet_known_obstacles = self.static_obstacles.copy()

    def add_obstacle(self, pos: Tuple[int, int], duration: Optional[float] = None,
                    end_time: Optional[float] = None):
        """
        Add a dynamic obstacle.

        Args:
            pos: Position (x, y)
            duration: How long the obstacle lasts (seconds) - NOT used in this version
            end_time: Absolute time when obstacle disappears (used internally)
        """
        if pos not in self.static_obstacles:
            self.dynamic_obstacles[pos] = end_time
            self.fleet_known_obstacles.add(pos)

    def remove_obstacle(self, pos: Tuple[int, int]):
        """Remove a dynamic obstacle"""
        if pos in self.dynamic_obstacles:
            del self.dynamic_obstacles[pos]
            # Keep in fleet knowledge until explicitly forgotten
            self.fleet_known_obstacles.discard(pos)

    def update_time(self, current_time: float):
        """Update dynamic obstacles based on current time"""
        expired = []
        for pos, end_time in self.dynamic_obstacles.items():
            if end_time is not None and current_time >= end_time:
                expired.append(pos)
        for pos in expired:
            self.remove_obstacle(pos)

    def robot_discovers_obstacle(self, robot_id: int, pos: Tuple[int, int]):
        """Robot discovers an obstacle (now shared with fleet)"""
        if robot_id not in self.robot_discoveries:
            self.robot_discoveries[robot_id] = set()
        self.robot_discoveries[robot_id].add(pos)
        self.fleet_known_obstacles.add(pos)

    def robot_discovers_clear(self, robot_id: int, pos: Tuple[int, int]):
        """Robot discovers an obstacle is gone"""
        if pos in self.dynamic_obstacles:
            self.remove_obstacle(pos)
        # Remove from fleet knowledge
        if pos in self.fleet_known_obstacles and pos not in self.static_obstacles:
            self.fleet_known_obstacles.discard(pos)

    def get_all_obstacles(self) -> Set[Tuple[int, int]]:
        """Get all known obstacles (static + dynamic + fleet knowledge)"""
        return self.fleet_known_obstacles.copy()

    def get_dynamic_obstacles(self) -> Set[Tuple[int, int]]:
        """Get only dynamic obstacles"""
        return set(self.dynamic_obstacles.keys())

    def is_obstacle(self, pos: Tuple[int, int]) -> bool:
        """Check if position is an obstacle"""
        return pos in self.fleet_known_obstacles

    def print_status(self):
        """Print current obstacle status"""
        print(f"\nObstacle Manager Status:")
        print(f"  Static obstacles: {len(self.static_obstacles)}")
        print(f"  Dynamic obstacles: {len(self.dynamic_obstacles)}")
        print(f"  Fleet known obstacles: {len(self.fleet_known_obstacles)}")
        if self.dynamic_obstacles:
            print(f"  Active dynamic obstacles: {list(self.dynamic_obstacles.keys())[:5]}...")
