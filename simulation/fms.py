import random
from typing import List, Tuple, Dict, Optional


class FleetManagementSystem:
    """
    Manages dynamic task assignment for robots in a warehouse.
    Assigns robots to source/destination pairs based on proximity.
    """

    def __init__(self):
        self.robot_tasks: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]] = {}
        self.available_robots = set()
        self.sources = []
        self.destinations = []

    def manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def find_closest_robot(self, position: Tuple[int, int],
                          robot_positions: List[Tuple[int, int]],
                          available_only: bool = True) -> Optional[int]:
        """
        Find the robot closest to a given position.

        Args:
            position: Target position
            robot_positions: List of robot positions (index = robot_id)
            available_only: Only consider robots without assigned tasks

        Returns:
            Robot ID of closest robot, or None if no robots available
        """
        candidates = []
        for robot_id in range(len(robot_positions)):
            if available_only and robot_id not in self.available_robots:
                continue
            distance = self.manhattan_distance(position, robot_positions[robot_id])
            candidates.append((distance, robot_id))

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][1]

    def initialize_tasks(self, robot_positions: List[Tuple[int, int]],
                        sources: List[Tuple[int, int]],
                        destinations: List[Tuple[int, int]]):
        """
        Initialize task assignments for all robots using greedy assignment.

        Each robot is assigned one source and one destination.
        Uses greedy algorithm: assign closest robot to each unassigned source.

        Args:
            robot_positions: List of starting positions
            sources: List of source/input locations
            destinations: List of destination/output locations
        """
        self.sources = sources[:]
        self.destinations = destinations[:]
        self.available_robots = set(range(len(robot_positions)))
        self.robot_tasks.clear()

        # Ensure we have enough tasks for all robots
        num_robots = len(robot_positions)
        if len(sources) < num_robots or len(destinations) < num_robots:
            print(f"Warning: Not enough sources/destinations for all robots")
            print(f"Robots: {num_robots}, Sources: {len(sources)}, Destinations: {len(destinations)}")

        # Greedy assignment: for each source, assign closest available robot
        assigned_robots = set()
        for source in sources:
            if len(assigned_robots) >= num_robots:
                break

            closest_robot = None
            min_distance = float('inf')

            for robot_id in range(num_robots):
                if robot_id in assigned_robots:
                    continue
                distance = self.manhattan_distance(source, robot_positions[robot_id])
                if distance < min_distance:
                    min_distance = distance
                    closest_robot = robot_id

            if closest_robot is not None:
                # Assign this robot a random destination
                dest = random.choice(destinations)
                self.robot_tasks[closest_robot] = (source, dest)
                assigned_robots.add(closest_robot)
                self.available_robots.discard(closest_robot)

        # Assign remaining robots to random source/dest pairs
        for robot_id in range(num_robots):
            if robot_id not in assigned_robots and self.sources and self.destinations:
                source = random.choice(self.sources)
                dest = random.choice(self.destinations)
                self.robot_tasks[robot_id] = (source, dest)
                self.available_robots.discard(robot_id)

    def get_task(self, robot_id: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get assigned task for a robot (source, destination)"""
        return self.robot_tasks.get(robot_id)

    def complete_task(self, robot_id: int) -> bool:
        """Mark a task as complete and mark robot as available for new task"""
        if robot_id in self.robot_tasks:
            del self.robot_tasks[robot_id]
            self.available_robots.add(robot_id)
            return True
        return False

    def reassign_task(self, robot_id: int, new_source: Tuple[int, int],
                     new_dest: Tuple[int, int]) -> bool:
        """Reassign a robot to a new task"""
        if robot_id in self.robot_tasks:
            self.robot_tasks[robot_id] = (new_source, new_dest)
            return True
        return False

    def get_all_tasks(self) -> Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get all current task assignments"""
        return self.robot_tasks.copy()

    def print_tasks(self):
        """Print current task assignments"""
        print("\nFMS Task Assignments:")
        for robot_id, (source, dest) in sorted(self.robot_tasks.items()):
            print(f"  Robot {robot_id}: {source} -> {dest}")
