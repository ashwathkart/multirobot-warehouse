import heapq
from typing import Optional, List
from .base_planner import Planner


class DijkstraPlanner(Planner):
    """Dijkstra path planning algorithm (A* with zero heuristic)"""

    def plan(self, start: tuple, goal: tuple, sim, occupied: set) -> Optional[List[tuple]]:
        """Plan a path using Dijkstra's algorithm"""
        if start == goal:
            return [start]

        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == goal:
                break

            for next_pos in self.get_neighbors(current, sim):
                if next_pos in occupied:
                    continue

                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    # Dijkstra uses cost only (no heuristic)
                    heapq.heappush(frontier, (new_cost, next_pos))
                    came_from[next_pos] = current

        if goal not in came_from:
            return None

        # Reconstruct path
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path
