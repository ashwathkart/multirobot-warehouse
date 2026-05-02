from .astar_planner import AStarPlanner
from .dijkstra_planner import DijkstraPlanner


class PlannerFactory:
    """Factory for creating planner instances"""

    AVAILABLE_PLANNERS = {
        'astar': AStarPlanner,
        'dijkstra': DijkstraPlanner,
    }

    @staticmethod
    def create(planner_name: str):
        """
        Create a planner instance by name.

        Args:
            planner_name: Name of the planner ('astar', 'dijkstra', etc.)

        Returns:
            Planner instance

        Raises:
            ValueError: If planner_name is not recognized
        """
        planner_name = planner_name.lower()
        if planner_name not in PlannerFactory.AVAILABLE_PLANNERS:
            available = ', '.join(PlannerFactory.AVAILABLE_PLANNERS.keys())
            raise ValueError(
                f"Unknown planner '{planner_name}'. Available planners: {available}"
            )
        return PlannerFactory.AVAILABLE_PLANNERS[planner_name]()

    @staticmethod
    def list_available():
        """Get list of available planner names"""
        return list(PlannerFactory.AVAILABLE_PLANNERS.keys())
