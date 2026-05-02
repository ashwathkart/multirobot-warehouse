import pygame
import argparse
import sys

from config import FPS, DEFAULT_ROBOTS
from simulation.world import RobotGridSimulation
from visualization.renderer import save_frame_as_surface
from visualization.frame_manager import FrameManager
from collision.resolver import exchange2x2, exchangeToMoves
from planning.planner_factory import PlannerFactory


def get_move_direction(current_pos, next_pos):
    """Convert position delta to move direction"""
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]
    if dx == 1: return 'r'
    if dx == -1: return 'l'
    if dy == 1: return 'u'
    if dy == -1: return 'd'
    return None


def manhattan_distance(p1, p2):
    """Calculate Manhattan distance"""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def solve_robot_paths(sim, planner, frame_manager):
    """Solve robot paths using the specified planner"""
    if not pygame.get_init():
        pygame.init()

    # Set up pygame window
    window_width = sim.width * 50  # CELL_SIZE from config
    window_height = sim.height * 50
    screen = None
    clock = None

    if frame_manager.enable_rendering:
        screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Robot Path Planning")
        clock = pygame.time.Clock()

    unplaced_robots = list(range(len(sim.robots)))
    moves_sequence = []

    # Capture initial state
    surface = save_frame_as_surface(sim)
    frame_manager.capture_frame(surface)
    if frame_manager.enable_rendering:
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(FPS)

    while unplaced_robots:
        # Try to move each unplaced robot
        for robot_id in unplaced_robots[:]:
            current_pos = sim.robots[robot_id]
            goal_pos = sim.robotGoals[robot_id]

            if current_pos == goal_pos:
                unplaced_robots.remove(robot_id)
                continue

            # Get occupied positions except current robot
            occupied = set(sim.robots[i] for i in range(len(sim.robots)) if i != robot_id)

            # Find path to goal
            path = planner.plan(current_pos, goal_pos, sim, occupied)

            if path and len(path) > 1:
                # Move one step along the path
                next_pos = path[1]

                # Check if next position is occupied by another robot
                blocking_robot = None
                for other_robot in range(len(sim.robots)):
                    if sim.robots[other_robot] == next_pos:
                        blocking_robot = other_robot
                        break

                if blocking_robot is not None:
                    # Use exchange algorithm if robots are adjacent
                    if manhattan_distance(current_pos, next_pos) == 1:
                        exchange_moves = exchange2x2(
                            current_pos, sim.robots[blocking_robot],
                            next_pos, current_pos
                        )
                        robot_moves = exchangeToMoves(exchange_moves, [robot_id, blocking_robot])
                        moves_sequence.extend(robot_moves)

                        # Update positions after exchange
                        sim.robots[robot_id], sim.robots[blocking_robot] = (
                            sim.robots[blocking_robot], sim.robots[robot_id]
                        )
                else:
                    # Make the move
                    direction = get_move_direction(current_pos, next_pos)
                    moves_sequence.append([(robot_id, direction)])
                    sim.robots[robot_id] = next_pos

        # Update display and capture frame
        surface = save_frame_as_surface(sim)
        frame_manager.capture_frame(surface)
        if frame_manager.enable_rendering:
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(FPS)

            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return moves_sequence

    # Keep window open until user closes it
    if frame_manager.enable_rendering:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return moves_sequence
            clock.tick(FPS)

    return moves_sequence


def main():
    parser = argparse.ArgumentParser(description='Multi-Robot Warehouse Simulation')
    parser.add_argument('--robots', type=int, default=DEFAULT_ROBOTS, help='Number of robots')
    parser.add_argument('--map', type=str, default='map1.txt', help='Path to map file')
    parser.add_argument('--planner', type=str, default='astar',
                        choices=PlannerFactory.list_available(),
                        help='Path planning algorithm')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')

    args = parser.parse_args()

    # Initialize simulation
    sim = RobotGridSimulation(args.robots, 11, 6)
    sim.loadMap(args.map)

    # Create planner
    planner = PlannerFactory.create(args.planner)

    # Create frame manager
    frame_manager = FrameManager(enable_rendering=not args.no_render)

    # Solve paths
    moves = solve_robot_paths(sim, planner, frame_manager)

    # Save GIF
    if frame_manager.enable_rendering:
        frame_manager.save_gif('robot_simulation.gif')

    print(f"Solution found with {len(moves)} steps")
    print(f"Frames captured: {frame_manager.get_frame_count()}")


if __name__ == '__main__':
    main()
