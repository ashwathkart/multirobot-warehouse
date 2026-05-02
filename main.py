import pygame
import argparse
import sys
import random

from config import FPS, DEFAULT_ROBOTS, DEFAULT_WIDTH, DEFAULT_HEIGHT, IO_POINTS_PER_SIDE
from simulation.world import RobotGridSimulation
from visualization.renderer import save_frame_as_surface
from visualization.frame_manager import FrameManager
from collision.resolver import exchange2x2, exchangeToMoves
from planning.planner_factory import PlannerFactory
from simulation.fms import FleetManagementSystem


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


def solve_robot_paths(sim, planner, frame_manager, fms=None):
    """Solve robot paths using the specified planner"""
    if not pygame.get_init():
        pygame.init()

    # If FMS provided, override robot goals with FMS assignments
    if fms:
        fms.initialize_tasks([sim.robots[i] for i in range(len(sim.robots))],
                            fms.sources, fms.destinations)
        for robot_id, (source, dest) in fms.get_all_tasks().items():
            sim.robotGoals[robot_id] = dest
        fms.print_tasks()

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
    parser.add_argument('--generate-map', action='store_true',
                        help='Generate a new map instead of loading')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH, help='Map width (for generation)')
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT, help='Map height (for generation)')
    parser.add_argument('--io-points', type=int, default=IO_POINTS_PER_SIDE,
                        help='I/O points per side (for generation)')
    parser.add_argument('--planner', type=str, default='astar',
                        choices=PlannerFactory.list_available(),
                        help='Path planning algorithm')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Generate or load map
    if args.generate_map:
        print(f"Generating {args.width}x{args.height} map with {args.io_points} I/O points per side...")
        grid, input_locs, output_locs = generate_warehouse_map(
            width=args.width,
            height=args.height,
            io_points_per_side=args.io_points
        )
        # Create robot starting positions (spread across center-left area)
        robot_positions = [
            (args.width // 4 + i % 2 * 2, args.height // 2 + (i // 2) * 2 - (args.robots // 2) * 2)
            for i in range(args.robots)
        ]
        save_map_file(args.map, grid, input_locs, output_locs, robot_positions)
        print(f"Generated and saved map to {args.map}")

    # Initialize simulation
    sim = RobotGridSimulation(args.robots, args.width, args.height)
    sim.loadMap(args.map)

    # Create FMS and extract I/O locations from map
    fms = FleetManagementSystem()
    # Find all numbered locations (1-9) as I/O points
    input_locs = []
    output_locs = []
    visited = set()
    for x in range(sim.width):
        for y in range(sim.height):
            # Check if this position was marked as a goal in the map
            for goal in sim.robotGoals:
                if goal and goal == (x, y) and goal not in visited:
                    output_locs.append(goal)
                    visited.add(goal)
    # For input locations, use the lower-numbered positions if available
    # For now, duplicate the output locations as inputs
    input_locs = output_locs[:len(output_locs)//2] if output_locs else [(1, sim.height//2)]
    output_locs = output_locs[len(output_locs)//2:] if len(output_locs) > 1 else output_locs
    if not input_locs:
        input_locs = [(1, sim.height // 2)]
    if not output_locs:
        output_locs = [(sim.width - 2, sim.height // 2)]

    fms.sources = input_locs
    fms.destinations = output_locs

    # Create planner
    planner = PlannerFactory.create(args.planner)

    # Create frame manager
    frame_manager = FrameManager(enable_rendering=not args.no_render)

    # Solve paths with FMS
    moves = solve_robot_paths(sim, planner, frame_manager, fms)

    # Save GIF
    if frame_manager.enable_rendering:
        frame_manager.save_gif('robot_simulation.gif')

    print(f"Solution found with {len(moves)} steps")
    print(f"Frames captured: {frame_manager.get_frame_count()}")


if __name__ == '__main__':
    main()
