import pygame
from config import (
    CELL_SIZE, ROBOT_RADIUS, WHITE, BLACK, GRAY, ROBOT_COLORS
)


def save_frame_as_surface(sim):
    """Create a pygame surface for a single frame"""
    window_width = sim.width * CELL_SIZE
    window_height = sim.height * CELL_SIZE
    surface = pygame.Surface((window_width, window_height))

    # Clear surface
    surface.fill(WHITE)

    # Draw grid and obstacles
    for x in range(sim.width):
        for y in range(sim.height):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, BLACK, rect, 1)
            if sim.obstacles[x][y]:
                pygame.draw.rect(surface, GRAY, rect)

    # Draw robots and goals
    for i, (robot, goal) in enumerate(zip(sim.robots, sim.robotGoals)):
        # Draw robot
        robot_x = robot[0] * CELL_SIZE + CELL_SIZE // 2
        robot_y = robot[1] * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(surface, ROBOT_COLORS[i % len(ROBOT_COLORS)], (robot_x, robot_y), ROBOT_RADIUS)

        # Draw robot label
        font = pygame.font.Font(None, 24)
        label = font.render(sim.robotNames[i], True, BLACK)
        label_rect = label.get_rect(center=(robot_x, robot_y))
        surface.blit(label, label_rect)

        # Draw goal
        if goal is not None:
            goal_x = goal[0] * CELL_SIZE + CELL_SIZE // 2
            goal_y = goal[1] * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(surface, ROBOT_COLORS[i % len(ROBOT_COLORS)], (goal_x, goal_y),
                             ROBOT_RADIUS // 2, 2)

    return surface
