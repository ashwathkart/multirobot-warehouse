# Simulation configuration constants

CELL_SIZE = 50  # Size of each grid cell in pixels
ROBOT_RADIUS = 20
FPS = 2  # Frames per second for animation

# Visualization colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
ROBOT_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
]

# Grid dimensions
DEFAULT_WIDTH = 100
DEFAULT_HEIGHT = 20
DEFAULT_ROBOTS = 5
IO_POINTS_PER_SIDE = 10

# Simulation defaults
DEFAULT_MAX_TIME = 300  # seconds
DEFAULT_PLANNER = 'astar'
OBSTACLE_SPAWN_RATE = 0.05
OBSTACLE_DURATION = 30
