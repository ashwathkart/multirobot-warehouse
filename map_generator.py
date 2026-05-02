import random


def generate_warehouse_map(width=100, height=20, io_points_per_side=10):
    """
    Generate a warehouse map with I/O points on sides.

    Args:
        width: Map width in cells
        height: Map height in cells
        io_points_per_side: Number of I/O points on each side (left/right), max 9

    Returns:
        (grid, input_locations, output_locations) tuple where:
        - grid: 2D list where 1 = obstacle, 0 = free
        - input_locations: List of (x, y) tuples for input points
        - output_locations: List of (x, y) tuples for output points
    """
    # Limit to 9 points per side (digits 1-9)
    io_points_per_side = min(io_points_per_side, 9)

    # Create empty grid (0 = free, 1 = obstacle)
    grid = [[0 for _ in range(width)] for _ in range(height)]

    # Add some obstacles (simple pattern for now)
    # Create vertical obstacle columns with gaps
    for x in range(20, width - 20, 15):
        for y in range(height):
            if y != height // 2 and y != height // 2 - 1:
                grid[y][x] = 1

    # Create I/O points on left and right sides
    input_locations = []
    output_locations = []

    # Left side: y-coordinates spaced evenly
    left_y_positions = [
        int(i * (height - 1) / (io_points_per_side - 1))
        for i in range(io_points_per_side)
    ]

    # Right side: y-coordinates spaced evenly
    right_y_positions = [
        int(i * (height - 1) / (io_points_per_side - 1))
        for i in range(io_points_per_side)
    ]

    # Randomly assign left side points as input or output
    for y in left_y_positions:
        if random.random() < 0.5:
            input_locations.append((1, y))
        else:
            output_locations.append((1, y))

    # Randomly assign right side points as input or output
    for y in right_y_positions:
        if random.random() < 0.5:
            input_locations.append((width - 2, y))
        else:
            output_locations.append((width - 2, y))

    # Ensure we have at least some inputs and outputs
    if len(input_locations) == 0:
        input_locations.append((1, height // 2))
        if (width - 2, height // 2) in output_locations:
            output_locations.remove((width - 2, height // 2))
    if len(output_locations) == 0:
        output_locations.append((width - 2, height // 2))
        if (1, height // 2) in input_locations:
            input_locations.remove((1, height // 2))

    return grid, input_locations, output_locations


def grid_to_map_string(grid, input_locations, output_locations, robots=None):
    """
    Convert grid to map file format string.

    Args:
        grid: 2D list where 1 = obstacle, 0 = free
        input_locations: List of (x, y) tuples for input points
        output_locations: List of (x, y) tuples for output points
        robots: List of (x, y) tuples for robot starting positions (optional)

    Returns:
        String in map file format
    """
    if robots is None:
        robots = []

    height = len(grid)
    width = len(grid[0])

    # Create map character grid
    map_grid = [['.' for _ in range(width)] for _ in range(height)]

    # Add obstacles
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 1:
                map_grid[y][x] = '#'

    # Add input/output points (using 1-9, max 9 per side)
    input_idx = 1
    output_idx = 1
    for x, y in input_locations:
        if 0 <= x < width and 0 <= y < height and input_idx <= 9:
            map_grid[y][x] = str(input_idx)
            input_idx += 1
    for x, y in output_locations:
        if 0 <= x < width and 0 <= y < height and output_idx <= 9:
            map_grid[y][x] = str(output_idx)
            output_idx += 1

    # Add robots
    robot_chars = 'ABCDEFGHI'
    for idx, (x, y) in enumerate(robots):
        if idx < len(robot_chars) and 0 <= x < width and 0 <= y < height:
            map_grid[y][x] = robot_chars[idx]

    # Convert to string (reverse for proper orientation)
    map_lines = []
    for row in reversed(map_grid):
        map_lines.append(''.join(row))

    return '\n'.join(map_lines)


def save_map_file(filename, grid, input_locations, output_locations, robots=None):
    """Save map to file"""
    map_string = grid_to_map_string(grid, input_locations, output_locations, robots)
    with open(filename, 'w') as f:
        f.write(map_string)
    print(f"Saved map to {filename}")
