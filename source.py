from matplotlib.patches import Rectangle,Circle
import matplotlib.pyplot as plt
import os
import math
import numpy as np

class RobotGridSimulation:
    def __init__(self,Nrobots:int,width:int,height:int):
        self.width = width
        self.height = height
        self.range = (width,height)
        self.robots = [(0,0) for i in range(Nrobots)]
        if Nrobots > 26:
            self.robotNames = ['R'+str(i) for i in range(Nrobots)]
        else:
            self.robotNames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:Nrobots]
        self.robotGoals = [None for i in range(Nrobots)]
        self.obstacles = [[False]*height for i in range(width)]
        self.obstacle_list = []
        self.reserve = np.zeros(Nrobots)
        self.giveway = np.zeros(Nrobots, dtype=bool)
        self.crashed = False
    def state(self):
        return [r for r in self.robots]
    def setState(self,s):
        self.robots = [r for r in s]
    def moveRobot(self,robot:int,d:str,check=True):
        if d=='l':
            self.moveRobotLeft(robot,check)
        elif d=='r':
            self.moveRobotRight(robot,check)
        elif d=='u':
            self.moveRobotUp(robot,check)
        elif d=='d':
            self.moveRobotDown(robot,check)
        else:
            raise ValueError("Invalid direction "+d+", must be one of uldr")
    def moveRobotLeft(self,robot:int,check=True):
        x,y = self.robots[robot]
        self.robots[robot] = (x-1,y)
        if check:
            if not self.valid(robot):
                print("Robot",robot,"crashed while moving left")
                self.crashed = True
                return False
        return True
    def moveRobotRight(self,robot:int,check=True):
        x,y = self.robots[robot]
        self.robots[robot] = (x+1,y)
        if check:
            if not self.valid(robot):
                print("Robot",robot,"crashed while moving right")
                self.crashed = True
                return False
        return True
    def moveRobotUp(self,robot:int,check=True):
        x,y = self.robots[robot]
        self.robots[robot] = (x,y+1)
        if check:
            if not self.valid(robot):
                print("Robot",robot,"crashed while moving up")
                self.crashed = True
                return False
        return True
    def moveRobotDown(self,robot:int,check=True):
        x,y = self.robots[robot]
        self.robots[robot] = (x,y-1)
        if check:
            if not self.valid(robot):
                print("Robot",robot,"crashed while moving down")
                self.crashed = True
                return False
        return True 
    def valid(self,robots='all'):
        if robots == 'all':
            robots = range(len(self.robots))
            fixedRobots = []
        elif not hasattr(robots,'__iter__'):
            fixedRobots = [i for i in range(len(self.robots)) if i != robots]
            robots = [robots]
        else:
            rset = set(robots)
            fixedRobots = [i for i in range(len(self.robots)) if i not in rset]
        rpos = set()
        for r in fixedRobots:
            rpos.add(self.robots[r])
        for r in robots:
            xy = self.robots[r]
            if xy[0] < 0 or xy[0] >= self.width:
                return False
            if xy[1] < 0 or xy[1] >= self.height:
                return False
            if self.obstacles[xy[0]][xy[1]]:
                return False
            if xy in rpos:
                return False
            rpos.add(xy)
        return True
    def plot(self,ax):
        import matplotlib.pyplot as plt
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for i in range(len(self.obstacles)):
            for j in range(len(self.obstacles[i])):
                if self.obstacles[i][j]:
                    ax.add_patch(Rectangle((i,j),1,1,fill=True,color=(0.5,0.5,0.5),zorder=0))
        for index,(r,g) in enumerate(zip(self.robots,self.robotGoals)):
            i,j = r
            ax.add_patch(Circle((i+0.5,j+0.5),0.4,fill=False,lw=2,color=colors[index],zorder=4))
            if g is not None:
                ax.add_patch(Circle((g[0]+0.5,g[1]+0.5),0.2,fill=False,lw=1,linestyle='--',color=colors[index],zorder=4))
        ax.plot([0,self.width,self.width,0,0],[0,0,self.height,self.height,0],lw=1,color='k')
    def plotPaths(self,path:list,ax):
        """path: a list of (robot,action) pairs, where action is one of 'l','r','u','d' (left,right,up,down)"""
        import matplotlib.pyplot as plt
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        mvs = [[] for r in self.robots]
        for (rob,mv) in path:
            mvs[rob].append(mv)
        paths = [[r] for r in self.robots]
        for i,actions in enumerate(mvs):
            for mv in actions:
                self.moveRobot(i,mv,check=False)
                paths[i].append(self.robots[i])
            if len(paths[i]) > 1:
                ax.plot([v[0]+0.5 for v in paths[i]],[v[1]+0.5 for v in paths[i]],lw=1,color=colors[i])
        #restore state
        self.robots = [path[0] for path in paths]
    def moveStr(self,moves:list) -> str:
        """Returns a string indicating a simultaneous move, in the format
        requested by the problem.

        moves: a list of (robot,action) pairs
        """
        s = ''
        for (rob,mv) in moves:
            s += self.robotNames[rob]+mv
        return s
    def loadMap(self,fn):
        """Loads a map from a .txt in the format specified in the problem"""
        with open(fn,'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].strip()
                if lines[i] == '':
                    lines = lines[:i]
                    break
            height = len(lines)
            if height == 0:
                raise ValueError("Empty map file")
            width = len(lines[0])
            for line in lines:
                if len(line) != width:
                    raise ValueError("Map file is not rectangular")
            self.width = width
            self.height = height
            self.crashed = False
            self.obstacles = [[False]*height for i in range(width)]
            self.robots = []
            self.robotGoals = []
            robots = {}
            goals = {}
            for i,line in enumerate(lines[::-1]):
                for j,c in enumerate(line):
                    if c=='.':
                        self.obstacles[j][i] = False
                    elif c=='#':
                        self.obstacles[j][i] = True
                        self.obstacle_list.append((j,i))
                    elif c in 'ABCDEFGHI':
                        robots[c] = (j,i)
                    elif c in '123456789':
                        goals[c] = (j,i)
                    else:
                        raise ValueError("Invalid character "+c+" in map file")
            for c,g in zip('ABCDEFGHI','123456789'):
                if c in robots:
                    self.robots.append(robots[c])
                    if g in goals:
                        self.robotGoals.append(goals[g])
                    else:
                        print("Warning, robot",c,"has no goal")
                        self.robotGoals.append(None)
                else:
                    break
        print("Read map of width x height ({},{}) with {} robots".format(width,height,len(self.robots)))

def exchange2x2(a0,b0,adest,bdest):
    """Given a and b within a 2x2 cell with no other obstacles, returns
    a sequence of moves that will get a from position a0 to adest and b
    from position b0 to bdest.
    
    The sequence is a list in the form ['au','bl*','ar'] where the first
    character is the object to move, and the second character is the direction
    (u: up, d: down, r: right, l: left)
    If the third character * is given, then this means it can be done at the same time as the 
    previous action
    """
    assert a0 != b0,"Objects have the same start?"
    assert adest != bdest,"Objects have the same destination?"
    xlow = min(a0[0],b0[0],adest[0],bdest[0])
    assert max(a0[0],b0[0],adest[0],bdest[0]) <= xlow+1,"Objects aren't in a 2x2 box?"
    ylow = min(a0[1],b0[1],adest[1],bdest[1])
    assert max(a0[1],b0[1],adest[1],bdest[1]) <= ylow+1,"Objects aren't in a 2x2 box?"
    a0 = (a0[0]-xlow,a0[1]-ylow)
    b0 = (b0[0]-xlow,b0[1]-ylow)
    adest = (adest[0]-xlow,adest[1]-ylow)
    bdest = (bdest[0]-xlow,bdest[1]-ylow)
    path = []
    while a0 != adest or b0 != bdest:
        #print a0,b0
        aopts = []
        aacts = []
        bopts = []
        bacts = []
        if adest[0] > a0[0]:
            aopts.append((1,a0[1]))
            aacts.append('r')
        elif adest[0] < a0[0]:
            aopts.append((0,a0[1]))
            aacts.append('l')
        if adest[1] > a0[1]:
            aopts.append((a0[0],1))
            aacts.append('u')
        elif adest[1] < a0[1]:
            aopts.append((a0[0],0))
            aacts.append('d')
        if bdest[0] > b0[0]:
            bopts.append((1,b0[1]))
            bacts.append('r')
        elif bdest[0] < b0[0]:
            bopts.append((0,b0[1]))
            bacts.append('l')
        if bdest[1] > b0[1]:
            bopts.append((b0[0],1))
            bacts.append('u')
        elif bdest[1] < b0[1]:
            bopts.append((b0[0],0))
            bacts.append('d')
        #print "A options",aopts,"B options",bopts
        moved = False
        for i in range(len(aopts)):
            if aopts[i] != b0:
                path.append('a'+aacts[i])
                a0 = aopts[i]
                moved = True
                break
        for i in range(len(bopts)):
            if bopts[i] != a0:
                path.append('b'+bacts[i])
                if moved:
                    path[-1] = path[-1] + '*'
                b0 = bopts[i]
                moved = True
                break
        if not moved:
            if a0 == adest and b0 == bdest:
                break
            else:
                #might need to move the other out of the way
                if a0[0] == b0[0]:
                    if b0[0] == 0:
                        b0 = (1,b0[1])
                        path.append('br')
                    else:
                        b0 = (0,b0[1])
                        path.append('bl')
                else:
                    assert a0[1] == b0[1]
                    if b0[1] == 0:
                        b0 = (b0[0],1)
                        path.append('bu')
                    else:
                        b0 = (b0[0],0)
                        path.append('bd')
                #print "Moving B out of the way to",b0
    return path

def exchangeToMoves(soln:list, robots:list, simultaneous=True) -> list:
    """From an exchange2x2 solution and a list of robot integers, returns
    a list of moves.  If simultaneous=True, these are simultaneous moves
    in the form [moves1,moves2,...] with each element
    movesN = [(robot1,move1),...,(robotk,movek)] giving the moves performed
    simultaneously."""
    if simultaneous:
        moves = []
        lastmove = []
        for mv in soln:
            if mv[-1]!='*' and len(lastmove) > 0:
                moves.append(lastmove)
                lastmove = []
            lastmove.append([robots[0] if mv[0]=='a' else robots[1],mv[1]])
        moves.append(lastmove)
    else:
        moves = []
        for mv in soln:
            moves.append([robots[0] if mv[0]=='a' else robots[1],mv[1]])
    return moves

def is_move_valid(next_position):
    valid_move = True
    if not (0 <= next_position[0] < sim.range[0] and 0 <= next_position[1] < sim.range[1]):
        valid_move = False
    if next_position in sim.obstacle_list:
        valid_move = False
    return valid_move    

def give_way_maneuver(current_position, robot_index):
    if sim.giveway[robot_index]:  # Already trying to give way
        return False
        
    # Try to find a position that's not in any robot's path
    side_positions = [
        (current_position[0], current_position[1]+1),
        (current_position[0], current_position[1]-1),
        (current_position[0]+1, current_position[1]),
        (current_position[0]-1, current_position[1])
    ]
    
    # Sort positions by how many other robots' paths they intersect
    def count_path_intersections(pos):
        count = 0
        for path in all_paths:
            if pos in path:
                count += 1
        return count
    
    side_positions.sort(key=count_path_intersections)
    
    for temp_position in side_positions:
        if (is_move_valid(temp_position) and 
            temp_position not in [r for r in sim.robots]):
            dx = temp_position[0] - current_position[0]
            dy = temp_position[1] - current_position[1]
            
            moved = False
            if dx == 1:
                moved = sim.moveRobotRight(robot_index)
            elif dx == -1:
                moved = sim.moveRobotLeft(robot_index)
            elif dy == 1:
                moved = sim.moveRobotUp(robot_index)
            elif dy == -1:
                moved = sim.moveRobotDown(robot_index)
                
            if moved:
                sim.giveway[robot_index] = True
                return True
                
    return False

def golden_path(all_paths):
    current_steps = [0 for _ in all_paths]
    
    while any(step < len(path) - 1 for step, path in zip(current_steps, all_paths)):
        # First, check which robots can move without conflicts
        moves_to_make = []
        for robot_index, path in enumerate(all_paths):
            if current_steps[robot_index] >= len(path) - 1:
                continue
                
            current_position = path[current_steps[robot_index]]
            next_position = path[current_steps[robot_index] + 1]
            
            # Check if move is valid and destination is not occupied by another robot
            if (is_move_valid(next_position) and 
                next_position not in [sim.robots[i] for i in range(len(sim.robots))]):
                moves_to_make.append((robot_index, current_position, next_position))
        
        # If no moves are possible, try to resolve deadlocks
        if not moves_to_make:
            for robot_index, path in enumerate(all_paths):
                if current_steps[robot_index] >= len(path) - 1:
                    continue
                if not sim.giveway[robot_index]:
                    give_way_maneuver(sim.robots[robot_index], robot_index)
        
        # Execute valid moves
        for robot_index, current_position, next_position in moves_to_make:
            dx = next_position[0] - current_position[0]
            dy = next_position[1] - current_position[1]
            
            if dx == 1:
                sim.moveRobotRight(robot_index)
            elif dx == -1:
                sim.moveRobotLeft(robot_index)
            elif dy == 1:
                sim.moveRobotUp(robot_index)
            elif dy == -1:
                sim.moveRobotDown(robot_index)
            
            current_steps[robot_index] += 1
            sim.giveway[robot_index] = False  # Reset giveway flag after successful move
        
        show_figure()

def show_figure():
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.axis('equal')
    sim.plot(ax)
    plt.show()

#load the data file and get the map
sim = RobotGridSimulation(6,11,6)
# sim.loadMap("D:\Avaritia\MENG Autonomy\Assignment\KKH\KKH HW\Final\Prompt2\map_test.txt")
sim.loadMap("map1.txt")

show_figure()

# Define constants
VELOCITY = 5
# Define obstacles and bounds
OBSTACLES = sim.obstacle_list
BOUNDS = (0, 0, sim.width, sim.height)

start_positions = (sim.robots)  # Starting coordinates
goal_positions = (sim.robotGoals)  # Goal coordinates

# Node representation
class Node:
    def __init__(self, position, g_cost=0, parent=None):
        self.position = position
        self.g_cost = g_cost  # Total cost from start node to this node
        self.parent = parent  # Parent node in path

def is_position_valid(position):
    """Check if the position is within bounds and not an obstacle."""
    x, y = position
    if x < BOUNDS[0] or x > BOUNDS[2] or y < BOUNDS[1] or y > BOUNDS[3]:
        return False  # Out of bounds
    if position in OBSTACLES:
        return False  # Position is an obstacle
    return True

def get_successors(node):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    movement_cost = 1
    
    successors = []
    for d in directions:
        next_position = (node.position[0] + d[0], node.position[1] + d[1])
        if is_position_valid(next_position):
            next_node = Node(next_position, node.g_cost + movement_cost, node)
            successors.append(next_node)
    return successors

def dijkstra_search(start, goal):
    open_set = set()
    closed_set = set()
    start_node = Node(start)
    goal_node = Node(goal)
    
    open_set.add(start_node)
    
    while open_set:
        # Select node with minimum g_cost in open set
        current_node = min(open_set, key=lambda n: n.g_cost)

        if current_node.position == goal_node.position:
            return reconstruct_path(current_node)
        
        open_set.remove(current_node)
        closed_set.add(current_node)
        
        for successor in get_successors(current_node):
            if successor in closed_set:
                continue
            # Improved handling for open_set updates
            in_open_set = False
            for open_node in open_set:
                if open_node.position == successor.position:
                    in_open_set = True
                    if successor.g_cost < open_node.g_cost:
                        open_set.remove(open_node)
                        open_set.add(successor)
                    break
            if not in_open_set:
                open_set.add(successor)

    return []  # Return empty path if goal not found

# Reconstruct path from goal to start
def reconstruct_path(node):
    path = []
    while node:
        path.append(node.position)
        node = node.parent
    return path[::-1]  # Return reversed path

def find_paths_for_all_robots(start_positions, goal_positions):
    all_paths = []
    for start_position, goal_position in zip(start_positions, goal_positions):
        path = dijkstra_search(start_position, goal_position)
        all_paths.append(path)
    return all_paths

all_paths = find_paths_for_all_robots(start_positions, goal_positions)

golden_path(all_paths)