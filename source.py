import os
import math
import numpy as np
import pygame
import time
import heapq

CELL_SIZE = 50  # Size of each grid cell in pixels
ROBOT_RADIUS = 20
FPS = 2  # Frames per second for animation

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

def show_pygame_window():
    """Initialize and show the Pygame window"""
    # Initialize pygame if not already initialized
    if not pygame.get_init():
        pygame.init()
    
    window_width = sim.width * CELL_SIZE
    window_height = sim.height * CELL_SIZE
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Robot Grid Simulation")
    
    # Define colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    # Handle any pending events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return
    
    # Clear screen
    screen.fill(WHITE)
    
    # Draw grid
    for x in range(sim.width):
        for y in range(sim.height):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            
            # Draw obstacles
            if sim.obstacles[x][y]:
                pygame.draw.rect(screen, GRAY, rect)
    
    # Draw robots and their goals
    all_robots_at_goals = True
    for i, (robot, goal) in enumerate(zip(sim.robots, sim.robotGoals)):
        # Draw robot
        robot_x = robot[0] * CELL_SIZE + CELL_SIZE // 2
        robot_y = robot[1] * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, COLORS[i % len(COLORS)], (robot_x, robot_y), ROBOT_RADIUS)
        
        # Draw robot label
        font = pygame.font.Font(None, 24)
        label = font.render(sim.robotNames[i], True, BLACK)
        label_rect = label.get_rect(center=(robot_x, robot_y))
        screen.blit(label, label_rect)
        
        # Draw goal
        if goal is not None:
            goal_x = goal[0] * CELL_SIZE + CELL_SIZE // 2
            goal_y = goal[1] * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(screen, COLORS[i % len(COLORS)], (goal_x, goal_y), 
                             ROBOT_RADIUS//2, 2)  # Draw as outline
            
            # Check if robot is at its goal
            if robot != goal:
                all_robots_at_goals = False
    
    pygame.display.flip()
    
    # If all robots have reached their goals, wait briefly and close
    if all_robots_at_goals:
        time.sleep(2)  # Show final state for 2 seconds
        pygame.quit()

#load the data file and get the map
sim = RobotGridSimulation(6,11,6)
sim.loadMap("map1.txt")

show_pygame_window()

# Define constants
VELOCITY = 5
# Define obstacles and bounds
OBSTACLES = sim.obstacle_list
BOUNDS = (0, 0, sim.width, sim.height)

start_positions = (sim.robots)  # Starting coordinates
goal_positions = (sim.robotGoals)  # Goal coordinates