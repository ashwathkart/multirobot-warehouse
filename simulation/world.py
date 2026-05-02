import numpy as np

class RobotGridSimulation:
    def __init__(self,Nrobots:int,width:int,height:int):
        self.width = width
        self.height = height
        self.range = (width, height)
        self.robots = [(0, 0) for i in range(Nrobots)]
        if Nrobots > 26:
            self.robotNames = ['R' + str(i) for i in range(Nrobots)]
        else:
            self.robotNames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:Nrobots]
        self.robotGoals = [None for i in range(Nrobots)]
        self.obstacles = [[False] * height for i in range(width)]
        self.obstacle_list = []
        self.reserve = np.zeros(Nrobots)
        self.giveway = np.zeros(Nrobots, dtype=bool)
        self.crashed = False

    def state(self):
        return [r for r in self.robots]

    def setState(self, s):
        self.robots = [r for r in s]

    def moveRobot(self, robot: int, d: str, check=True):
        if d == 'l':
            self.moveRobotLeft(robot, check)
        elif d == 'r':
            self.moveRobotRight(robot, check)
        elif d == 'u':
            self.moveRobotUp(robot, check)
        elif d == 'd':
            self.moveRobotDown(robot, check)
        else:
            raise ValueError("Invalid direction " + d + ", must be one of uldr")

    def moveRobotLeft(self, robot: int, check=True):
        x, y = self.robots[robot]
        self.robots[robot] = (x - 1, y)
        if check:
            if not self.valid(robot):
                print("Robot", robot, "crashed while moving left")
                self.crashed = True
                return False
        return True

    def moveRobotRight(self, robot: int, check=True):
        x, y = self.robots[robot]
        self.robots[robot] = (x + 1, y)
        if check:
            if not self.valid(robot):
                print("Robot", robot, "crashed while moving right")
                self.crashed = True
                return False
        return True

    def moveRobotUp(self, robot: int, check=True):
        x, y = self.robots[robot]
        self.robots[robot] = (x, y + 1)
        if check:
            if not self.valid(robot):
                print("Robot", robot, "crashed while moving up")
                self.crashed = True
                return False
        return True

    def moveRobotDown(self, robot: int, check=True):
        x, y = self.robots[robot]
        self.robots[robot] = (x, y - 1)
        if check:
            if not self.valid(robot):
                print("Robot", robot, "crashed while moving down")
                self.crashed = True
                return False
        return True

    def valid(self, robots='all'):
        if robots == 'all':
            robots = range(len(self.robots))
            fixedRobots = []
        elif not hasattr(robots, '__iter__'):
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

    def moveStr(self, moves: list) -> str:
        s = ''
        for (rob, mv) in moves:
            s += self.robotNames[rob] + mv
        return s

    def loadMap(self, fn):
        with open(fn, 'r') as f:
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
            self.obstacles = [[False] * height for i in range(width)]
            self.robots = []
            self.robotGoals = []
            robots = {}
            goals = {}
            for i, line in enumerate(lines[::-1]):
                for j, c in enumerate(line):
                    if c == '.':
                        self.obstacles[j][i] = False
                    elif c == '#':
                        self.obstacles[j][i] = True
                        self.obstacle_list.append((j, i))
                    elif c in 'ABCDEFGHI':
                        robots[c] = (j, i)
                    elif c in '123456789':
                        goals[c] = (j, i)
                    else:
                        raise ValueError("Invalid character " + c + " in map file")
            for c, g in zip('ABCDEFGHI', '123456789'):
                if c in robots:
                    self.robots.append(robots[c])
                    if g in goals:
                        self.robotGoals.append(goals[g])
                    else:
                        print("Warning, robot", c, "has no goal")
                        self.robotGoals.append(None)
                else:
                    break
        # Update robotNames to match actual number of robots loaded
        Nrobots = len(self.robots)
        if Nrobots > 26:
            self.robotNames = ['R' + str(i) for i in range(Nrobots)]
        else:
            self.robotNames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:Nrobots]
        print("Read map of width x height ({},{}) with {} robots".format(
            width, height, len(self.robots)))
