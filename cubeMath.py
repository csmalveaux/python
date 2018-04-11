import itertools

import random


def convertToCell(base, pos):
    return pos[0] * (base ** 2) + pos[1] * base + pos[2]


def convertToPos(base, cellNumber):
    xPos = cellNumber // (base ** 2)
    yPos = (cellNumber % (base ** 2)) // base
    zPos = cellNumber % base

    return [xPos, yPos, zPos]


def findCominations(number, permutations):
    returnList = []
    for perm in permutations:
        if(perm[0] + perm[1] + perm[2] == number):
            returnList.append(perm)
    return returnList


def convertPermutation(permutation):
    return permutation[0] * 100 + permutation[1] * 10 + permutation[2]


def getPower(n):
    return lambda x: n ** x


def getMoves(p):
    return [p[0] - p[1], p[1] - p[2], p[2] - p[0]]


def decodeDimension(dim):
    a = dim // 100
    b = (dim % 100) // 10
    c = dim % 10
    return (a + b + c)


class Cell:
    code = []

    start_pos = []
    curr_pos = []
    # next_pos = []
    # dest_pos = []

    available_mv = []
    history_mv = []

    def __init__(self, code):
        self.code = code
        self.start_pos = list(map(decodeDimension, code))
        self.curr_pos = self.start_pos
        self.available_mv = ["x"] * 3 + ["y"] * 3 + ["z"] * 3

    # def getMoves(dim):
    #    return [dim[0] - dim[1], dim[1] - dim[2], dim[2] - dim[0]]

    def print(self):
        print("\tPos: {0}".format(self.start_pos))
        print("\tCode: {0}".format(self.code))
    # def completedCycle(self):
    #    if(self.start_pos == self.curr_pos):
    #        self.cycles += 1
    #        return True
    #    return False

    def isavaliable(self, mv):
        return (mv in self.available_mv)

    def move(self, mv):
        self.history_mv.append(mv)
        self.available_mv.remove(mv)

    def randDirection(self):
        secure_random = random.SystemRandom()
        return secure_random.choice(self.available_mv)


class Cube:
    Size = 0
    Cells = {}
    Curr_Cells = {}
    Perm = {}
    Direction = []

    def __init__(self, size):
        self.Size = size
        self.Perm = dict.fromkeys(range(1, size + 1))
        for x in self.Perm.keys():
            self.Perm[x] = findCominations(
                x, list(itertools.product(range(10), repeat=3)))

        self.Cells = dict.fromkeys(range(0, size**3))
        for x in self.Cells.keys():
            coor = convertToPos(size, x)
            secure_random = random.SystemRandom()
            xperm = secure_random.choice(
                self.Perm[coor[0] + 1])
            yperm = secure_random.choice(
                self.Perm[coor[1] + 1])
            zperm = secure_random.choice(
                self.Perm[coor[2] + 1])
            self.Cells[x] = Cell(
                list(map(convertPermutation, [xperm, yperm, zperm])))
        self.Curr_Cells = dict.fromkeys(range(0, size**3))
        for x in self.Curr_Cells.keys():
            self.Curr_Cells[x] = x

    def adjCells(self, x):
        pos = convertToPos(self.Size, x)

        top = convertToCell(self.Size, list(pos[:2]) + [pos[2] - 1])
        bottom = convertToCell(self.Size, list(pos[:2]) + [pos[2] + 1])
        north = convertToCell(self.Size, [pos[0], pos[1] - 1, pos[2]])
        south = convertToCell(self.Size, [pos[0], pos[1] + 1, pos[2]])
        east = convertToCell(self.Size, [pos[0] - 1] + pos[1:])
        west = convertToCell(self.Size, [pos[0] + 1] + pos[1:])
        adjCells = [top, bottom, north, south, east, west]
        return filter(lambda x: x >= 0 and x < self.Size ** 3, adjCells)

    def initMoves(self):
        directions = dict.fromkeys(range(0, self.Size**3))
        for x in directions.keys():
            mv = self.Cells[x].randDirection()
            self.Cells[x].move(mv)
            directions[x] = mv
            print(list(self.adjCells(x)))
        self.Direction.append(directions)


primeList = list(range(2, 1000))
for x in primeList:
    for y in range(2 * x, 1000, x):
        if(y in primeList):
            primeList.remove(y)

primePowers = list()
for x in primeList:
    powLamda = getPower(x)
    primePower = powLamda(2)
    count = 2
    while(primePower < 1000):
        primePower = powLamda(count)
        count = count + 1
        if(primePower is not primePowers and primePower < 1000):
            primePowers.append(primePower)

trapRooms = list(primeList)
trapRooms.extend(primePowers)
trapRooms.sort()

cube = Cube(5)
cube.initMoves()
