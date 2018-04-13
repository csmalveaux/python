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
    next_pos = []
    dest_pos = []

    movements = []
    sequence = []
    seq_p = 0

    def __init__(self, code):
        self.code = code
        self.start_pos = list(map(decodeDimension, code))
        self.curr_pos = self.start_pos
        self.movements = [getMoves(convertToPos(10, self.code[0])), getMoves(
            convertToPos(10, self.code[1])), getMoves(convertToPos(10, self.code[2]))]
        self.seq_p = 0

    def print(self):
        print("Pos: {0}".format(self.start_pos))
        print("\tCode: {0}".format(self.code))

    def generate_seq(self):
        self.sequence.append(self.start_pos)
        count = 0
        for col in range(3):
            for row in range(3):
                self.sequence.reverse()
                pos = self.sequence[0]
                pos[row] += self.movements[row][col]
                self.sequence.reverse()
                self.sequence.append(pos)
        self.dest_pos = self.sequence[0]

    def nextmove(self):
        if self.dest_pos == self.curr_pos:
            if self.seq_p != len(self.sequence):
                self.seq_p = + 1
            else:
                self.seq_p = 0
            self.dest_pos = self.sequence[self.seq_p]


class Cube:
    Size = 0
    Cells = {}
    Curr_Cells = {}
    Perm = {}
    Direction = []

    def __init__(self, size):
        self.Size = size
        self.Perm = dict.fromkeys(range(size + 1))
        for x in self.Perm.keys():
            self.Perm[x] = findCominations(
                x, list(itertools.product(range(10), repeat=3)))

        self.Cells = dict.fromkeys(range(0, size**3))
        for x in self.Cells.keys():
            coor = convertToPos(size, x)
            secure_random = random.SystemRandom()
            xperm = secure_random.choice(
                self.Perm[coor[0]])
            yperm = secure_random.choice(
                self.Perm[coor[1]])
            zperm = secure_random.choice(
                self.Perm[coor[2]])
            self.Cells[x] = Cell(
                list(map(convertPermutation, [xperm, yperm, zperm])))
        self.Curr_Cells = dict.fromkeys(range(0, size**3))
        for x in self.Curr_Cells.keys():
            self.Curr_Cells[x] = x
            self.Cells[x].print()
            self.Cells[x].generate_seq()

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

cube = Cube(26)
