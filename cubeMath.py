import itertools
import random
import numpy
import sys
import os


def convertToCell(base, pos):
    return pos[0] * (base ** 2) + pos[1] * base + pos[2]


def convertToPos(base, cellNumber):
    xPos = cellNumber // (base ** 2)
    yPos = (cellNumber % (base ** 2)) // base
    zPos = cellNumber % base

    return numpy.array([xPos, yPos, zPos])


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
    return numpy.array([p[0] - p[1], p[1] - p[2], p[2] - p[0]])


def decodeDimension(dim):
    a = dim // 100
    b = (dim % 100) // 10
    c = dim % 10
    return (a + b + c)


def determineCode(permutations, size, coor):
    loc = coor.copy() + 1
    perm = permutations.copy()
    secure_random = random.SystemRandom()
    pos = loc
    while len(perm) > 0:
        code = secure_random.choice(perm[loc])
        moves = getMoves(code)
        usable = True
        pos = loc
        for x in moves:
            pos += x
            if pos < 0 or pos >= (size + 2):
                perm[loc].remove(code)
                usable = False
                break

        if usable:
            return code


def selectPermutation(permutations, size, position):
    pos = position.copy() + 1
    perm = permutations[pos]
    secure_random = random.SystemRandom()
    loc = pos
    selection = list(range(len(perm)))
    while len(selection) > 0:
        sel = secure_random.randint(0, len(selection) - 1)
        code = perm[sel]
        moves = getMoves(code)
        usable = True
        loc = pos
        for x in moves:
            loc += x
            if loc < 0 or loc >= (size + 2):
                selection.remove(sel)
                usable = False
                break

        if usable:
            return sel


def generatePermutationDict(size):
    permutations = dict.fromkeys(range(1, size + 1))
    for x in permutations.keys():
        permutations[x] = findCominations(
            x, list(itertools.product(range(10), repeat=3)))
    return permutations

class Cell:
    code = []

    start_pos = []
    curr_pos = []
    next_pos = []
    dest_pos = []

    movements = []
    sequence = []
    seq_p = 0
    blocked = 0
    blockedby = -1

    def __init__(self, code):
        self.code = code
        self.start_pos = numpy.array(list(map(decodeDimension, code)))
        self.curr_pos = self.start_pos
        x_moves = getMoves(convertToPos(10, self.code[0]))
        y_moves = getMoves(convertToPos(10, self.code[1]))
        z_moves = getMoves(convertToPos(10, self.code[2]))
        self.sequence = []
        self.movements = numpy.array([x_moves, y_moves, z_moves])
        self.seq_p = 0
        self.blocked = 0
        self.blockedby = -1

    def print(self):
        print("Pos: {0}".format(self.start_pos))
        print("Code: {0}".format(self.code))
        print("Curr: {0}".format(self.curr_pos))
        print("Move: {0}".format(self.next_pos))
        print("Dest: {0}".format(self.dest_pos))
        print("seqp: {0}".format(self.seq_p))
        print("Seq: {0}".format(self.sequence))
        print("Blocked: {0}".format(self.blocked))
        print("Blocked by: {0}".format(self.blockedby))

    def generate_seq(self):
        sequence = []
        sequence.append(self.start_pos[:])
        for col in range(3):
            for row in range(3):
                pos = sequence[len(sequence) - 1].copy()
                pos[row] += self.movements[row, col]
                sequence.append(pos[:])
        self.dest_pos = sequence[0]
        self.sequence = sequence

    def nextmove(self):
        if numpy.array_equal(self.dest_pos, self.curr_pos):
            if self.seq_p < len(self.sequence) - 1:
                self.seq_p += 1
            else:
                self.seq_p = 0
        try:
            self.dest_pos = self.sequence[self.seq_p]
        except:
            print(self.sequence)
            print(self.seq_p)
            sys.exit(1)
        delta = numpy.subtract(self.dest_pos, self.curr_pos)
        if(numpy.sum(delta) != 0):
            self.next_pos = numpy.add(self.curr_pos,
                                      numpy.floor_divide(delta,
                                                         numpy.abs(numpy.sum(delta))))
            return self.next_pos
        return self.curr_pos

class Seed:
    gene = []

    def __init__(self, size, permutations):
        self.gene = []
        for x in range(size ** 3):
            coordinates = convertToPos(size, x)
            self.gene.append(selectPermutation(
                permutations, size, coordinates[0]))
            self.gene.append(selectPermutation(
                permutations, size, coordinates[1]))
            self.gene.append(selectPermutation(
                permutations, size, coordinates[2]))

    def print(self):
        print("Genome: {0}".format(self.gene))


class Cube:
    Size = 0
    Cells = {}
    Curr_Cells = {}
    Perm = {}
    Space = []
    Original = []
    BlockedPairs = []

    def __init__(self, size):
        self.BlockedPairs = []
        self.Space = numpy.zeros((size + 2, size + 2, size + 2))
        self.Space.fill(-1)
        self.Size = size
        self.Perm = dict.fromkeys(range(1, size + 1))
        for x in self.Perm.keys():
            self.Perm[x] = findCominations(
                x, list(itertools.product(range(10), repeat=3)))

        self.Cells = dict.fromkeys(range(0, size**3))
        for x in self.Cells.keys():
            coor = convertToPos(size, x)
            self.Space[coor[0] + 1, coor[1] + 1, coor[2] + 1] = x
            xperm = determineCode(self.Perm, size, coor[0])
            yperm = determineCode(self.Perm, size, coor[1])
            zperm = determineCode(self.Perm, size, coor[2])
            self.Cells[x] = Cell(
                list(map(convertPermutation, [xperm, yperm, zperm])))
        self.Curr_Cells = dict.fromkeys(range(0, size**3))
        for x in self.Curr_Cells.keys():
            self.Curr_Cells[x] = x
            self.Cells[x].generate_seq()
        self.Original = self.Space.copy()

    def iterate(self):
        for x in self.Cells.keys():
            next_move = self.Cells[x].nextmove()
            coor = numpy.add(convertToPos(self.Size, x), 1)
            os.system('clear')
            self.Cells[x].print()
            try:
                space_val = self.Space[
                    next_move[0], next_move[1], next_move[2]]
            except:
                print(self.Cells[x].sequence)
                sys.exit(1)

            print("Space not occupied" if space_val == -
                  1 else "Space occupied by {0}".format(space_val))
            if (space_val == -1 or space_val == x):
                curr_pos = self.Cells[x].curr_pos
                self.Space[curr_pos[0], curr_pos[1], curr_pos[2]] = -1
                self.Space[next_move[0], next_move[1], next_move[2]] = x
                self.Cells[x].curr_pos = next_move
                self.Cells[x].blocked = 0
            else:
                self.Cells[x].blocked += 1
                self.Cells[x].blockedby = space_val
                self.BlockedPairs.append([x, space_val])
            print(self.Space)

            print("\n")

# primeList = list(range(2, 1000))
# for x in primeList:
#     for y in range(2 * x, 1000, x):
#         if(y in primeList):
#             primeList.remove(y)

# primePowers = list()
# for x in primeList:
#     powLamda = getPower(x)
#     primePower = powLamda(2)
#     count = 2
#     while(primePower < 1000):
#         primePower = powLamda(count)
#         count = count + 1
#         if(primePower is not primePowers and primePower < 1000):
#             primePowers.append(primePower)

# trapRooms = list(primeList)
# trapRooms.extend(primePowers)
# trapRooms.sort()

# cube = Cube(27)
# for x in cube.Perm.keys():
#     print("{0}: {1}".format(x, len(cube.Perm[x])))
# print("\n")
# while hash(str(cube.Original)) != hash(str(cube.Space)):
# while True:
#     cube.iterate()
#     if(cube.Original == cube.Space).all():
#         break
#     if(len(cube.BlockedPairs) > 500):
#         print(cube.BlockedPairs)
#         break
# uniquePairs = []
# for pair in cube.BlockedPairs:
#     if not (pair in uniquePairs):
#         uniquePairs.append(pair)
# print(uniquePairs)
# print("Done!")
size = 27
permutations = generatePermutationDict(size)
seeds = []
for x in range(100):
    seed = Seed(size, permutations)
    seed.print()
    seeds.append(seed)