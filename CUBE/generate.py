from . component import Cube
from . component import Cell
import numpy


class Seed:

    def convertToPos(base, cellNumber):
        xPos = cellNumber // (base ** 2)
        yPos = (cellNumber % (base ** 2)) // base
        zPos = cellNumber % base

        return numpy.array([xPos, yPos, zPos])

    def convertPermutation(permutation):
        return permutation[0] * 100 + permutation[1] * 10 + permutation[2]

    def matchPermutation(permutations, position, code):
        pos = position.copy() + 1
        perm = permutations[pos].copy()
        for x in list(range(len(perm))):
            if convertPermutation(perm[x]) == code:
                return x
        return -1


    def validPermutations(permutations, size, position):
        pos = position.copy()
        loc = pos
        perm = permutations[pos].copy()
        results = []
        for x in perm:
            moves = getMoves(x)
            usable = True
            loc = pos
            for m in moves:
                loc += m
                if loc < 0 or loc >= (size + 2):
                    usable = False
            if usable:
                results.append(x)
        return results

    def selectPermutation(permutations, size, position, density=None):
        pos = position.copy() + 1
        perm = permutations[pos].copy()
        options = validPermutations(permutations, size, pos)
        selection = []
        trap = []
        safe = []
        secure_random = random.SystemRandom()

        if density is not None:
            for x in options:
                if convertPermutation(x) in trapRooms:
                    trap.append(x)
                else:
                    safe.append(x)
            if len(trap) > 0 and len(safe) > 0:
                ran = secure_random.random()
                if ran > density:
                    selection = safe 
                else:
                    selection = trap
            elif len(trap) > 0:
                selection = trap
            else:
                selection = safe
        else:
            selection = options

        sel = secure_random.choice(selection)
        if pos == 6 and density == 1.0:
            sel = (2, 2, 2)
        if pos == 21 and density == 1.0:
            sel = (7, 7, 7)
        if pos == 24 and density == 1.0:
            sel = (8, 8, 8)
        retVal = matchPermutation(permutations, position, convertPermutation(sel))
        return retVal

    def __init__(self, size, permutations, cube=None, density=None, gene=None, cell=None):
        self.gene = []
        if cube is None:
            if gene is None:
                for x in range(size ** 3):
                    coordinates = self.convertToPos(size, x)
                    for dim in range(3):
                        self.gene.append()


