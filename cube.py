import numpy
import cubeMath
import matplotlib
import globals

from cell import Cell

class Cube:
    size = 0
    space = []
    origin = []
    cells = {}
    blocked_pairs = {}
    deadlocks = []
    locked = False
    traps = 0

    def __init__(self, gene):
        size = int(numpy.rint(numpy.power(len(gene) / 3, (1. / 3.))))
        self.size = size
        self.space = numpy.zeros((size + 2, size + 2, size + 2))
        self.space.fill(-1)

        self.cells = dict.fromkeys(range(0, size**3))
        for x in self.cells.keys():
            start = x * 3
            values = gene[start: (start + 3)]
            coor = cubeMath.convertToPos(size, x)
            #self.space[coor[0] + 1, coor[1] + 1, coor[2] + 1] = x
            
            permutation = globals.permutations[coor[0] + 1]
            xperm = None if (values[0] is None) else permutation[values[0]]
            permutation = globals.permutations[coor[1] + 1]
            yperm = None if (values[1] is None) else permutation[values[1]]
            permutation = globals.permutations[coor[2] + 1]
            zperm = None if (values[2] is None) else permutation[values[2]]
            self.cells[x] = Cell(
                list(map(cubeMath.convertPermutation, [xperm, yperm, zperm])))
            self.cells[x].generate_seq()
            if(self.cells[x].isTrap()):
                self.traps += 1
            self.space[coor[0] + 1, coor[1] + 1, coor[2] + 1] = -1 if self.cells[x].isEmpty() else x
        self.origin = self.space.copy()
        self.blocked_pairs = dict.fromkeys(range(size ** 3))
        self.locked = False

    def deadlockcheck(self):
        deadlocked_pairs = []
        for x in self.blocked_pairs.keys():
            other_cell = self.blocked_pairs[x]
            if(other_cell is not None):
                if(self.blocked_pairs[other_cell] == x):
                    if(self.cells[x].start_pos != self.cells[x].curr_pos).any() or (self.cells[other_cell].start_pos != self.cells[other_cell].curr_pos).any():
                        self.locked = True
                        deadlocked_pair = (x, other_cell)
                        deadlocked_pairs.append(deadlocked_pair)
        return deadlocked_pairs

    def analyze(self):
        matplotlib.pyplot.ion()
        matplotlib.pyplot.figure(4).clf()
        matplotlib.pyplot.gca(projection='3d')

        x, y, z = numpy.meshgrid(numpy.arange(self.size + 2),
                                 numpy.arange(self.size + 2),
                                 numpy.arange(self.size + 2))
        u = numpy.zeros((self.size + 2, self.size + 2, self.size + 2))
        v = numpy.zeros((self.size + 2, self.size + 2, self.size + 2))
        w = numpy.zeros((self.size + 2, self.size + 2, self.size + 2))

        for i in self.cells.keys():
            coor = cubeMath.convertToPos(self.size, i)
            movements = self.cells[i].movements
            u[coor[0] + 1, coor[1] + 1, coor[2] + 1] = movements[0, 0]
            v[coor[0] + 1, coor[1] + 1, coor[2] + 1] = movements[1, 0]
            w[coor[0] + 1, coor[1] + 1, coor[2] + 1] = movements[2, 0]
        matplotlib.pyplot.quiver(
            x, y, z, u, v, w, cmap=matplotlib.pyplot.cm.jet, normalize=True)
        input()

    def iterate(self):
        points = 0
        for x in self.cells.keys():
            if self.cells[x].isEmpty():
                continue
            next_move = self.cells[x].move()
            space_val = self.space[
                next_move[0], next_move[1], next_move[2]]
            if (space_val == -1 or space_val == x or self.cells[space_val].isTrap() or self.cells[space_val].isEmpty()):
                curr_pos = self.cells[x].curr_pos
                self.space[curr_pos[0], curr_pos[1], curr_pos[2]] = -1
                self.space[next_move[0], next_move[1], next_move[2]] = x
                self.cells[x].curr_pos = next_move
                if self.cells[x].blocked > 0:
                    points += self.cells[x].blocked
                else:
                    points += 1
                self.cells[x].blocked = 0
                self.blocked_pairs[x] = None
            else:
                self.cells[x].blocked += 1
                points -= self.cells[x].blocked
                self.blocked_pairs[x] = int(space_val)
        self.deadlocks = self.deadlockcheck()
        return points

    def trapCount(self):
        count = 0
        for x in self.cells.keys():
            if self.cells[x].isTrap():
                count += 1
        return count
    
    def emptyCount(self):
        count = 0
        for x in self.cells.keys():
            if self.cells[x].isEmpty():
                count += 1
        return count
