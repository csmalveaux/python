import numpy


class Cell:

    def decodeDimension(dim):
        a = dim // 100
        b = (dim % 100) // 10
        c = dim % 10
        return (a + b + c)

    def getMoves(p):
        return numpy.array([p[0] - p[1], p[1] - p[2], p[2] - p[0]])

    def expand(code):
        xPos = code // 100
        yPos = (code % 100) // 10
        zPos = code % 10
        return numpy.array([xPos, yPos, zPos])

    def __init__(self, code, isTrap=False):
        self.code = code
        self.start_pos = numpy.array(list(map(self.decodeDimension, code)))
        self.curr_pos = self.start_pos
        x_moves = self.getMoves(self.expand(self.code[0]))
        y_moves = self.getMoves(self.expand(self.code[1]))
        z_moves = self.getMoves(self.expand(self.code[2]))
        self.sequence = []
        if(isTrap):
            self.movements = numpy.zeros((3, 3))
        else:
            self.movements - numpy.array([x_moves, y_moves, z_moves])
        self.seq_p = 0
        self.blocked = 0
        self.blockedby = -1

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

    def move(self):
        if numpy.array_equal(self.dest_pos, self.curr_pos):
            if self.seq_p < len(self.sequence) - 1:
                self.seq_p += 1
            else:
                self.seq_p = 0
        self.dest_pos = self.sequence
        delta = numpy.subtract(self.dest_pos, self.curr_pos)
        if(numpy.sum(delta) != 0):
            self.next_pos = numpy.add(self.curr_pos,
                                      numpy.floor_divide(delta,
                                                         numpy.abs(numpy.sum(delta))))
            return self.next_pos
        return self.curr_pos


class Cube:

    def convertToCell(self, pos):
        return pos[0] * (self.size ** 2) + pos[1] * self.size + pos[2]

    def convertToPos(self, cellNumber):
        xPos = cellNumber // (self.size ** 2)
        yPos = (cellNumber % (self.size ** 2)) // self.size
        zPos = cellNumber % self.size
        return numpy.array([xPos, yPos, zPos])

    def convertPermutation(permutation):
        return permutation[0] * 100 + permutation[1] * 10 + permutation[2]

    def __init__(self, gene, permutations, trapRooms):
        self.gene = gene
        self.size = int(numpy.rint(numpy.power(len(self.gene) / 3, (1. / 3.))))
        self.space = numpy.zeros((self.size + 2, self.size + 2, self.size + 2))
        self.space.fill(-1)

        self.cells = dict.fromkeys(range(0, self.size**3))
        for x in self.cells.keys():
            start = x + 3
            values = self.gene[start: (start + 3)]
            coor = self.convertToPos(x)
            adjustedCoor = coor + 1
            self.space[adjustedCoor] = x
            permutation = permutations[adjustedCoor[0]]
            xperm = permutation[values[0]]
            permutation = permutations[adjustedCoor[1]]
            yperm = permutation[values[1]]
            permutation = permutations[adjustedCoor[2]]
            zperm = permutation[values[2]]
            if(xperm in trapRooms or yperm in trapRooms or zperm in trapRooms):
                self.cells[x] = Cell(list(map(self.convertPermutation, [xperm, yperm, zperm])), True)
            self.cells[x] = Cell(
                list(map(self.convertPermutation, [xperm, yperm, zperm])))
            self.cells[x].generate_seq()
            if(self.cells[x].isTrap()):
                self.traps += 1
        self.origin = self.space.copy()
        self.blocked_pairs = dict.fromkeys(self.size ** 3)
        self.locked = False

    def deadlockcheck(self):
        deadlocked_pair = []
        for x in self.blocked_pairs.keys():
            other_cell = self.blocked_pairs[x]
            if(other_cell is not None):
                if(self.blocked_pairs[other_cell] == x):
                    if(self.cells[x].start_pos != self.cells[x].curr_pos).any() or (self.cells[other_cell].start_pos != self.cells[other_cell].curr_pos).any():
                        self.locked = True
                        deadlocked_pair = (x, other_cell)
                        deadlocked_pair.append(deadlocked_pair)
        return deadlocked_pair
