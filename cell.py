import numpy
import cubeMath
import globals

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
        if not self.isEmpty():
            self.start_pos = numpy.array(list(map(cubeMath.decodeDimension, code)))
            self.curr_pos = self.start_pos
            x_moves = cubeMath.getMoves(cubeMath.convertToPos(10, self.code[0]))
            y_moves = cubeMath.getMoves(cubeMath.convertToPos(10, self.code[1]))
            z_moves = cubeMath.getMoves(cubeMath.convertToPos(10, self.code[2]))

        self.sequence = []
        if (self.isEmpty() or self.isTrap()):
            self.movements = numpy.zeros((3, 3))
        else:
            self.movements = numpy.array([x_moves, y_moves, z_moves])
        self.seq_p = 0
        self.blocked = 0
        self.blockedby = -1

    def generate_seq(self):
        sequence = []
        if self.isEmpty():
            return
        sequence.append(self.start_pos[:])
        for col in range(3):
            for row in range(3):
                pos = sequence[len(sequence) - 1].copy()
                pos[row] += self.movements[row, col]
                sequence.append(pos[:])
        self.dest_pos = sequence[0]
        self.sequence = sequence

    def move(self):
        if self.isEmpty():
            return None
        if numpy.array_equal(self.dest_pos, self.curr_pos):
            if self.seq_p < len(self.sequence) - 1:
                self.seq_p += 1
            else:
                self.seq_p = 0
        self.dest_pos = self.sequence[self.seq_p]
        delta = numpy.subtract(self.dest_pos, self.curr_pos)
        if(numpy.sum(delta) != 0):
            self.next_pos = numpy.add(self.curr_pos,
                                      numpy.floor_divide(delta,
                                                         numpy.abs(numpy.sum(delta))))
            return self.next_pos
        return self.curr_pos

    def isTrap(self):
        if(self.code[0] in globals.trapRooms or
            self.code[1] in globals.trapRooms or
                self.code[2] in globals.trapRooms):
            return True
        return False
    
    def isEmpty(self):
        return self.code[0] is None or self.code[1] is None or self.code[2] is None