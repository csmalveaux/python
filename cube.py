import numpy as np

class Cube:
	def __init__(self, gene, permutations, startcoor=[0,0,0]):
		size = int(np.rint(np.power(len(gene) / 3, (1. / 3.))))
        self.size = size
        self.space = np.zeros((size + 2, size + 2, size + 2))
        self.space.fill(-1)

        self.cells = dict.fromkeys(range(0, size**3))
        for x in self.cells.keys():
            start = x * 3
            values = gene[start: (start + 3)]
            coor = numpy.array([sum(i) for i in zip(convertToPos(size, x), startcoor)])
            self.space[coor[0] + 1, coor[1] + 1, coor[2] + 1] = x
            permutation = permutations[coor[0] + 1]
            xperm = permutation[values[0]]
            permutation = permutations[coor[1] + 1]
            yperm = permutation[values[1]]
            permutation = permutations[coor[2] + 1]
            zperm = permutation[values[2]]
            self.cells[x] = Cell(
                list(map(convertPermutation, [xperm, yperm, zperm])))
            self.cells[x].generate_seq()
            if(self.cells[x].isTrap()):
                self.traps += 1
        self.origin = self.space.copy()
        self.blocked_pairs = dict.fromkeys(range(size ** 3))
        self.locked = False

    @property
    def size(self):
    	return self._size
    
    @property
    def space(self):
    	return self._space
    
    @property
    def deadlocks(self):
    	return self._deadlocks
    