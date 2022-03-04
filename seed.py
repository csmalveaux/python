import cubeMath
import random
import globals

class Seed:
    gene = []

    def __init__(self, cube=None, density=None, gene=None, cell=None):
        self.gene = []
        if cube is None:
            if gene is None:
                for x in range(globals.base_size ** 3):
                    coordinates = cubeMath.convertToPos(globals.base_size, x)
                    for dim in range(3):
                        self.gene.append(cubeMath.selectPermutation(coordinates[dim], density))
            else:
                for x in range(globals.base_size ** 3):
                    if x != cell:
                        for g in range(3):
                            self.gene.append(gene[(x * 3) + g])
                    else:
                        secure_random = random.SystemRandom()
                        coordinates = cubeMath.convertToPos(globals.base_size, x)
                        cells = []
                        if not cubeMath.edgeCheck(x):
                            cells.append((None, None, None))
                        real_cell = []
                        for dim in range(3):
                            real_cell.append(cubeMath.selectPermutation(coordinates[dim], density))
                        cells.append(real_cell)
                        adding_cell = secure_random.choice(cells)
                        for c in adding_cell:
                            self.gene.append(c)
                        
        else:
            originSize = cube.size
            for x in range(globals.base_size ** 3):
                coordinates = cubeMath.convertToPos(globals.base_size, x)
                if(coordinates < originSize).all():
                    position = cubeMath.convertToCell_(originSize, coordinates)
                    code = cube.cells[position].code
                    for dim in range(3):
                        self.gene.append(cubeMath.matchPermutation(coordinates[dim], code[dim]))
                else:
                    coordinates = cubeMath.convertToPos(globals.base_size, x)
                    if density == 1.0:
                        # cannotTrap = [False, False, False]
                        for dim in range(3):
                            value = cubeMath.selectPermutation(coordinates[dim], density)
                            self.gene.append(value)
                            perms = globals.permutations[coordinates[dim] + 1]
                            value = cubeMath.convertPermutation(perms[value])
                            # if value not in trapRooms:
                            #     cannotTrap[dim] = True
                        # if cannotTrap[0] and cannotTrap[1] and cannotTrap[2]:
                        #     print("{0} is not a trap".format(coordinates))
                    else:
                        for dim in range(3):
                            if density is not None:
                                self.gene.append(cubeMath.selectPermutation(coordinates[dim], density / 3))
                            else:
                                self.gene.append(cubeMath.selectPermutation(coordinates[dim]))
        # input()

    def print(self):
        print("Genome: {0}".format(self.gene))