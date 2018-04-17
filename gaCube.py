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


def selectPermutation(permutations, size, position):
    pos = position.copy() + 1
    perm = permutations[pos].copy()
    secure_random = random.SystemRandom()
    loc = pos
    selection = list(range(len(perm)))
    while len(selection) > 0:
        sel = secure_random.choice(selection)
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
    permutations = dict.fromkeys(range(1, size + 2))
    for x in permutations.keys():
        permutations[x] = findCominations(
            x, list(itertools.product(range(10), repeat=3)))
    return permutations


class Seed:
    gene = []

    def __init__(self, size, permutations):
        self.gene = []
        for x in range(size ** 3):
            coordinates = convertToPos(size, x)
            a = selectPermutation(permutations, size, coordinates[0])
            b = selectPermutation(permutations, size, coordinates[1])
            c = selectPermutation(permutations, size, coordinates[2])
            self.gene.append(a)
            self.gene.append(b)
            self.gene.append(c)

    def print(self):
        print("Genome: {0}".format(self.gene))


def generatePopulation(permutation, size, populationSize):
    population = []
    for x in range(populationSize):
        seed = Seed(size, permutations)
        population.append(seed.gene)
    return population


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

    def move(self):
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


class Cube:
    size = 0
    space = []
    origin = []
    cells = {}
    blocked_pairs = {}
    deadlocks = 0
    locked = False

    def __init__(self, gene, permutations):
        size = int(numpy.rint(numpy.power(len(gene) / 3, (1. / 3.))))
        self.size = size
        self.space = numpy.zeros((size + 2, size + 2, size + 2))
        self.space.fill(-1)

        self.cells = dict.fromkeys(range(0, size**3))
        for x in self.cells.keys():
            start = x * 3
            values = gene[start: (start + 3)]
            coor = convertToPos(size, x)
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
        self.origin = self.space.copy()
        self.blocked_pairs = dict.fromkeys(range(size ** 3))
        self.locked = False

    def deadlockcheck(self):
        deadlocks = 0
        for x in self.blocked_pairs.keys():
            other_cell = self.blocked_pairs[x]
            if(other_cell is not None):
                if(self.blocked_pairs[other_cell] == x):
                    deadlocks += 1
                    if(self.cells[x].start_pos != self.cells[x].curr_pos).any():
                        self.locked = True
                    if(self.cells[other_cell].start_pos != self.cells[other_cell].curr_pos).any():
                        self.locked = True
        return deadlocks // 2

    def iterate(self):
        points = 0
        for x in self.cells.keys():
            next_move = self.cells[x].move()
            space_val = self.space[
                next_move[0], next_move[1], next_move[2]]
            if (space_val == -1 or space_val == x):
                curr_pos = self.cells[x].curr_pos
                self.space[curr_pos[0], curr_pos[1], curr_pos[2]] = -1
                self.space[next_move[0], next_move[1], next_move[2]] = x
                self.cells[x].curr_pos = next_move
                if self.cells[x].blocked > 0:
                    points += self.cells[x].blocked
                else:
                    points += 1
                self.cells[x].blocked = 0
            else:
                self.cells[x].blocked += 1
                points -= self.cells[x].blocked
                self.blocked_pairs[x] = int(space_val)
        self.deadlocks = self.deadlockcheck()
        return points


def getFitness(individual, permutations):
    cube = Cube(individual, permutations)
    iterations = 0
    limit = int(numpy.power(100, int(numpy.log(cube.size ** 3))))
    points = limit

    while points > 0:
        iterations += 1
        points += cube.iterate()
        if(cube.origin == cube.space).all():
            break
        if cube.locked:
            break
        if(iterations > limit):
            break

    if(not cube.locked and iterations > 1 and iterations < limit):
        if points > 0:
            return iterations
        else:
            return points
    else:
        return -1 * limit


def calculateAverageFitness(population_fitness):

    total_fitness = 0
    for x in population_fitness:
        total_fitness += x[0]
    return total_fitness / len(population_fitness)


def mutation(individual, percent_mutation, permutations):
    chromosome_len = len(individual)
    base_size = int(numpy.rint(numpy.power(chromosome_len / 3, (1. / 3.))))
    for x in range(int(chromosome_len * percent_mutation)):
        gene_index = int(random.random() * chromosome_len)
        cell_index = gene_index // 3
        dim_index = gene_index % 3
        coordinates = convertToPos(base_size, cell_index)
        mutant_gene = selectPermutation(
            permutations, base_size, coordinates[dim_index])
        individual[gene_index] = mutant_gene
    return individual


def cross_breed(individual1, individual2):
    offspring = []
    for i in range(len(individual1)):
        if(int(100 * random.random()) < 50):
            offspring.append(individual1[i])
        else:
            offspring.append(individual2[i])

    return offspring


def print_individual(individual):
    print("\tGenome: {0}".format(individual[1]))
    print("\tScore: {0}".format(individual[0]), end='\n\n')


size = 5
populationSize = 100
permutations = generatePermutationDict(size)
population = generatePopulation(permutations, size, populationSize)
total_generations = 1000
average_fitness = 0
population_fitness = []
archive = {}

for generation in range(total_generations):
    print("Current Genertation: {0}".format(generation))

    if(generation > 0):
        # if(average_fitness > 0):
        #     survivors = list(
        #         filter(lambda x: x[0] > average_fitness, population_fitness))
        # else:
        #     survivors = list(
        #         filter(lambda x: x[0] > 0, population_fitness))
        survivors = list(
                filter(lambda x: x[0] > average_fitness, population_fitness))

        population = []
        for survivor in survivors:
            population.append(survivor[1].copy())

        for x in range(0, len(survivors), 2):
            if(len(population) < populationSize and len(survivors) - 1 >= x + 1):
                child = cross_breed(survivors[x][1], survivors[x + 1][1])
                child = mutation(child, 0.05, permutations)
                population.append(child.copy())

        if(len(population) < populationSize):
            random_children = generatePopulation(permutations, size, populationSize - len(population))
            population.extend(random_children)

    population_fitness = []
    for individual in population:
        if(str(individual) in archive.keys()):
            fitness_score = archive[str(individual)]
        else:
            fitness_score = getFitness(individual, permutations)
            if(fitness_score > average_fitness):
                archive.update({str(individual): fitness_score})
        population_fitness.append([fitness_score, individual])

    population_fitness = sorted(
        population_fitness, key=lambda x: x[0], reverse=True)
    print("Top Five Fittest: ")
    for x in range(5):
        print_individual(population_fitness[x])

    average_fitness = calculateAverageFitness(population_fitness)
    print("Average fitness of population: {0} ".format(
        average_fitness), end='\n\n')

print("Done!")
print("All functional Cubes")

for x in list(filter(lambda x: x[0] > 0, population_fitness)):
    print_individual(x)
