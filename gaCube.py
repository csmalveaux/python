
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot
import matplotlib.animation
from matplotlib import colors
from matplotlib.offsetbox import AnchoredText
import itertools
import datetime
import random
import numpy
import dill
import glob
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


def generatePopulation(permutations, size, populationSize):
    population = []
    for x in range(populationSize):
        seed = Seed(size, permutations)
        population.append(seed.gene)
        if(x != populationSize - 1):
            print("Generating {0} out of {1}".format(x + 1, populationSize), end='\r')
        else:
            print("Generating {0} out of {1}".format(x + 1, populationSize))
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
        if (code[0] in trapRooms or code[1] in trapRooms or code[2] in trapRooms):
            self.movements = numpy.zeros((3, 3))
        else:
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

    def isTrap(self):
        if(self.code[0] in trapRooms or
            self.code[1] in trapRooms or
                self.code[2] in trapRooms):
            return True
        return False


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
            coor = convertToPos(self.size, i)
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
            next_move = self.cells[x].move()
            space_val = self.space[
                next_move[0], next_move[1], next_move[2]]
            if (space_val == -1 or space_val == x or self.cells[space_val].isTrap()):
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
        count = 0;
        for x in self.cells.keys():
            if self.cells[x].isTrap():
                count += 1;
        return count


def getFitness(individual, permutations):
    cube = Cube(individual, permutations)
    iterations = 0
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
            return (iterations, cube.trapCount())
        else:
            return (points, cube.trapCount())
    else:
        return (-1 * limit, cube.trapCount()) 


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


def similarity(individual1, individual2):
    gene_match = 0
    for i in range(len(individual1)):
        if(individual1[i] == individual2[i]):
            gene_match += 1
    return (gene_match / len(individual1))


def deleteDuplicates(fitness_list):
    isDuplicates = True
    while isDuplicates:
        isDuplicates = False
        for i in range(len(fitness_list)):
            item = fitness_list[i]
            if(i < len(fitness_list) - 1):
                if(item[1] == fitness_list[i + 1][1]):
                    isDuplicates = True
                    fitness_list.remove(fitness_list[i + 1])
                    break
    return fitness_list


def generateGeneration(baseSize, populationSize, fitness, permutations):
    average_fitness = calculateAverageFitness(fitness)
    fittest = fitness[0]
    fitness = deleteDuplicates(fitness)
    if(len(fitness) < populationSize):
        survivors = fitness
    else:
        survivors = list(
            filter(lambda x: x[0] > average_fitness, fitness))
    population = []
    for survivor in survivors:
        population.append(survivor[1].copy())

    if(len(survivors) == 0 or fittest[0] <= -limit):
        population.append(fittest[1])

    for x in range(0, len(survivors), 2):
        if(len(population) < populationSize and len(survivors) - 1 >= x + 1):
            child = cross_breed(survivors[x][1], survivors[x + 1][1])
            # match_score = similarity(survivors[x][1], survivors[x + 1][1])
            child = mutation(child, 0.25, permutations)
            population.append(child.copy())

    if(len(population) < populationSize):
        random_children = generatePopulation(
            permutations, size, populationSize - len(population))
        population.extend(random_children)
    return population


def evolve(baseSize, populationSize, totalGenerations, saveDir, seeds):
    permutations = generatePermutationDict(baseSize)
    if(seeds is None):
        population = generatePopulation(permutations, baseSize, populationSize)
    else:
        population = generateGeneration(
            baseSize, populationSize, seeds, permutations)
    average_fitness = 0
    fitness = []
    genePool = {}
    trend = numpy.array([])
    top10_trend = numpy.array([])
    fitness_trapRatio = []
    scatter_fittness_failure = []
    matplotlib.pyplot.ion()

    for generation in range(total_generations):
        print("Current Genertation: {0}".format(generation + 1))
        if(generation > 0):
            population = generateGeneration(
                baseSize, populationSize, fitness, permutations)

        fitness = []
        for individual in population:
            if(str(individual) in genePool.keys()):
                fitness_score = genePool[str(individual)]
            else:
                fitness_score, numtraps = getFitness(
                    individual, permutations)
                if(fitness_score > average_fitness):
                    genePool.update({str(individual): fitness_score})
                if(fitness_score > -limit):    
                    fitness_trapRatio.append([fitness_score, (numtraps/(baseSize ** 3))])
            fitness.append([fitness_score, individual])

        fitness = sorted(
            fitness, key=lambda x: x[0], reverse=True)
        average_fitness = calculateAverageFitness(fitness)
        average_fitness_10 = calculateAverageFitness(
            fitness[0:int(len(fitness) * 0.10)])
        print("Average fitness of population: {0} ".format(
            average_fitness), end='\n\n')
        trend = numpy.append(trend, average_fitness)
        top10_trend = numpy.append(top10_trend, average_fitness_10)

        matplotlib.pyplot.figure(1).clf()
        matplotlib.pyplot.subplot(211)
        matplotlib.pyplot.plot(trend, label='Average Fitness')
        text = matplotlib.pyplot.text(0.6, 0.15, "Generation: {0} \nAverage Fittness: {1}".format(generation, average_fitness), horizontalalignment='left', verticalalignment='center', transform=matplotlib.pyplot.subplot(211).transAxes)
        matplotlib.pyplot.xlabel('Generation', fontsize=10)
        matplotlib.pyplot.ylabel('Fitness', fontsize=10)
        matplotlib.pyplot.title('Fitness vs Generation')

        matplotlib.pyplot.subplot(212)
        matplotlib.pyplot.plot(top10_trend, label='Top 10 Fitness')
        text = matplotlib.pyplot.text(0.6, 0.15, "Generation: {0} \nAverage Fittness: {1}".format(generation, average_fitness_10), horizontalalignment='left', verticalalignment='center', transform=matplotlib.pyplot.subplot(212).transAxes)
        matplotlib.pyplot.xlabel('Generation', fontsize=10)
        matplotlib.pyplot.ylabel('Fitness', fontsize=10)
        matplotlib.pyplot.title('Top 10 Fitness vs Generation')
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.figure(2).clf()

        histlist = [item[0] for item in fitness]
        histlist = list(filter(lambda x: x > -limit, histlist))
        matplotlib.pyplot.hist(numpy.asarray(histlist))
        matplotlib.pyplot.title("Fitness Scores")

        matplotlib.pyplot.figure(3).clf()
        x = list(map(lambda x: x[0], fitness_trapRatio))
        y = list(map(lambda x: x[1], fitness_trapRatio))
        #matplotlib.pyplot.scatter(x, y, alpha=0.8, c='blue', edgecolors='none', s=30, label='success')
        matplotlib.pyplot.hist2d(x, y, norm=colors.LogNorm())
        matplotlib.pyplot.colorbar()
        # x = list(map(lambda x: x[0], scatter_fittness_failure))
        # y = list(map(lambda x: x[1], scatter_fittness_failure))
        # #matplotlib.pyplot.scatter(x, y, alpha=0.8, c='red', edgecolors='none', s=30, label='failure')
        # matplotlib.pyplot.hist2d(x, y, bins=100, norm=colors.LogNorm())
        matplotlib.pyplot.title('Score vs Trap Density')
        matplotlib.pyplot.xlabel('Fitness', fontsize=10)
        matplotlib.pyplot.ylabel('Trap Density', fontsize=10)
        matplotlib.pyplot.pause(0.10)
    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.savefig(os.path.join(
        saveDir, "fitgraphs_{date:%Y%m%d_%H%M%S}.png".format(date=datetime.datetime.now())), orientation='portrait')
    matplotlib.pyplot.figure(2)
    matplotlib.pyplot.savefig(os.path.join(
        saveDir, "histgraph_{date:%Y%m%d_%H%M%S}.png".format(date=datetime.datetime.now())))
    matplotlib.pyplot.figure(3)
    matplotlib.pyplot.savefig(os.path.join(
        saveDir, "trapgraph_{date:%Y%m%d_%H%M%S}.png".format(date=datetime.datetime.now())))
    return list(
        filter(lambda x: x[0] > 0, fitness))


size = int(sys.argv[1])
populationSize = int(sys.argv[2])
total_generations = int(sys.argv[3])
startWithSeed = int(sys.argv[4])

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

global trapRooms

trapRooms = list(primeList)
trapRooms.extend(primePowers)
trapRooms.sort()

global limit
limit = int(numpy.power(10, int(numpy.log(size ** 3))))
directory = str(size)
if not os.path.isdir(directory):
    os.mkdir(directory)
files = []
files += [f for f in os.listdir(directory) if f.endswith('.pkl')]
files = [os.path.join(directory,i) for i in files]
files = sorted(files, key=os.path.getmtime, reverse=True)
if(len(files) > 0 and startWithSeed == 1):
    with open(files[0], 'rb') as f:
        fittest = dill.load(f)
    fittest = evolve(size, populationSize,
                     total_generations, directory, fittest)
else:
    fittest = evolve(size, populationSize, total_generations, directory, None)

if len(fittest) > 0:
    with open(os.path.join(directory, "fittest_{date:%Y%m%d_%H%M%S}.pkl".format(date=datetime.datetime.now())), 'wb') as f:
        dill.dump(list(fittest), f)

print("Done!")

if len(fittest) > 0:
    print("Top Functional Cubes")
    for x in fittest:
        print_individual(x)

# print("Done!")
# print("Top 10% Cubes")
# for x in fittest[0:int(len(fittest) * 0.10)]:
#     print_individual(x)

# permutations = generatePermutationDict(size)
# population = generatePopulation(permutations, size, populationSize)
# for individual in population:
#     cube = Cube(individual, permutations)
#     cube.analyze()
