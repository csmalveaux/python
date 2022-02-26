
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot
import matplotlib.animation
from matplotlib import colors
import itertools
import datetime
import random
import numpy
import dill
import sys
import os

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph import GlobbingFilter
from pycallgraph.output import GraphvizOutput


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


def getMoves(p):
    return numpy.array([p[0] - p[1], p[1] - p[2], p[2] - p[0]])


def decodeDimension(dim):
    a = dim // 100
    b = (dim % 100) // 10
    c = dim % 10
    return (a + b + c)


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
            if loc < 0 or loc >= 27:
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


def generatePermutationDict(size):
    permutations = dict.fromkeys(range(1, size + 2))
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
        if (code[0] in trapRooms or code[1] in trapRooms or code[2] in trapRooms):
            self.movements = numpy.zeros((3, 3))
        else:
            self.movements = numpy.array([x_moves, y_moves, z_moves])
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
    deadlocks = []
    locked = False
    traps = 0
    startcoor = []

    def __init__(self, gene, permutations, startcoor=[0,0,0]):
        size = int(numpy.rint(numpy.power(len(gene) / 3, (1. / 3.))))
        self.size = size
        self.space = numpy.zeros((27, 27, 27))
        self.space.fill(-1)
        self.startcoor = startcoor
        self.cells = dict.fromkeys(range(0, size**3))
        for x in self.cells.keys():
            start = x * 3
            values = gene[start: (start + 3)]
            coor = numpy.array([sum(i) for i in zip(convertToPos(size, x), startcoor)])
            #self.space[coor[0] + 1, coor[1] + 1, coor[2] + 1] = x
            self.space[coor + 1] = x
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
        count = 0
        for x in self.cells.keys():
            if self.cells[x].isTrap():
                count += 1
        return count


class Seed:
    gene = []

    def __init__(self, size, permutations, cube=None, density=None, gene=None, cell=None, startcoor=[0, 0, 0]):
        self.gene = []
        if cube is None:
            if gene is None:
                for x in range(size ** 3):
                    coordinates = [sum(i) for i in zip(convertToPos(size, x), startcoor)]  
                    for dim in range(3):
                        self.gene.append(selectPermutation(
                            permutations, size, coordinates[dim], density))
            else:
                for x in range(size ** 3):
                    if x != cell:
                        for g in range(3):
                            self.gene.append(gene[(x * 3) + g])
                    else:
                        coordinates = convertToPos(size, x)
                        for dim in range(3):
                            self.gene.append(selectPermutation(
                                permutations, size, coordinates[dim], density))
        else:
            originSize = cube.size
            for x in range(size ** 3):
                coordinates = [sum(i) for i in zip(convertToPos(size, x), startcoor)]  
                if(coordinates < originSize).all():
                    position = convertToCell(originSize, coordinates)
                    code = cube.cells[position].code
                    for dim in range(3):
                        self.gene.append(matchPermutation(
                            permutations, coordinates[dim], code[dim]))
                else:
                    coordinates = [sum(i) for i in zip(convertToPos(size, x), startcoor)]  
                    if density == 1.0:
                        # cannotTrap = [False, False, False]
                        for dim in range(3):
                            value = selectPermutation(permutations, size, coordinates[dim], density)
                            self.gene.append(value)
                            perms = permutations[coordinates[dim] + 1]
                            value = convertPermutation(perms[value])
                            # if value not in trapRooms:
                            #     cannotTrap[dim] = True
                        # if cannotTrap[0] and cannotTrap[1] and cannotTrap[2]:
                        #     print("{0} is not a trap".format(coordinates))
                    else:
                        for dim in range(3):
                            if density is not None:
                                self.gene.append(selectPermutation(
                                    permutations, size, coordinates[dim], density / 3))
                            else:
                                self.gene.append(selectPermutation(
                                    permutations, size, coordinates[dim]))
        # input()

    def print(self):
        print("Genome: {0}".format(self.gene))


def generatePopulation(permutations, size, populationSize, cube=None, density=None, startcoor=[0, 0, 0]):
    population = []
    for x in range(populationSize):
        if cube is None:
            seed = Seed(size, permutations, None, None, None, None, startcoor)
        else:
            if density is None:
                seed = Seed(size, permutations, cube, None, startcoor)
            else:
                seed = Seed(size, permutations, cube, density, startcoor)
                if(density == 1.0):
                    density = None
        population.append(seed.gene)
        if(x != populationSize - 1):
            print("Generating {0} out of {1}".format(
                x + 1, populationSize), end='\r')
        else:
            print("Generating {0} out of {1}".format(x + 1, populationSize))
    return population


def getFitness(individual, permutations, startcoor=[0,0,0]):
    cube = Cube(individual, permutations, startcoor)
    traps = cube.traps
    totalCells = cube.size ** 3
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
            score = iterations * (1 - (traps / totalCells))
            return (score, iterations, traps, cube.locked)
        else:
            score = points * (1 - (traps / totalCells))
            return (score, iterations, traps, cube.locked)
    else:
        return (-1 * limit, iterations, traps, cube.locked)


def calculateAverageFitness(population_fitness):
    total_fitness = 0
    for x in population_fitness:
        total_fitness += x[0]
    return total_fitness / len(population_fitness)


def extendedCubeCells(size1, size2):
    retVals = []
    if size1 == size2:
        return retVals

    if size1 > size2:
        temp = size1
        size1 = size2
        size2 = temp

    for x in range(size2 ** 3):
        coordinates = convertToPos(size2, x)
        if(coordinates > size1).any():
            retVals.append(x)

    return retVals


def mutation(individual, percent_mutation, permutations, seed=None, startcoor=[0, 0, 0]):
    chromosome_len = len(individual)
    base_size = int(numpy.rint(numpy.power(chromosome_len / 3, (1. / 3.))))
    if seed is None:
        for x in range(int(chromosome_len * percent_mutation)):
            gene_index = int(random.random() * chromosome_len)
            cell_index = gene_index // 3
            dim_index = gene_index % 3
            coordinates = [sum(i) for i in zip(convertToPos(base_size, cell_index), startcoor)]   
            mutant_gene = selectPermutation(
                permutations, base_size, coordinates[dim_index])
            individual[gene_index] = mutant_gene
        return individual
    else:
        seed_size = int(numpy.rint(numpy.power(len(seed) / 3, (1. / 3.))))
        changeableCells = extendedCubeCells(seed_size, base_size)
        for x in range(int(len(changeableCells) * percent_mutation)):
            cell_index = random.choice(changeableCells)
            dim_index = random.choice(range(3))
            gene_index = cell_index * 3 + dim_index
            coordinates = [sum(i) for i in zip(convertToPos(base_size, cell_index), startcoor)]
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
    print("\tScore: {0}".format(individual[0]))
    print("\tDensity: {0}".format(individual[2]))
    print("\tCycles: {0}".format(individual[3]), end='\n\n')


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


def fixDeadlocks(cube, gene, permutations):
    while not cube.locked:
        cube.iterate()

    new_gene = gene.copy()
    for x in cube.deadlocks:
        cell_0, cell_1 = x
        new_cube = Cube(new_gene, permutations, cube.startcoor)
        if not new_cube.cells[cell_0].isTrap() and not new_cube.cells[cell_1].isTrap():
            rand = random.SystemRandom()
            new_seed = Seed(cube.size, permutations, None,
                            1.0, gene, rand.choice(list(x)))
            new_gene = new_seed.gene.copy()
    if hash(str(gene)) != hash(str(new_gene)):
        return new_gene
    else:
        return None


def generateGeneration(baseSize, populationSize, fitness, deadlocked, permutations, seed=None, startcoor=[0, 0, 0]):
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

    if(len(survivors) == 0 and fittest[0] > -limit):
        population.append(fittest[1])

    for x in range(0, len(survivors), 2):
        if(len(population) < populationSize and len(survivors) - 1 >= x + 1):
            child = cross_breed(survivors[x][1], survivors[x + 1][1])
            # match_score = similarity(survivors[x][1], survivors[x + 1][1])
            child = mutation(child, 0.25, permutations, seed, startcoor)
            population.append(child.copy())

    deadlockCount = 0
    for x in deadlocked:
        deadlockCount += 1
        print("Deadlock(s) processed: {0}".format(deadlockCount), end="\r")
        cube = Cube(x, permutations, startcoor)
        corrected_individual = fixDeadlocks(cube, x, permutations)
        if len(population) < populationSize and corrected_individual is not None:
            population.append(corrected_individual)
        if len(population) == populationSize:
            break
    if deadlockCount > 0:
        print("Deadlock(s) processed: {0}".format(deadlockCount))

    if(len(population) < populationSize):
        if seed is None:
            random_children = generatePopulation(
                permutations, baseSize, populationSize - len(population), None, None, startcoor)
        else:
            cube = Cube(seed[1], permutations, startcoor)
            if len(survivors) == 0 or average_fitness == -limit:
                random_children = generatePopulation(
                    permutations, baseSize, populationSize - len(population), cube, 1.0)
            else:
                random_children = generatePopulation(
                    permutations, baseSize, populationSize - len(population), cube)
        population.extend(random_children)
    return population


def evolve(baseSize, populationSize, totalGenerations, saveDir, seeds=None, seed=None, startcoor=[0, 0, 0]):
    permutations = generatePermutationDict(26)
    if seeds is None:
        if seed is None:
            population = generatePopulation(
                permutations, baseSize, populationSize, None, None, startcoor)
        else:
            cube = Cube(seed[1], permutations, startcoor)
            population = generatePopulation(
                permutations, baseSize, populationSize, cube, 1.0)
    else:
        population = generateGeneration(
            baseSize, populationSize, seeds, [], permutations)

    average_fitness = 0
    viable = []
    fitness = []
    deadlocked = []
    genePool = {}
    trend = numpy.array([])
    top10_trend = numpy.array([])
    fitness_trapRatio = []
    fitness_iteration = []
    trapRatio_iteration = []
    matplotlib.pyplot.ion()

    for generation in range(total_generations):
        print("Current Genertation: {0}".format(generation + 1))
        if(generation > 0):
            if seed is None:
                population = generateGeneration(
                    baseSize, populationSize, fitness, deadlocked, permutations, None, startcoor)
            else:
                population = generateGeneration(
                    baseSize, populationSize, fitness, deadlocked, permutations, seed)

        fitness = []
        deadlocked = []
        for individual in population:
            if(hash(str(individual)) in genePool.keys()):
                fitness_score = genePool[hash(str(individual))]
            else:
                fitness_score, iterations, numtraps, locked = getFitness(
                    individual, permutations, startcoor)
                if(fitness_score > average_fitness):
                    genePool.update({hash(str(individual)): fitness_score})
                if(fitness_score > -limit):
                    fitness_trapRatio.append(
                        [fitness_score, (numtraps / (baseSize ** 3))])
                    if fitness_score > 0:
                        fitness_iteration.append(
                            [fitness_score, iterations])
                        trapRatio_iteration.append(
                            [(numtraps / (baseSize ** 3)), iterations])
                        viable.append([fitness_score, individual, (numtraps / (baseSize ** 3)), iterations])
                if locked: 
                    deadlocked.append(individual)

            fitness.append([fitness_score, individual, (numtraps / (baseSize ** 3)), iterations])
            print("Processed: {0}".format(len(fitness)), end='\r')
        print("Processed: {0}".format(len(fitness)))
        viable = sorted(viable, key=lambda x: x[0], reverse=False)
        fitness = sorted(
            fitness, key=lambda x: x[0], reverse=True)
        average_fitness = calculateAverageFitness(fitness)
        average_fitness_10 = calculateAverageFitness(
            fitness[0:int(len(fitness) * 0.10)])
        print("Average fitness of population: {0} ".format(
            average_fitness))
        print("Average fitness of the fittest: {0}".format(
            average_fitness_10), end='\n\n')
        trend = numpy.append(trend, average_fitness)
        top10_trend = numpy.append(top10_trend, average_fitness_10)

        matplotlib.pyplot.figure(1).clf()
        matplotlib.pyplot.subplot(211)
        matplotlib.pyplot.plot(trend, label='Average Fitness')
        text = matplotlib.pyplot.text(0.6, 0.15, "Generation: {0} \nAverage Fittness: {1:.2f}".format(
            generation, average_fitness), horizontalalignment='left', verticalalignment='center', transform=matplotlib.pyplot.subplot(211).transAxes)
        matplotlib.pyplot.xlabel('Generation', fontsize=10)
        matplotlib.pyplot.ylabel('Fitness', fontsize=10)
        matplotlib.pyplot.title('Fitness vs Generation')

        matplotlib.pyplot.subplot(212)
        matplotlib.pyplot.plot(top10_trend, label='Top 10 Fitness')
        text = matplotlib.pyplot.text(0.6, 0.15, "Generation: {0} \nAverage Fittness: {1:.2f}".format(
            generation, average_fitness_10), horizontalalignment='left', verticalalignment='center', transform=matplotlib.pyplot.subplot(212).transAxes)
        matplotlib.pyplot.xlabel('Generation', fontsize=10)
        matplotlib.pyplot.ylabel('Fitness', fontsize=10)
        matplotlib.pyplot.title('Top 10 Fitness vs Generation')
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.figure(2).clf()

        histlist = [item[0] for item in fitness]
        histlist = list(filter(lambda x: x > -limit, histlist))
        matplotlib.pyplot.hist(numpy.asarray(histlist))
        matplotlib.pyplot.title("Fitness Scores")
        
        if(len(fitness_trapRatio) > 0):
            matplotlib.pyplot.figure(3).clf()
            x = list(map(lambda x: x[0], fitness_trapRatio))
            y = list(map(lambda x: x[1], fitness_trapRatio))
            matplotlib.pyplot.hist2d(x, y, norm=colors.LogNorm())
            matplotlib.pyplot.colorbar()
            matplotlib.pyplot.title('Fitness vs Trap Density')
            matplotlib.pyplot.xlabel('Fitness', fontsize=10)
            matplotlib.pyplot.ylabel('Trap Density', fontsize=10)

        if(len(fitness_iteration) > 0):
            matplotlib.pyplot.figure(4).clf()
            x = list(map(lambda x: x[1], fitness_iteration))
            y = list(map(lambda x: x[0], fitness_iteration))
            matplotlib.pyplot.hist2d(x, y, norm=colors.LogNorm())
            matplotlib.pyplot.colorbar()
            matplotlib.pyplot.title('Cycle Length vs Fitness')
            matplotlib.pyplot.xlabel('Cycle Length', fontsize=10)
            matplotlib.pyplot.ylabel('Fitness', fontsize=10)

        if(len(trapRatio_iteration) > 0):
            matplotlib.pyplot.figure(5).clf()
            x = list(map(lambda x: x[1], trapRatio_iteration))
            y = list(map(lambda x: x[0], trapRatio_iteration))
            matplotlib.pyplot.hist2d(x, y, norm=colors.LogNorm())
            matplotlib.pyplot.colorbar()
            matplotlib.pyplot.title('Cycle Length vs Trap Density')
            matplotlib.pyplot.xlabel('Cycle Length', fontsize=10)
            matplotlib.pyplot.ylabel('Trap Density', fontsize=10)
        matplotlib.pyplot.pause(0.10)
    if seed is None:
        matplotlib.pyplot.figure(1)
        matplotlib.pyplot.savefig(os.path.join(
            saveDir, "fitgraphs_{date:%Y%m%d_%H%M%S}.png".format(date=datetime.datetime.now())), orientation='portrait')
        matplotlib.pyplot.figure(2)
        matplotlib.pyplot.savefig(os.path.join(
            saveDir, "histgraph_{date:%Y%m%d_%H%M%S}.png".format(date=datetime.datetime.now())))
        if(len(fitness_trapRatio) > 0):
            matplotlib.pyplot.figure(3)
            matplotlib.pyplot.savefig(os.path.join(
                saveDir, "trapgraph_{date:%Y%m%d_%H%M%S}.png".format(date=datetime.datetime.now())))
        if(len(fitness_iteration) > 0):
            matplotlib.pyplot.figure(4)
            matplotlib.pyplot.savefig(os.path.join(
                saveDir, "score_cyclegraph_{date:%Y%m%d_%H%M%S}.png".format(date=datetime.datetime.now())))
        if(len(trapRatio_iteration) > 0):
            matplotlib.pyplot.figure(5)
            matplotlib.pyplot.savefig(os.path.join(
                saveDir, "trap_cyclegraph_{date:%Y%m%d_%H%M%S}.png".format(date=datetime.datetime.now())))
    
    return viable



populationSize = int(sys.argv[1])
total_generations = int(sys.argv[2])

primeList = list(range(2, 1000))
for x in primeList:
    for y in range(2 * x, 1000, x):
        if(y in primeList):
            primeList.remove(y)

primePowers = list()
for x in primeList:
    powLamda = lambda y: x ** y
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
limit = int(numpy.power(10, int(numpy.log(3 ** 3))))

directory = str(26)
if not os.path.isdir(directory):
    os.mkdir(directory)
files = []
files += [f for f in os.listdir(directory) if f.endswith('.pkl')]
files = [os.path.join(directory, i) for i in files]
files = sorted(files, key=os.path.getmtime, reverse=True)

subcube = dict.fromkeys(itertools.combinations_with_replacement([2, 6, 10, 15, 19, 23], 3))
for coordinate in subcube.keys():
    fittest = evolve(3, populationSize, total_generations, directory, None, None, coordinate)
    subcube[coordinate] = fittest
    print("# Viable: {0}".format(len(fittest)))
    print("Top 10% Cubes")
    for x in fittest[0:int(len(fittest) * 0.10)]:
        print_individual(x)
    with open(os.path.join(directory, "fittest_{coor}_{date:%Y%m%d_%H%M%S}.pkl".format(coor=''.join(map(str, coordinate)),date=datetime.datetime.now())), 'wb') as f:
        dill.dump(list(fittest), f)

print("Done!")
