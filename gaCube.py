
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot
import matplotlib.animation
from matplotlib import colors
import datetime
import random
import numpy
import dill
import sys
import os
import cubeMath
import globals

from seed import Seed
from cube import Cube


def generatePopulation(populationSize, cube=None, density=None):
    population = []
    for x in range(populationSize):
        if cube is None:
            seed = Seed()
        else:
            if density is None:
                seed = Seed(cube)
            else:
                seed = Seed(cube, density)
                if(density == 1.0):
                    density = None
        population.append(seed.gene)
        if(x != populationSize - 1):
            print("Generating {0} out of {1}".format(
                x + 1, populationSize), end='\r')
        else:
            print("Generating {0} out of {1}".format(x + 1, populationSize))
    return population


def getFitness(individual):
    cube = Cube(individual)
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
        coordinates = cubeMath.convertToPos(size2, x)
        if(coordinates > size1).any():
            retVals.append(x)

    return retVals


def mutation(individual, percent_mutation, seed=None, isSurvivor=False):
    chromosome_len = len(individual)
    base_size = int(numpy.rint(numpy.power(chromosome_len / 3, (1. / 3.))))
    if seed is None:
        for x in range(int(chromosome_len * percent_mutation)):
            gene_index = int(random.random() * chromosome_len)
            cell_index = gene_index // 3
            dim_index = gene_index % 3
            coordinates = cubeMath.convertToPos(base_size, cell_index)
            mutant_gene = cubeMath.selectPermutation(coordinates[dim_index])
            individual[gene_index] = mutant_gene 
            if not isSurvivor and not cubeMath.edgeCheck(cell_index):
                individual[gene_index] = random.choice([mutant_gene, None])
        return individual
    else:
        seed_size = int(numpy.rint(numpy.power(len(seed) / 3, (1. / 3.))))
        changeableCells = extendedCubeCells(seed_size, base_size)
        for x in range(int(len(changeableCells) * percent_mutation)):
            cell_index = random.choice(changeableCells)
            dim_index = random.choice(range(3))
            gene_index = cell_index * 3 + dim_index
            coordinates = cubeMath.convertToPos(base_size, cell_index)
            mutant_gene = cubeMath.selectPermutation(coordinates[dim_index])
            individual[gene_index] = mutant_gene
            if not isSurvivor and cubeMath.edgeCheck(cell_index):
                individual[gene_index] = random.choice([mutant_gene, None])
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


def fixDeadlocks(cube, gene):
    while not cube.locked:
        cube.iterate()

    new_gene = gene.copy()
    for x in cube.deadlocks:
        cell_0, cell_1 = x
        new_cube = Cube(new_gene)
        if not new_cube.cells[cell_0].isTrap() and not new_cube.cells[cell_1].isTrap():
            new_seed = Seed(None, 1.0, gene, cell_0)
            new_gene = new_seed.gene.copy()
    if hash(str(gene)) != hash(str(new_gene)):
        return new_gene
    else:
        return None


def generateGeneration(populationSize, fitness, deadlocked, seed=None):
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
            child = mutation(child, 0.25, seed)
            population.append(child.copy())

    deadlockCount = 0

    maxDeadlocks = populationSize - len(population)
    if maxDeadlocks == 0:
        return population
    for x in deadlocked:
        deadlockCount += 1
        print("Deadlock(s) processed: {0} out of {1}".format(deadlockCount, maxDeadlocks), end="\r")
        cube = Cube(x)
        corrected_individual = fixDeadlocks(cube, x)
        if len(population) < populationSize and corrected_individual is not None:
            population.append(corrected_individual)
        if len(population) == populationSize:
            break
    if deadlockCount > 0:
        print("Deadlock(s) processed: {0} out of {1}".format(deadlockCount, maxDeadlocks))

    if(len(population) < populationSize):
        if seed is None:
            random_children = generatePopulation(populationSize - len(population))
        else:
            cube = cube.Cube(seed[1])
            if len(survivors) == 0 or average_fitness == -limit:
                random_children = generatePopulation(populationSize - len(population), cube, 1.0)
            else:
                random_children = generatePopulation(populationSize - len(population), cube)
        population.extend(random_children)
    return population


def evolve(populationSize, totalGenerations, saveDir, seeds=None, seed=None):
    cubeMath.generatePermutationDict()
    if seeds is None:
        if seed is None:
            population = generatePopulation(populationSize)
        else:
            cube = Cube(seed[1])
            population = generatePopulation(populationSize, cube, 1.0)
    else:
        population = generateGeneration(populationSize, seeds, [])

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

    for generation in range(totalGenerations):
        print("Current Generation: {0} out of {1}".format(generation + 1, totalGenerations))
        if(generation > 0):
            if seed is None:
                population = generateGeneration(populationSize, fitness, deadlocked)
            else:
                population = generateGeneration(populationSize, fitness, deadlocked, seed)

        fitness = []
        deadlocked = []
        for individual in population:
            if(hash(str(individual)) in genePool.keys()):
                fitness_score = genePool[hash(str(individual))]
            else:
                fitness_score, iterations, numtraps, locked = getFitness(individual)
                if(fitness_score > average_fitness):
                    genePool.update({hash(str(individual)): fitness_score})
                if(fitness_score > -limit):
                    fitness_trapRatio.append(
                        [fitness_score, (numtraps / (globals.base_size ** 3))])
                    if fitness_score > 0:
                        fitness_iteration.append(
                            [fitness_score, iterations])
                        trapRatio_iteration.append(
                            [(numtraps / (globals.base_size ** 3)), iterations])
                        viable.append([fitness_score, individual, (numtraps / (globals.base_size ** 3)), iterations])
                if locked: 
                    deadlocked.append(individual)

            fitness.append([fitness_score, individual, (numtraps / (globals.base_size ** 3)), iterations])
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
        #matplotlib.pyplot.pause(0.10)
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


globals.base_size = int(sys.argv[1])
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
    powLamda = lambda y: x ** y
    primePower = powLamda(2)
    count = 2
    while(primePower < 1000):
        primePower = powLamda(count)
        count = count + 1
        if(primePower is not primePowers and primePower < 1000):
            primePowers.append(primePower)

globals.trapRooms = list(primeList)
globals.trapRooms.extend(primePowers)
globals.trapRooms.sort()

global limit
limit = int(numpy.power(10, int(numpy.log(globals.base_size ** 3))))

directory = str(globals.base_size)
if not os.path.isdir(directory):
    os.mkdir(directory)
files = []
files += [f for f in os.listdir(directory) if f.endswith('.pkl')]
files = [os.path.join(directory, i) for i in files]
files = sorted(files, key=os.path.getmtime, reverse=True)
fittest = []
if(len(files) > 0 and startWithSeed == 1):
    for file in files:
        with open(file, 'rb') as f:
            for i in dill.load(f):
                fittest.append(i)
    if len(fittest) > populationSize:
        fittest = fittest[0: populationSize - 1]
    fittest = evolve(populationSize,
                     total_generations, directory, fittest)
else:
    if(startWithSeed == 1):
        directory = str(globals.base_size - 1)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        files = []
        files += [f for f in os.listdir(directory) if f.endswith('.pkl')]
        files = [os.path.join(directory, i) for i in files]
        files = sorted(files, key=os.path.getmtime, reverse=True)
        if(len(files) > 0):
            for file in files:
                with open(file, 'rb') as f:
                    for i in dill.load(f):
                        fittest.append(i)

            directory = str(globals.base_size)
            average_fitness = calculateAverageFitness(fittest)
            seedVarities = []
            fittest = sorted(fittest, key=lambda x: x[2], reverse=False)
            for x in fittest:
                print_individual(x)
                seeds = evolve(int(populationSize / 10), total_generations, str(globals.base_size), None, x)
                seedVarities.extend(seeds)
                if len(seedVarities) > populationSize:
                    break
            seedVarities = sorted(
                seedVarities, key=lambda x: x[0], reverse=True)
            if len(seedVarities) > populationSize:
                seedVarities = seedVarities[0:populationSize - 1]
            fittest = evolve(populationSize, total_generations, directory, seedVarities)
        else:
            print("No seeds avaliable.")

    else:
        fittest = evolve(populationSize,
                         total_generations, directory)

if len(fittest) > 0:
    with open(os.path.join(directory, "fittest_{date:%Y%m%d_%H%M%S}.pkl".format(date=datetime.datetime.now())), 'wb') as f:
        dill.dump(list(fittest), f)

print("Done!")

print("# Viable: {0}".format(len(fittest)))

# if len(fittest) > 0:
#     print("Top Functional Cubes")
#     for x in fittest:
#         print_individual(x)

# print("Done!")
print("Top 10% Cubes")
for x in fittest[0:int(len(fittest) * 0.10)]:
    print_individual(x)

# permutations = generatePermutationDict(size)
# population = generatePopulation(permutations, size, populationSize)
# for individual in population:
#     cube = Cube(individual, permutations)
#     cube.analyze()
