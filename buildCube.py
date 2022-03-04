

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
import itertools

from seed import Seed
from cube import Cube

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

globals.base_size = int(sys.argv[1])

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
cubeMath.generatePermutationDict()

base_gene = [None] * ((globals.base_size ** 3) * 3)

viable_genes = [base_gene]

for cell in range(1, globals.base_size ** 3):
    new_genes = []
    cell_index = cell - 1
    [x, y, z] = cubeMath.convertToPos(globals.base_size, cell_index)

    x_values = cubeMath.validPermutations(x + 1)
    y_values = cubeMath.validPermutations(y + 1)
    z_values = cubeMath.validPermutations(z + 1)

    new_values = list(itertools.product(x_values, y_values, z_values, repeat=1))

    if not cubeMath.edgeCheck(cell_index):
        new_values.append((None, None, None))

    for base_gene in viable_genes:
        for values in new_values:
            new_gene = base_gene.copy()
            new_gene[cell_index * 3] = cubeMath.matchPermutation(x, cubeMath.convertPermutation(values[0])) 
            new_gene[cell_index * 3 + 1] = cubeMath.matchPermutation(y, cubeMath.convertPermutation(values[1])) 
            new_gene[cell_index * 3 + 2] = cubeMath.matchPermutation(z, cubeMath.convertPermutation(values[2])) 
            
            fitness_score, iterations, numtraps, locked = getFitness(new_gene)
            if not locked and fitness_score >= 0:
                new_genes.append(new_gene)
                print("Fitness: {0} Viable Gene: {1}\r".format(fitness_score, new_gene))
    viable_genes = new_genes
        