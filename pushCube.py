from TwitterAPI import TwitterAPI
import itertools
import datetime
import random
import numpy
import dill
import sys
import os
import threading
import queue
import time


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
            if loc < 0 or loc >= (size + 2):
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

    def __init__(self, size, permutations, cube=None, density=None, gene=None, cell=None):
        self.gene = []
        if cube is None:
            if gene is None:
                for x in range(size ** 3):
                    coordinates = convertToPos(size, x)
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
                coordinates = convertToPos(size, x)
                if(coordinates < originSize).all():
                    position = convertToCell(originSize, coordinates)
                    code = cube.cells[position].code
                    for dim in range(3):
                        self.gene.append(matchPermutation(
                            permutations, coordinates[dim], code[dim]))
                else:
                    coordinates = convertToPos(size, x)
                    if density == 1.0:
                        # cannotTrap = [False, False, False]
                        for dim in range(3):
                            value = selectPermutation(
                                permutations, size, coordinates[dim], density)
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


def getFitness(individual, permutations):
    cube = Cube(individual, permutations)
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


def print_individual(individual):
    print("\tGenome: {0}".format(individual[1]))
    print("\tScore: {0}".format(individual[0]))
    print("\tDensity: {0}".format(individual[2]))
    print("\tCycles: {0}".format(individual[3]), end='\n\n')


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
        new_cube = Cube(new_gene, permutations)
        if not new_cube.cells[cell_0].isTrap() and not new_cube.cells[cell_1].isTrap():
            rand = random.SystemRandom()
            new_seed = Seed(cube.size, permutations, None,
                            1.0, gene, rand.choice(list(x)))
            new_gene = new_seed.gene.copy()
    if hash(str(gene)) != hash(str(new_gene)):
        return new_gene
    else:
        return None


def generateCubes(baseSize):
    workQueue = queue.Queue()
    threads = []
    global stack_hash
    global viable
    global last_posttime
    global api
    viable = []
    stack_hash = {}
    last_posttime = datetime.datetime.now()
    global totalCount

    totalCount = 0

    for x in range(10):
        thread = Thread(workQueue)
        thread.start()
        threads.append(thread)

    consumer_key = 'wF8ZKWtg4g7YSZYUS9K3pXv6x'
    consumer_secret = '5JDlb9SpgxP2IDH84G83db1hRCuj2ChaENuMsWZdVXYKQShPNK'
    access_token_key = '1583661984-uqYMpEIs14CdtgQoskcc4bEKSoIkfh0Rolu9Kl5'
    access_token_secret = 'v6MZYsKLAT0nC90N2ZfqAam6wejB8hvpVXU9kfrWbp6vg'
    api = TwitterAPI(consumer_key, consumer_secret,
                     access_token_key, access_token_secret)
    global permutations
    permutations = generatePermutationDict(26)

    print("Initializing random cube:")
    random_cube = Seed(size, permutations)

    workQueue.put(random_cube.gene)
    exitFlag = 0
    while not workQueue.empty():
        pass

    exitFlag = 1

    for th in threads:
        th.join()

    try:
        r = api.request('statuses/update', {'status': 'Completed'})
    except:
        print(r.status_code)
    return viable


class Thread (threading.Thread):
    def __init__(self, q):
        threading.Thread.__init__(self)
        self.q = q

    def run(self):
        while not exitFlag:
            queueLock.acquire()
            if not self.q.empty():
                queueLock.release()
                process_data(self.q)
            else:
                queueLock.release()


def process_data(q):
    while not exitFlag:
        queueLock.acquire()
        if not q.empty():
            curr_seed = q.get()
            global viable
            global stack_hash
            global totalCount
            global last_posttime
            global permutations
            global api
            totalCount = totalCount + 1
            stack_size = q.qsize()
            queueLock.release()
            print("Stack size: {0}".format(stack_size))
            queueLock.acquire()
            if hash(str(curr_seed)) not in stack_hash.keys():
                stack_hash.update({hash(str(curr_seed)): totalCount})
                queueLock.release()
                cube = Cube(curr_seed, permutations)
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

                if not cube.locked and iterations > 1 and iterations < limit:
                    if points > 0:
                        score = iterations * (1 - (traps / totalCells))
                        queueLock.acquire()
                        viable.append(
                            [score, curr_seed, (traps / (cube.size ** 3)), iterations])
                        queueLock.release()
                        try:
                            r = api.request(
                                'statuses/update', {'status': "v:{0}".format(hash(str(curr_seed)))})
                        except:
                            print(r.status_code)
                    else:
                        blocklist = []
                        for x in cube.cells.keys():
                            blocklist.append([x, cube.cells[x].blocked])

                        blocklist = sorted(
                            blocklist, key=lambda x: x[1], reverse=True)
                        mostblocked = blocklist[0]
                        new_seed = Seed(cube.size, permutations,
                                        None, 1.0, curr_seed, mostblocked[0])
                        altered_gene = new_seed.gene
                        queueLock.acquire()
                        q.put(altered_gene)
                        queueLock.release()
                else:
                    if iterations < limit and cube.locked:
                        deadlockList = cube.deadlocks
                        for pair in deadlockList:
                            new_seed = Seed(
                                cube.size, permutations, None, 1.0, curr_seed, pair[0])
                            altered_gene = new_seed.gene
                            queueLock.acquire()
                            q.put(altered_gene)
                            queueLock.release()

                            new_seed = Seed(
                                cube.size, permutations, None, 1.0, curr_seed, pair[1])
                            altered_gene = new_seed.gene
                            queueLock.acquire()
                            q.put(altered_gene)
                            queueLock.release()
                    else:
                        print("spins off into infinity")
            else:
                queueLock.release()
        else:
            queueLock.release()

        curr_time = datetime.datetime.now()
        queueLock.acquire()
        timedelta = curr_time - last_posttime
        queueLock.release()
        if timedelta >= datetime.timedelta(seconds=3600):
            queueLock.acquire()
            try:
                r = api.request(
                    'statuses/update', {'status': "p:{0}s:{1}".format(totalCount, q.qsize())})
            except:
                print(r.status_code)
            last_posttime = datetime.datetime.now()
            queueLock.release()


size = int(sys.argv[1])

global viable
global stack_hash
global totalCount
global last_posttime
global queueLock
global permutations
queueLock = threading.Lock()

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

fittest = generateCubes(size)
if len(fittest) > 0:
    with open(os.path.join(directory, "fittest_{date:%Y%m%d_%H%M%S}.pkl".format(date=datetime.datetime.now())), 'wb') as f:
        dill.dump(list(fittest), f)

print("Done!")

if len(fittest) > 0:
    print("Top Functional Cubes")
    for x in fittest:
        print_individual(x)
