import dill
import itertools
import os


def findCominations(number, permutations):
    returnList = []
    for perm in permutations:
        if(perm[0] + perm[1] + perm[2] == number):
            returnList.append(perm)
    return returnList


def generatePermutationDict(size):
    perms = dict.fromkeys(range(1, size + 2))
    for x in perms.keys():
        perms[x] = findCominations(
            x, list(itertools.product(range(10), repeat=3)))
    return perms


def getPower(n):
    return lambda x: n ** x


def main():
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

    trapRooms = list(primeList)
    trapRooms.extend(primePowers)
    trapRooms.sort()

    with open(os.path.join('.', "traps.pkl"), 'wb') as f:
        dill.dump(list(trapRooms), f)


    permutations = generatePermutationDict(26)
    with open(os.path.join('.', "permutations.pkl"), 'wb') as f:
        dill.dump(permutations, f)


if __name__ == '__main__':
    main()
