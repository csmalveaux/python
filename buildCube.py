

def convertPermutation(permutation):
    return permutation[0] * 100 + permutation[1] * 10 + permutation[2]

def matchPermutation(permutations, position, code):
    pos = position.copy() + 1
    perm = permutations[pos].copy()
    for x in list(range(len(perm))):
        if convertPermutation(perm[x]) == code:
            return x
    return -1

def translatePermutation(permutation, old_pos, new_pos, code):
	perms = permutation[old_pos + 1].copy()
	select_perm = perms[matchPermutation(permutation, old_pos, code)]
	deltas = []