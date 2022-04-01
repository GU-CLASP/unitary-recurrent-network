import random
from collections import defaultdict
import functools

openClosePairs = [("(",")"),("{","}"),("[","]"),("<",">"),("+","-")]
openParens = list(o for o,c in openClosePairs)
closeParen = list(c for (_,c) in openClosePairs)

# openOf = {c: o }

# Picture dyck strings as path below diagonal on a grid.
# The size of the grid is N*N, where MAXLEN = 2*N

# Consider a point (i,j)
# i is the number of open
# j is the number of close
# so, i >= j
# depth of this point is i-j
# d=i-j
# i=j+d

def random_at_depth(N,d):
    i = random.randint(d,N)
    j = i-d # note that this is within [d,N]
    return (i,j)


def gen_path(p0,p1):
    (i0,j0) = p0
    (i1,j1) = p1
    if p0 == p1:
        return ""
    elif i0 == i1:
        return ")" * (j1-j0)
    elif j0 == j1:
        return "(" * (i1-i0)
    else:
        jMid = random.randint(j0,j1)
        iMid = random.randint(max(i0,jMid),i1)
        pMid = (iMid,jMid)
        return gen_path(p0,pMid) + gen_path(pMid,p1)


def gen_path_below(d,p0,p1):
    (i0,j0) = p0
    (i1,j1) = p1
    if p0 == p1:
        return ""
    elif i0 == i1:
        return ")" * (j1-j0)
    elif j0 == j1:
        return "(" * (i1-i0)
    else:
        jMid = random.randint(j0,j1)
        iMid = random.randint(max(i0,jMid),    # above diagonal
                              min(i1,d+jMid))  # no deeper than d
        # dMid = iMid-jMid    # by def
        # dMid <= d           # by assumption
        # iMid-jMid <= d      # subst.
        # iMid <= d + jMid
        pMid = (iMid,jMid) 
        return gen_path_below(d,p0,pMid) + gen_path_below(d,pMid,p1)



def gen_path(p0,p1):
    (i0,j0) = p0
    (i1,j1) = p1
    if p0 == p1:
        return ""
    elif i0 == i1:
        return ")" * (j1-j0)
    elif j0 == j1:
        return "(" * (i1-i0)
    else:
        jMid = random.randint(j0,j1)
        iMid = random.randint(max(i0,jMid),i1)
        pMid = (iMid,jMid)
        return gen_path(p0,pMid) + gen_path(pMid,p1)

def gen_single_type_string_any(N):
    return gen_path((0,0),(N,N))

def gen_single_type_string_min_depth(N,d):
    mid = random_at_depth(N,d)
    return gen_path((0,0),mid) + gen_path(mid,(N,N))

def gen_single_type_string_with_depth(N,d):
    mid = random_at_depth(N,d)
    return gen_path_below(d,(0,0),mid) + gen_path_below(d,mid,(N,N))

def randomize_parens(s):
    output = ""
    while s and s[0] == '(':
        s = s[1:]
        (o,c) = random.choice(openClosePairs)
        (randomized,s) = randomize_parens(s)
        s = s[1:]
        output += o+randomized+c
    return (output,s)

# max depth is equal to d
def gen_phrase_with_depth(N,d):
    return randomize_parens(gen_single_type_string_with_depth(N,d))[0]

# max depth is completely random
def gen_phrase_with_any_depth(N):
    return randomize_parens(gen_single_type_string_any(N))[0]



def updateMaxDepth(depths):
    return [max(d0,d1) for (d0,d1) in zip (depths, range(len(depths)))]

# measure max nesting between closing and opening parentheses
def parenDepths(x):
    result = [0] * len(x)
    depths = []
    for j in range(len(x)):
        c = x[j]
        if c in closeParen:
            curDepth = depths[0]
            result[j] = curDepth
            depths = depths[1:]
        else:
            depths = updateMaxDepth([0] + depths)
    return result

# measure distance from opening parentheses
def parenLengths(x):
    result = [0] * len(x)
    lens = [] # stack of lengths
    for j in range(len(x)):
        c = x[j] # read character
        if c in closeParen:
            result[j] = lens[0] # 
            lens = lens[1:] # pop
        else:
            lens = [0] + lens # push distance 0
        lens = [l+1 for l in lens] # increment all distances
    return result

def countAttractors(x):
    result = [0] * len(x)
    attrac = [] # stack of opening parens and number of attractors
    for j in range(len(x)):
        c = x[j]
        if c in closeParen:
            result[j] = attrac[0][1] # record result
            attrac = attrac[1:] # pop it
        else:
            attrac = [(c,0)] + attrac # push
        if c in openParens:
            attrac = list((o,l+1 if o != c else l) for (o,l) in attrac) 
    return result


def printResults(correct,total):
    print (sum(correct.values()), "correct predictions")
    print("data [set=r]{")
    print ("metric","correct","total","accuracy", sep=", ")
    for j in total:
        if total[j] > 0:
            print (j,correct[j],total[j],float(correct[j]) / total[j], sep=", ")
    print("}")


def commonEvaluator(MAXLEN,ys,accurs):
    metrics = {"nesting":parenDepths, "distance":parenLengths, "position":lambda s: list(range(len(s))),
               "attractors":countAttractors}
    correct = defaultdict(dict)
    total = defaultdict(dict)
    for m in metrics:
        correct[m] = dict((i,0) for i in range(MAXLEN))
        total[m] = dict((i,0) for i in range(MAXLEN))
    for (accur,x) in zip(accurs,ys):
        v = dict((m,metrics[m](x)) for m in metrics) # computaion of the metrics
        for j in range(len(x)):
            if accur[j] is not None: # this was a closing paren
                for m in metrics:
                    k = v[m][j] # value of the metric at position j
                    correct[m][k] += accur[j]
                    total[m][k] += 1
    for m in metrics:
        values = dict()
        print(m)
        printResults(correct[m],total[m])
    return False


def test_paren_length_depth():
  for i in range(100):
    x = gen_phrase_with_depth(10, 6)
    print(x)
    print(parenLengths(x))
    print(parenDepths(x))

def has_max_paren_dist(maxlen, x):
    ls = parenLengths(x)
    return functools.reduce(lambda x,y: x and y,[l <= maxlen for l in ls])

def generate_examples_by_max_dist(N,max_paren_dist,n):
    while n > 0:
        x = gen_phrase_with_any_depth(N)
        if has_max_paren_dist(max_paren_dist,x):
            yield(x)
            n-=1


def test_max_len():
  n = 0
  for i in range(100):
    x = gen_phrase_with_any_depth(10)
    if has_max_paren_dist(10,x):
        n+=1
        print(x)
  print(n)



if __name__ == "__main__":
    for x in generate_examples_by_max_dist(20,10,100):
        print (x)
