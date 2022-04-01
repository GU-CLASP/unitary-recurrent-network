
import TypedFlow.typedflow_rts as tyf
import tensorflow as tf
import numpy as np
import latex
import importlib
import sys
modelkind = sys.argv[1]
lm = importlib.import_module(modelkind)
import os
import math
import cmath
import random
import itertools
import scipy as sp
import functools
import generator2
import dyck_utils

imag = complex(0,1)
one = complex(1,0)

magic = (1/np.sqrt (2)) * np.matrix([[1,    1],
                                    [-imag, imag]])

# inverse of magic
imagic = np.sqrt(1/2) * np.matrix([[1,  imag],
                                   [1, -imag]])



print("Loading vocabulary")
if modelkind == "orn_classifier":
    vocab = [l.rstrip("\n") for l in open("data/DICT").readlines()];
    MAXLEN = 50
    meaning = {0: "plural", 1: "singular"}
    word_idx = {word:idx+1 for (idx,word) in enumerate(vocab)}
    projname = "dense"
else:
    openClosePairs = [("(",")"),("{","}"),("[","]"),("<",">"),("+","-")]
    vocab = sorted([" ","s"] + [p for oc in openClosePairs for p in oc])
    N = 10
    MAXLEN = 2*N+1
    word_idx = {word:idx for (idx,word) in enumerate(vocab)}
    meaning = {idx:word for (idx,word) in enumerate(vocab)}
    projname = "projection"

def pad(x):
    return (x + [0] * (MAXLEN-len(x)))
    
# print (word_idx) # debug

parametersfile = sys.argv[2]

print("Loading model")
model = lm.mkModel()

tyf.load(model, parametersfile)

def coplanarity(a,b):
    # (_,sigma,_) = np.linalg.svd(np.matrix(np.concatenate([a,b])))
    # return sigma

    # this is another procrustes problem
    (_,sigma,_) = np.linalg.svd(a @ b.T)
    return np.sum(sigma)
    

# analyse_coplanarity([[1,0,0,0], [0,1,0,0],[1,0,0,0]] )



def analyse_antihermitian(x):
    # print(x)
    (u,sigma,vh) = np.linalg.svd(x)
    sh = np.shape(sigma)
    n = sh[0]
    b = np.diag([1,-1]*(n//2)) # see theory of antisymmetric matrices
    print("Rotation angles:",sigma[::2]) # [::2] to avoid redundant data
    # print("Rotation planes given by:",np.matmul(vh.T,b))


def get_antihermitian_embs(ws):
     return list(x.numpy() for x in tyf.evaluate(model,lm.probeEmbs,{"wordIdx": [word_idx[w] for w in ws] },result="embsAntiHermitian"))

def get_antihermitian_emb(w):
     return get_antihermitian_embs([w])[0]

def get_unitary_embs(ws):
    return list(sp.linalg.expm(x) for x in get_antihermitian_embs(ws))

def get_unitary_emb(w):
    return get_unitary_embs([w])[0]

def get_unitary_phrase_emb(s):
    xs = get_unitary_embs(s.split())
    xs.reverse() # god help us
    return mat_product(xs)

def mat_similarity(a,b):
    # = trace(a.T @ b)
    return np.dot(a.flatten(), b.flatten())

def mat_dist(a,b):
    return sq_frob(a-b)

def sq_frob(x):
    return np.square(np.linalg.norm(x))

# def analyse_phrase(s):
#     print("Analysis of phrase ", s)
#     analyse_unitary(get_unitary_phrase_emb(s))

def mat_effect(x):
    sh = np.shape(x)
    n = sh[0]
    return (2 * (n - np.trace(x)))

def analyse_agr_embedding_latex(x):
    return [latex.fmt(mat_effect(x)), latex.small_matrix(proj_on_proj(x))]

def analyse_agr_embedding_latex_full(x):
    return [latex.fmt(mat_effect(x)), latex.angles(eigen_angles(x)), latex.small_matrix(proj_on_proj(x))]

def analyse_agr_embedding(x):
    return [mat_effect(x), proj_on_proj(x)]

def analyse_dyck_embedding(x):
    return [mat_effect(x), eigen_angles(x)]

def analyse_phrases(ss, features, tex_table=False,sorting=True):
    xs = list((get_unitary_phrase_emb(s), s) for s in ss)
    if sorting:
        xs.sort(key=lambda x: mat_effect(x[0]))
    res = list([[s] + features(x)  for x,s in xs])
    if tex_table:
        print(latex.rows(res))
    else:
        for info in res:
            print (*info)

def analyse_dyck_phrases_tex(ss):
    xs = list((get_unitary_phrase_emb(s), s) for s in ss)
    res = list([[latex.verb(s),latex.fmt(mat_effect(x)), latex.angles(eigen_angles(x))]  for x,s in xs])
    print(latex.rows(res))


def analyse_phrase_pairs(ss):
    xs = list(map(get_unitary_phrase_emb,ss)) # list call necessary (Guido est un con)
    print ("", "", *ss, sep=" | ")
    for i,x in enumerate(xs):
        print("",ss[i], *[latex.fmt(mat_dist(x,y)) for y in xs], sep=" | ")


def eigenvalues(a):
    return np.linalg.eigvals(a)[::2] # skip every other eigen value: λᵢ and λᵢ₊₁ are complex conjugates.

def rotation_rank(a):
    return len(eigen_angles(a))

def rotation_planes(a, truncation=None):
    sh = np.shape(a)
    n = sh[0]

    l, q = np.linalg.eig(a)
    q = np.matrix(q) # otherwise, the hermitian conjugate is not working ??!
    # we have:
    #  a = q @ diag(l) @ q.H
    #                        multiply by  q on the right
    #  a @ q = q.H @ diag(l)
    #                        diagonal matrices always commute
    #  a @ q = diag(l) @ q.H 
    r = np.matrix(sp.linalg.block_diag(*list([magic] * (n//2)))) 
    p = np.real(r @ q.H)

    # Sanity checks:
    # diagl = np.diag(l)
    # rinv  = np.matrix(sp.linalg.block_diag(*list([imagic] * (n//2)))) 
    # blockdiag = np.real(rinv.H @ diagl @ rinv)
    # print("blockdiag",blockdiag)
    # print("meth1=", np.real (q @ diagl @ q.H)  )
    # print("meth2=", a )
    # print("meth3=", p.T @ blockdiag @ p )
    
    if truncation is None:
        truncation=rotation_rank(a)
    p = p[:(2*truncation)]
    return p

def dist_to_id(a):
    sh = np.shape(a)
    n = sh[0]
    return np.square(np.linalg.norm(a - np.identity(n)))

    
def eigen_angles(a):
    return np.array(list(a for a in np.angle(eigenvalues(a), deg=True).tolist()
                         if abs(a) > 0.01))
    # re-convert to numpy so that we get the precision desired when printing


def coplanarity_analysis(a,b):
    ra = rotation_rank(a)
    rb = rotation_rank(b)
    pa = rotation_planes(a)
    pb = rotation_planes(b)
    
    # print("Coplanarity analysis for angle sets", eigen_angles(a), eigen_angles(b))
    print("2=equal planes 0=orthogonal planes")
    
    coplanarity_matrix = np.matrix([[coplanarity(pa[2*i:2*(i+1)], pb[2*j:2*(j+1)])
                                     for i in range(ra)]
                                    for j in range(rb)])
    print(coplanarity_matrix)
    

def analyse_mat_pair(a,b):
    print("Distance", mat_dist(a,b))
    print("Respective angles", eigen_angles(a),eigen_angles(b))
    coplanarity_analysis(a,b)
    # print("Similarity" if asPair else "Trace: ", np.trace(x))
    # print("Eigen Angles: ", eigen_angles(x))


def closest_mat(a, bs, names):
    bdns = list([(mat_similarity(a,b),b,name) for b,name in zip (bs,names)])
    bdns.sort()
    for (dist, b, name) in bdns:
        print(dist, name)        

def analyse_closest(reference, candidates, opp=False):
    print("Which is closest to" + (" opposite of" if opp else ""),reference, "among", candidates)
    ref = get_unitary_phrase_emb(reference)
    if opp:
        ref = ref.T
    closest_mat(ref, map(get_unitary_phrase_emb, candidates), candidates)
    

def mat_product(xs):
    return functools.reduce(np.matmul, xs)


def analyse_emb_pair(s1,s2):
    print("Analysing pair", s1, s2)
    a = get_unitary_phrase_emb(s1)
    b = get_unitary_phrase_emb(s2)
    analyse_mat_pair(a,b)


def analyse_predsVec(ws):
    print("Prediction analysis of ",ws)
    s = pad([word_idx[w] for w in ws])
    # print("Sentence encoding:",s)
    ps = tyf.evaluate(model,lm.probePreds, {"x" : [s]}, result="pred")[0].numpy()
    print("Evaluation done")
    for i,w in enumerate(ws):
        print("After", w, {meaning[k]: ps[i][k] for k in meaning})

def analyse_preds(ws):
    print("Prediction analysis of ",ws)
    s = pad([word_idx[w] for w in ws])
    # print("Sentence encoding:",s)
    ps = tyf.evaluate(model,lm.probePreds, {"x" : [s]}, result="y")[0].numpy()
    print("Evaluation done")
    for i,w in enumerate(ws):
        print("After", w, meaning[ps[i]],sep="\t")

def analyse_dense(ws=None):
    print("Dense layer analysis")

    biases = model["paramsdict"][projname + "_bias"].numpy()
    print(biases)

    p = model["paramsdict"][projname + "_w"].numpy()

    if ws:
        relevant,relevant_words = zip(*list((word_idx[w],w) for w in ws))
        print("Projection analysis for ", relevant)
        p = np.matrix (list (p[:,k] for k in relevant)).T

    print("p-shape",np.shape(p))
    print(p)
    print("Gram matrix:", p.T @ p, sep="\n")
    print("p-norm", np.linalg.norm(p, axis=0))
    (u,sigma,vh) = np.linalg.svd(p, full_matrices=False)
    print("State-Feature matrix ", u)
    print("Feature-Prediction ", vh)
    print("Correlation ", sigma)


# def analyse_proj(ws):
#     relevant,relevant_words = zip(*list((word_idx[w],w) for w in ws))
#     print("Projection analysis for ", relevant)
#     p0 = model["paramsdict"]["projection"].numpy()
#     print (np.shape(p0))
#     p = np.matrix (list (p0[:,k] for k in relevant)).T
#     for k,w in enumerate(relevant_words):
#         print("Projection vector for",w,"norm=",np.linalg.norm(p[:,k]))
#     print("Gram matrix:", p.T @ p, sep="\n")

    # (u,sigma,vh) = np.linalg.svd(p, full_matrices=False)
    # print("State-Feature matrix", u, sep='\n')
    # print("Singular values:", sigma, sep='\n')
    # for k,w in enumerate(relevant_words):
    #     print("Feature vector for",w,"is",vh[k])


    

def proj_on_proj(q):
    p = get_proj_plane()
    # print("dbg:", np.shape(q), np.shape(p))
    return p.T @ q @ p

def get_proj_plane():
    if modelkind == "orn_classifier":
        vecs = model["paramsdict"]["dense_w"].numpy()
        (u,_,_) = np.linalg.svd(vecs, full_matrices=False)
        return u
    else:
        1/0

def dyck_test():
    print("Generation of test sentences...")
    val_sentences = dyck_utils.make_examples(generator2.generate_examples_by_max_dist(N,20,512*10))
    print("Predictions...")
    preds = tyf.predict(model,lm.runModel,val_sentences)
    print("Computing accurate...")
    accurs = [dyck_utils.computeAccurate(x,preds[i]) for (i,x) in enumerate(val_sentences["y"])]
    ys = [dyck_utils.decode(y) for y in val_sentences["y"]] # strings with actual parens
    print("Evaluation")
    generator2.commonEvaluator(MAXLEN,ys,accurs)
    

print("Analysis for", modelkind)

np.set_printoptions(precision=2,
                    suppress=True # don't use scientific notation if possible
                    )

if modelkind == "orn_classifier":
    frequent = ["the", ",", ".", "of", "and", "in", "a", "is", "to", "was", "for", "(", ")", "on", "as", "with", "by", "from", "at", "his", "it", "that", "an", "are", "be", "he", "i", "not", "which", "this", "has", "or", "were", "also", "but", "had", "their",  "who", "one", "been", "its", "have", "first", "they", ";", "other", "would", "can", "two", "her", "into", "after", "time", "when", "all", "article", "about", "some", "-", "you", "him", "she"]

    some_pos = [ "NNS", "NN", "VBP", "VBZ" ]
    sing_nouns = ["article", "year", "area", "world", "family", "city"]
    plur_nouns = ["articles", "years", "areas", "worlds", "families", "cities"]
    all_pos = [
      "or", "and", # "CC", Coordinating conjunction
      # "CD", Cardinal number
      # "DT", Determiner
      "the", "a", 
      # "EX", existential quant.

      # "FW", Foreign word
      # "IN", Conjunction
      # "JJ", adjective
      "small", "large",
      # "JJR", comparative
      "smaller", "larger",
      # "JJS", superlative
      # "LS", list item marker
      # "MD", modal ("could", "should", etc.)
      
      "NN", "NNP", "NNPS", "NNS",
      # "PDT", predeterminer
      # "POS", # possesive ending ??
      "my",  # "PRP", "my", etc.

      # "PRP$",
      # "RB", adverb
      # "RBR",
      # "RBS", most, etc.
      # "RP", particle
      # "SYM", symbol
      # "TO", "to"
        
      # "UH", interj
      # "VB", verb, base form
      # "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT",
                 # "WP$", this is not actually trained because only the word "whose" has this POS.
                 # "WP", similar
               
               "WRB"] 

    # analyse_phrases("the keys to cabinet are on table".split(), features=analyse_agr_embedding)
    # analyse_phrases(frequent[:-20],features=analyse_agr_embedding_latex,tex_table=True)
    analyse_phrases(["clever","stupid","american","former","the keys to the cabinet","the keys", "the key", "him that", "them that"],
                    features=analyse_agr_embedding_latex_full,tex_table=True)
    # (some_pos + frequent + ["students", "the students","the students enrolled in the program","clever"], features=analyse_agr_embedding)
    analyse_phrase_pairs(sing_nouns + plur_nouns)
    verbs = ["is", "has", "goes",  "are", "have",  "go"]

    # analyse_closest("JJ NN", all_pos)
    # analyse_closest("the NN", all_pos)
    # analyse_closest("the student", all_pos)
    # analyse_closest("NNS", all_pos)
    # analyse_closest("the students", all_pos)

    # analyse_closest("is good", all_pos, opp=True)
    
    # analyse_closest("article", all_pos)
    
    analyse_emb_pair("the student", "john")
    analyse_emb_pair("steve", "john")
    analyse_emb_pair("he", "john")
    analyse_emb_pair("he", "it")

    analyse_emb_pair("they", "he")
    analyse_emb_pair("they", "that")
    analyse_emb_pair("the stupid student", "the student")
    analyse_emb_pair("the students", "the students enrolled in the program that we completed")

    
    # analyse_preds("the students enrolled in the program submit a final project to complete the course".split())
    # analyse_preds("the student enrolled in the program completes a final project to pass the course".split())
    # analyse_preds("the student enrolled in the program that professors design completes a final project to pass the course".split())
    # analyse_preds("the apple that students share is good".split())
    # analyse_preds("the NN that NNS VBP is good".split())
    # analyse_preds("a 1962 time magazine article about NNP makes some points that help understand the context of the NN and JJ actions of all sides in NNP before the vietnam war".split())

    analyse_dense()
else:
    analyse_dense([")",">","}","]","-"])

    analyse_dyck_phrases_tex(["(", "<", "{", "[", "+", ")",">","}","]" ,"-"] +
                             ["( )", "< >", "{ }", "[ ]", "+ -"])
    # analyse_phrases(["( (", "( {","( [","( <", "( < [ { +"])

    analyse_phrase_pairs(["(", "<", "{", "[", "+"])
    
    # print("Mistmatch OPEN/OPEN")
    # analyse_phrases("{ (", "{ [", "( [")

    # print("Simple distance OPEN")
    analyse_emb_pair("(", "[")
    analyse_emb_pair("(", "{")
    analyse_emb_pair("(", "<")
    analyse_emb_pair("(", ")")

    # analyse_predsVec("s([{}{<>}{}])")
    # analyse_preds("s([{}{<>}{}])")

    dyck_test()
