import TypedFlow.typedflow_rts as tyf
import tensorflow as tf
import numpy as np

import importlib
import sys
modelkind = sys.argv[1]
lm = importlib.import_module(modelkind)
import os
import math
import random
import csv
import itertools

tyf.cuda_use_one_free_device()

random.seed(1234)

MAXLEN=50


print("Loading vocabulary")
vocab = [l.rstrip("\n") for l in open("data/DICT").readlines()];
word_idx = {word:idx+1 for (idx,word) in enumerate(vocab)}


def deps_from_tsv(infile):
    return csv.DictReader(open(infile), delimiter='\t')

def pad(x):
    return (x + [0] * (MAXLEN-len(x)))


# create an example from input line as dict:
def create_example(d):
    d = dict(d)
    s = d["sentence"].split()
    vi = int(d["verb_index"]) - 1
    # print(d["verb"],s[vi])
    return {"x"      : pad(list(word_idx[w] for w in s)),
            "yIndex" : vi-1,
            "y"      : d["verb_pos"] in ["VBZ"],
            "distractors" : int(d["n_diff_intervening"])
            }

def create_examples(ds):
    return list(map(create_example,ds))



print("Loading input data")
input_data = deps_from_tsv("data/agr_50_mostcommon_10K.tsv")
all_examples = create_examples(input_data)
# all_examples = itertools.islice(all_examples,1000) # for debugging (maybe).
all_examples = list(all_examples) # actually load, so that we can compute length, etc.

print("Preparing train/val data")

split_index = len(all_examples)//10
print("Validation sample size=",split_index)
random.shuffle(all_examples)

def transposeDicts(dicts):
    return dict((k, list(d[k] for d in dicts)) for k in dicts[0])

val_examples = all_examples[:split_index]
val_examples_transposed = transposeDicts(val_examples)
train_examples = all_examples[split_index:]


print("Loading model")
model = lm.mkModel()

phs = lm.runModel["placeholders"]
types = {k: phs[k]["dtype"] for k in phs}

def dict_generator (xs,types):
    total_len = len(xs)

    def gen(bs):
      for i in range(0, bs*(total_len//bs), bs):
          # print(dict((k,xs[k][i:i+bs]) for k in xs))
          yield dict((k,tf.cast(list(x[k] for x in xs[i:i+bs]), types[k])) for k in types) 
    return gen

def eval_cb(values):
    tyf.save(model, modelkind + ".npz")
    print("Predicting ...")
    cast_input = dict((k,tf.cast(val_examples_transposed[k], types[k])) for k in types)
    preds = tyf.predict(model, lm.runModel, cast_input)
    print("Done predicting")
    # print(preds.shape)
    tot = [0] * MAXLEN
    ok  = [0] * MAXLEN
    baseline  = [0] * MAXLEN
    print("Compiling predictions")
    for i in range(split_index):
        e = val_examples[i]
        ndist = e["distractors"]
        tot[ndist] += 1
        if e["y"]:
            baseline[ndist] += 1
        if (preds[i][1] >= 0.5) == e["y"]:
            ok[ndist] += 1
    print("Done compiling")
    print ("attractors", "accuracy", "tests", "errLo","errHi", "baseline", sep = ", ")
    for j in range(MAXLEN):
        if tot[j] > 0:
            n = tot[j]
            ns = ok[j]
            nf = n - ns
            p = float(ns)/n
            z = 1.96
            pCenter = (ns + z*z/2) / (n+z*z)
            pW = z / (n+z*z) * math.sqrt (float(ns*nf)/n + z*z/4)
            pLo = pCenter - pW
            pHi = pCenter + pW
            print (j, p, n, p-pLo, pHi-p, float(baseline[j])/n, sep = ", ")


nepochs = 10
optimizer = tf.keras.optimizers.Adam()
train_stats = tyf.train(optimizer,
                        model, lm.runModel,
                        dict_generator(train_examples,types),
                        valid_generator = dict_generator(val_examples,types),
                        epochs=nepochs,
                        callbacks=[eval_cb])

