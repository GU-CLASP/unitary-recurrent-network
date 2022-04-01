import TypedFlow.typedflow_rts as tyf
import tensorflow as tf
import numpy as np
import importlib
import sys
modelkind =sys.argv[1]
lm = importlib.import_module(modelkind)
import os
import math
import random
import generator2
from dyck_utils import *

tyf.cuda_use_one_free_device()

random.seed(1234)




def my_sample(l,n):
    return list(random.sample(l,min(n,len(l))))


def generate_examples(d,n):
    return [generator2.gen_phrase_with_depth(N,d) for i in range(n)]


def rand_oc(l):
    while True:
        os = ""
        cs = ""
        for _ in range(l):
            (o,c) = random.choice(openClosePairs)
            os = o + os
            cs = cs + c
        yield (os,cs)

# total N = 10
def generate_examples_deep(d,n):
    return [o + s + c
            for (s,(o,c)) in zip([generator2.gen_phrase_with_any_depth(N-d) for i in range(n)],rand_oc(d))]


print("Generation of train sentences...")
train_sentences = make_examples(generate_examples(3,512*200))
# train_sentences = make_examples(generator2.generate_examples_by_max_dist(N,10,512*20))
print("Generation of val sentences...")
# val_sentences = make_examples(generate_examples(9,512*40))
# val_sentences = make_examples(generate_examples_deep(5,512*40))
val_sentences = make_examples(generator2.generate_examples_by_max_dist(N,20,512*10))

print ("Number of train sentences = ", len(train_sentences["x"]))
print ("Number of val sentences = ", len(val_sentences["x"]))

# print ("Ex:",train_sentences["x"][0], train_sentences["weights"][0])

# sess = tf.Session()
print("Loading model")
model = lm.mkModel()

def dict_generator (xs):
    k0 = next (iter (xs.keys())) # at least one key is needed
    total_len = len(xs[k0])

    def gen(bs):
      for i in range(0, bs*(total_len//bs), bs):
          # print(dict((k,xs[k][i:i+bs]) for k in xs))
          yield dict((k,xs[k][i:i+bs]) for k in xs)
    return gen

allResults = dict((i,0) for i in range(1000))

# def printResults(values):
#     correct = values["accuracy"]
#     total = values["total"]
#     print("data [set=r]{")
#     print ("metric","total","accuracy", sep=", ")
#     for j in range(MAXLEN):
#         if total[j] > 0:
#             print (j,correct[j],total[j],float(correct[j]) / total[j], sep=", ")
#     print("}")

def curDepths(n):
    return range(n)

def updateMaxDepth(depths):
    return [max(d0,d1) for (d0,d1) in zip (depths, curDepths(len(depths)))]




def eval_cb(values):
    tyf.save(model, modelkind + ".npz")

    preds = tyf.predict(model,lm.runModel,val_sentences)
    print(preds.shape)
    accurs = [computeAccurate(x,preds[i]) for (i,x) in enumerate(val_sentences["y"])]
    ys = [decode(y) for y in val_sentences["y"]] # strings with actual parens
    generator2.commonEvaluator(MAXLEN,ys,accurs)

nepochs = 100
optimizer = tf.keras.optimizers.Adam()
train_stats = tyf.train(optimizer,
                        model, lm.runModel,
                        dict_generator(train_sentences),
                        valid_generator = dict_generator(val_sentences),
                        epochs=nepochs,
                        callbacks=[eval_cb])

# print(train_stats)
bestEpoch = max(train_stats, key=lambda e: e["val"]["loss"]) # TODO: move "numpy" to rts

print(bestEpoch)





# beam_search(sess,model,15,char_indices["s"])
