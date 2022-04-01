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
from collections import Counter

MAXLEN=21

tyf.cuda_use_one_free_device()

random.seed(1234)


def my_sample(l,n):
    return list(random.sample(l,min(n,len(l))))


def generateExample(len):
    lPart = len // 2
    m = random.randrange(1,lPart)
    n = lPart - m
    # print("gen:", len, lPart, m, n)
    return [1] * m + [2] * n + [3] * m + [4] * n

def pad(ws): return (ws + [0] * (MAXLEN - len(ws)))

def generate_examples(mlen,n):
    return [generateExample(random.randrange(4,mlen)) for i in range(n)]

def make_input(examples):
    return np.array([pad([5]+seq) for seq in examples])

def make_output(examples):
    return np.array([pad(seq) for seq in examples])

def make_weights(seq):
    l = len(seq)
    assert l <= MAXLEN
    return [float(1)] * (len(seq)+1) + [float(0)] * (MAXLEN - len(seq) - 1)

def make_examples(l):
    l = list(l)
    return {"x":make_input(l),
            "y":make_output(l),
            "weights":np.array([make_weights(seq) for seq in l]),
            "len":np.array([len(x) for x in l])}

print("Generation of train sentences...")
train_sentences = make_examples(generate_examples(10,512*100)) # kinda stupid: there are much fewer cases than generated examples
print("Generation of val sentences...")
val_sentences = make_examples(generate_examples(20,512*10))
test_sentences = val_sentences
# val_sentences = make_examples(generate_examples_deep(5,512*40))
# val_sentences = make_examples(generator2.generate_examples_by_max_dist(N,20,512*10))

print ("Number of train sentences = ", len(train_sentences["x"]))
print ("Number of val sentences = ", len(val_sentences["x"]))

print ("Ex:",train_sentences["x"][0], train_sentences["y"][0], train_sentences["weights"][0])

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

def isCorrect(xs,ys,y_s):
  state = 1
  for (i,y) in enumerate(ys):
      y_ = y_s[i]
      x = xs[i]
      # print ("xs=",xs)
      # print ("x=",x)
      # Done
      if y == 0: 
          return y_ == 0
      # informing: can predict same or next state.
      if (x == 1) or (x == 2):
          if y_ < x:
              return False
          if y_ > x+1:
              return False
      # informed: must be predicted exactly because the counts are known by now.
      if (x == 3) or (x == 4):
          if (y_ != y):
              return False
  return True
          


def eval_cb(values):
    preds = tyf.predict(model,lm.runModel,test_sentences)
    tot = 0
    correct = 0
    for (i,y) in enumerate(test_sentences["y"]):
        x = test_sentences["x"][i]
        y_ = np.argmax(preds[i], axis = 1)
        yCount = Counter(y.tolist())
        y_Count = Counter(y_.tolist())
        y3 = yCount[3]
        y4 = yCount[4]
        y_3 = y_Count[3]
        y_4 = y_Count[4]
        tot += 1
        ok = isCorrect(x,y,y_)
        # if not ok: print(y,y_,ok)
        if ok:
            correct += 1
    print("Accuracy:", float(correct) / tot )

nepochs = 100
optimizer = tf.keras.optimizers.Adam()
train_stats = tyf.train(optimizer,
                        model, lm.runModel,
                        dict_generator(train_sentences),
                        valid_generator = dict_generator(val_sentences),
                        epochs=nepochs,
                        callbacks=[eval_cb])

for t in train_stats:
    print (t)

# bestEpoch = max(train_stats, key=lambda e: e["val"]["loss"]) # TODO: move "numpy" to rts

# print(bestEpoch)





# beam_search(sess,model,15,char_indices["s"])
