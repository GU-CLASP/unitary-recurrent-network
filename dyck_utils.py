import numpy as np


openClosePairs = [("(",")"),("{","}"),("[","]"),("<",">"),("+","-")]
chars = sorted([" ","s"] + [p for oc in openClosePairs for p in oc])
closeParen = sorted([c for (_,c) in openClosePairs])

# char_nice = {'(' : "C",
#              ')' : "C",
#              '[' : "L",
#              ']' : "L",
#              "{" : "S",
#              "}" : "S",
#              "<" : "A",
#              ">" : "A",
#              "+" : "+",
#              "-" : "+",
#              " " : "_"}

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print('Char indices:', char_indices)

N = 10
MAXLEN = 2*N+1
def pad(ws): return (ws + ' '*(MAXLEN - len(ws)))

encode = lambda ws: [char_indices[w] for w in ws]
decode = lambda ws: [indices_char[w] for w in ws]

def make_input(examples):
    return np.array([encode(pad("s"+seq)) for seq in examples])

def make_output(examples):
    return np.array([encode(pad(seq)) for seq in examples])

def make_weights(examples):
    return np.array([([float(1)] * len(seq) + [float(0)] * (MAXLEN - len(seq))) for seq in examples])

def make_examples(l):
    l = list(l)
    return {"x":make_input(l), "y":make_output(l), "weights":make_weights(l),
            "len":np.array([len(x) for x in l])}

def computeAccurate(x,preds):
    result = [None] * MAXLEN
    for j in range(len(x)): # len for this sentence
        c = indices_char[x[j]]
        if c in closeParen: # if we don't have a closing paren in the input, then we don't care
            # Among the closing parentheses, which one do we predict?
            predicted_close = max(closeParen, key=lambda c: preds[j][char_indices[c]])
            result[j] = int(c == predicted_close) # was that correct?
    return result

