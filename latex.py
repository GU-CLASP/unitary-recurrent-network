def rows(rows):
    return "\n".join([" & ".join(row) + "\\\\" for row in rows])

def fmt(x):
    return "{:.2f}".format(x)

def fmt_list(l):
    return list("{:.2f}".format(x) for x in l)


def cmd(c,body):
    return "\\"+c+"{" + body + "}"

def math(x):
    return cmd("ensuremath",x)

def env(x,body):
    return "\n".join([cmd("begin",x),body,cmd("end",x)])

def small_matrix(x):
    return math(cmd("parens",env("smallmatrix", rows(fmt_list(row) for row in x.tolist()))))

def angles(xs):
    xs = xs.tolist()
    xs.sort(reverse=True)
    return " ".join([cmd("showangle",fmt(x)) for x in xs])

def verb(s):
    return "\\verb:" + s + ":"
