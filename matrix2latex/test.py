"""This file is part of matrix2latex.

matrix2latex is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

matrix2latex is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with matrix2latex. If not, see <http://www.gnu.org/licenses/>.
"""

# tests for matrix2latex.py
from matrix2latex import matrix2latex
m = [[1, 2, 3], [4, 5, 6]]

f = open('test.tex')
answers = dict()
for line in f:
    if line.startswith('%%%'):
        name = line[3:-1]               # ignore %%% and \n
        answers[name] = ''
    else:
        answers[name] += line

def loopTwoLists(x, y):
    for ix in range(max([len(x), len(y)])):
        try: a = x[ix].strip()
        except: a = ''
        try: b = y[ix].strip()
        except: b = ''
        yield a, b

def assertEqual(x, name):
    # assert each line is equal, ignoring leading and trailing spaces
    print(x)
    y = answers[name]
    x = x.split('\n')
    y = y.split('\n')
    correct = True
    for a, b in loopTwoLists(x, y):
        if a != b:
            correct = False # found 1 or more error
            
    if not(correct):
        for a, b in loopTwoLists(x, y):
            print(a,b)
        raise AssertionError

def test_simple():
    t = matrix2latex(m)
    assertEqual(t, "simple")

def test_transpose1():
    t = matrix2latex(m, transpose=True)
    assertEqual(t, "transpose1")

def test_transpose2():
    cl = ["a", "b"]
    t = matrix2latex(m, transpose=True, headerRow=cl)
    assertEqual(t, "transpose2")

def test_file():
    matrix2latex(m, 'tmp.tex')
    f = open('tmp.tex')
    content = f.read()
    f.close()
    assertEqual(content, "file")

def test_environment1():
    t = matrix2latex(m, None, "table", "center", "tabular")
    assertEqual(t, "environment1")
 
def test_environment2():
    t = matrix2latex(m, None, "foo", "bar")
    assertEqual(t, "environment2")
   
def test_labels1():
    cl = ["a", "b"]
    rl = ["c", "d", "e"]
    t = matrix2latex(m, None, headerColumn=cl, headerRow=rl)
    assertEqual(t, "labels1")

def test_labels2():
    # only difference from above test is names, note how above function
    # handles having too few headerRow
    cl = ["a", "b"]
    rl = ["names", "c", "d", "e"]
    t = matrix2latex(m, None, headerColumn=cl, headerRow=rl)
    assertEqual(t, "labels2")

def test_labels3():
    # pass in environment as dictionary
    e = dict()
    e['headerColumn'] = ["a", "b"]
    e['headerRow'] = ["names", "c", "d", "e"]
    t = matrix2latex(m, None, **e)
    assertEqual(t, "labels3")

def test_labels4():
    t = matrix2latex(m, None, caption="Hello", label="la")
    assertEqual(t, "labels4")
    
def test_alignment1():
    t = matrix2latex(m, alignment='r')
    t = t.split('\n')[2].strip()
    assert t == r"\begin{tabular}{rrr}", t

def test_alignment2():
    cl = ["a", "b"]
    rl = ["names", "c", "d", "e"]
    t = matrix2latex(m, alignment='r', headerColumn=cl, headerRow = rl)
    t = t.split('\n')[2].strip()
    assert t == r"\begin{tabular}{rrrr}", t

def test_alignment2b():
    rl = ["a", "b"]
    cl = ["names", "c", "d", "e"]
    t = matrix2latex(m, alignment='r', headerColumn=cl, headerRow = rl, transpose=True)
    t = t.split('\n')[2].strip()
    assert t == r"\begin{tabular}{rrr}", t

def test_alignment3():
    t = matrix2latex(m, alignment='rcl')
    t = t.split('\n')[2].strip()
    assert t == r"\begin{tabular}{rcl}", t

def test_alignment4():
    t = matrix2latex(m, alignment='rcl', headerColumn=["a", "b"])
    t = t.split('\n')[2].strip()        # pick out only third line
    assert t == r"\begin{tabular}{rrcl}", t

def test_alignment5():
    t = matrix2latex(m, alignment='r|c|l', headerColumn=["a", "b"])
    t = t.split('\n')[2].strip()        # pick out only third line
    assert t == r"\begin{tabular}{rr|c|l}", t

def test_alignment_withoutTable():
    t = matrix2latex(m, None, "align*", "pmatrix", format="$%.2f$", alignment='c')
    assertEqual(t, "alignment_withoutTable")

def test_numpy():
    try:
        import numpy as np
        for a in (np.matrix, np.array):
            t = matrix2latex(a(m), None, "align*", "pmatrix")
            assertEqual(t, "numpy")
    # Systems without numpy raises import error,
    # pypy raises attribute since matrix is not implemented, this is ok.
    except (ImportError, AttributeError):
        pass

def test_string():
    t = matrix2latex([['a', 'b', '1'], ['1', '2', '3']], format='%s')
    assertEqual(t, "string")

def test_none():
    m = [[1,None,None], [2,2,1], [2,1,2]]
    t = matrix2latex(m)
    assertEqual(t, "none")
    
    m2 = [[1,float('NaN'),float('NaN')], [2,2,1], [2,1,2]]
    t2 = matrix2latex(m)    
    assertEqual(t2, "none")

    t3 = matrix2latex(m, format='$%d$')
    assertEqual(t3, "none")

def test_infty1():
    try:
        import numpy as np
        m = [[1,np.inf,float('inf')], [2,2,float('-inf')], [-np.inf,1,2]]
        t = matrix2latex(m)
        assertEqual(t, "infty1")
    except (ImportError, AttributeError):
        pass

def test_infty2():
    # same as above but without numpy
    inf = float('inf')
    m = [[1,inf,float('inf')], [2,2,float('-inf')], [-inf,1,2]]
    t = matrix2latex(m)
    assertEqual(t, "infty2")
    
if __name__ == '__main__':
    import test
    for d in test.__dict__:
        if 'test_' in d:
            eval(d+'()')
