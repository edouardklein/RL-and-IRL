# Tries to run testsuit with python version
# python<major>.<minor>
# where major and minor varies between 2-3 and 0-10.
# (expand this if you have say python1.11 or python4.0)
# Now also looks for pypy implementations.
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

import os
from subprocess import call
from matrix2latex import matrix2latex

table = list()
cl = list()

def test(python, table, cl):
    ret = call(python + " test.py", shell=True, stdout=file(os.devnull, "w"))

    print(python)            
    print(ret)
    if ret == 0: table.append(["True"])
    else: table.append(["False"])
    cl.append(python)


for major in range(2, 3+1):
    for minor in range(0, 10):
        python = 'python%d.%d' % (major, minor)
        
        if call(python + " -c ''", shell=True, stderr=file(os.devnull, "w")) == 0:
            test(python, table, cl)

if call("pypy-c" + " -c 'pass'", shell=True, stderr=file(os.devnull, "w")) == 0:
    test("pypy-c", table, cl)

c = "Does 'python test.py' return 0?"
compatibleTable = matrix2latex(table, 'doc/compatibleTable',
                               headerColumn=cl, headerRow=['Compatible'],
                               caption=c)
print compatibleTable
