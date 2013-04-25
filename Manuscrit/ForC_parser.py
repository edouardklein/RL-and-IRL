# Yacc example

import ply.yacc as yacc
import sys
import os

# Get the token map from the lexer.  This is required.
from ForC_lex import tokens,lexer

#Org-derived grammar rules
def p_expression_org_text(p):
    'expression : TEXT'

def p_expression_text_composition(p):
    'expression : expression TEXT'
    
#ForC grammar rules
def p_expression_definition_symbol(p):
    'expression : expression SHEBANG_s NAME ELEMENT ELEMENT'
    parser.dic_s[p[3]] = [p[4],p[5]]
def p_expression_definition_composite_symbol(p):
    'expression : expression SHEBANG_cs NUMBER NAME ELEMENT ELEMENT ELEMENT'
    parser.dic_cs[p[4]] = [p[3],p[5],p[6],p[7]]

# Build the parser
parser = yacc.yacc(debug=True)
parser.dic_s = {}
parser.dic_cs = {}
parser.org_text=""
#print()
parser.parse(open(sys.argv[1]).read(),debug=0)
#for i in [parser.dic_s,parser.dic_cs]:
#    print(i)
headers=""


def stripelem(s):
    return s.strip()[1:-1]

SYMBOL_TEMPLATE='''
\\newglossaryentry{{{name}}}{{type=notation,
name={{\\ensuremath{{{notation}}}}},
description={{{description}}},
}}
\\newcommand{{\\{name}}}{{\gls{{{name}}}}}
'''
for k in parser.dic_s.keys():
    headers+=SYMBOL_TEMPLATE.format(name=k.strip(),
                                 notation=stripelem(parser.dic_s[k][0]),
                                 description=stripelem(parser.dic_s[k][1]))


CS_TEMPLATE='''
\\newcommand{{\\{name}}}[{number}]{{\ensuremath{{\glsadd{{FORCDONOTUSE{name}}}{definition}}}}}
\\newglossaryentry{{FORCDONOTUSE{name}}}{{type=notation,
name={{\\ensuremath{{{notation}}}}},
description={{{description}}},
}}
'''

for k in parser.dic_cs.keys():
    headers+=CS_TEMPLATE.format(name=k.strip(),
                                number=parser.dic_cs[k][0].strip(),
                                definition=stripelem(parser.dic_cs[k][1]),
                                notation=stripelem(parser.dic_cs[k][2]),
                                description=stripelem(parser.dic_cs[k][3]))

if os.path.exists("glossaries_header.org"):
    print(open("glossaries_header.org").read())
else:
    print("#+LaTeX_header:"+r"\usepackage[toc,savewrites,xindy,sort=def]{glossaries}")
print("#+LaTeX_header:"+r"\newcommand{\dash}{\ensuremath{'}}")
print("#+LaTeX_header:"+r"\newglossary[nlg]{notation}{not}{ntn}{Notation}")
for line in headers.split('\n'):
    print("#+LaTeX_header:"+line)

print(open(sys.argv[1]).read())

    
