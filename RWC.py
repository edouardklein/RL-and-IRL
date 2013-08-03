
#code from http://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
import random

def weighted_choice(choices):
   total = sum(w for c,w in choices)
   r = random.uniform(0, total)
   upto = 0
   for c, w in choices:
      if upto+w > r:
         return c
      upto += w
   assert False, "Shouldn't get here"
