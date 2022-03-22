##Iteration tools

from itertools import combinations as comb

x = 'abcdefg'.upper()

y = comb(x,4.5)

b = [''.join(i) for i in y]
