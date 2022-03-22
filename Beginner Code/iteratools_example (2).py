####Iteration tools
##
##from itertools import combinations as comb
##
##x = 'abcdefg'.upper()
##
##y = comb(x,4)
##
##b = [''.join(i) for i in y]



##from itertools import count
##
##x = count(1)
##
##print(next(x))

##Build the pyramid of numbers
'''
        1
    2   3   4
5   6   7   8   9

'''

from itertools import count

start = count(1,2) # the starting point of the count

num = count(1)

for i in range(1,4):
    print(('  ') * (3-i), end = '')
    for j in range(next(start)):
        print(next(num), end = ' ')
    print()
