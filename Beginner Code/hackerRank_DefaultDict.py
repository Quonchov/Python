##Hacker Rank

from collections import defaultdict

d = defaultdict(list)
list1=[]

n, m = map(int,input('Enter the values of n and m: ').split())

for i in range(1, n+1):
    d[input('Enter five alphabets: ')].append(str(i))
    print(d)


for i in range(m):
    b = input('Enter two alphabets: ')
    if b in d: print(' '.join(d[b]))
    else: print(-1)
