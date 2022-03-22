####hash problem
##
##n = int(input())
##
##for i in range(n,n+n):
####    t = tuple(i)
##    print(i)
##    
    

n = int(input())
integer_list = map(int, input().split())
tup = ()
for x in integer_list:
    tup+=(x,)
    print(tup)

print(hash(tup))



