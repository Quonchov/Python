##list Challenge

####process = input("The required number of process involved: ")
##list_num = []
##
##list_num.insert(0,5)
##list_num.insert(1,10)
##list_num.insert(0,6)
##print(list_num)
##
##
####Remove and append
##list_num.remove(6)
##list_num.append(9)
##list_num.append(1)
##
##list_num = sorted(list_num)
##print(list_num)
##
##
##list_num.pop()
##list_num = sorted(list_num, reverse=True)
##print(list_num)
##
##    

N = int(input())
empty = []
conv = []

for i in range(N):
    x = input().split()
    empty.append(x)

for i in range(len(empty)):
    if empty[i][0] == 'insert':
        x = int(empty[i][1])
        y = int(empty[i][2])
        conv.insert(x,y)
    elif empty[i][0] == 'print':
        print(conv)
    elif empty[i][0] == 'remove':
        conv.remove(int(empty[i][1]))
    elif empty[i][0] == 'append':
        conv.append(int(empty[i][1]))
    elif empty[i][0] == 'sort':
        conv.sort()
    elif empty[i][0] == 'pop':
        conv.pop()
    elif empty[i][0] == 'reverse':
        conv.reverse()
    
