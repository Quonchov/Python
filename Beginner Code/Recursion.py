# Rescursion

# def factorial(x):
#     total = 1
#     while x > 0:
#         total *= x
#         x -= 1
#     return total

# print(factorial(6))


# for i in range(int(input())): s=input(); print(*["".join(s[::2]),"".join(s[1::2])])


# # Enter your code here. Read input from STDIN. Print output to STDOUT

# n = int(input())
# temp = []
# for i in range(0,n):
#     s = input()
#     temp.append(s)

# for i in range(0,n):
#     for j in range(0,len(temp[i]),2):
#         print(temp[i][j],end='')
#     print(end=' ')
#     for j in range(1,len(temp[i]),2):
#         print(temp[i][j],end='')
#     print()


#  Factorial
# def fact(n):
#     if n > -1:
#         if n == 0:
#             return 1
#         else:
#             return n * fact(n-1)
#     else:
#         return 'Negative'
    
    
# print(fact(5))


# Adding commas to the set of number
def expo(x):
    """ This gives a result of an exponential number"""
    x = x ** 2
    return x
# print("{:,}".format(expo(100)))
print(f"{expo(100):,}")