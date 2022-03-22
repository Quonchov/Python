# enclosing, nonlocal

# def exp(n):
#     # n = 1
#     def num(x):
#         # nonlocal = n
#         return x ** n
#     return num

# square = exp(2)
# cube = exp(3)

# print(square(2))
# print(cube(3))


#  Decorator

# def cube(func):
#     def wrapper(value):
#         return func(value) ** 3
#     return wrapper

# @cube
# def num(x):
#     return(x)

#  Shuffle and Choices

from random import shuffle, choice

doors = ['goat','goat','car']
shuffle(doors)
print(doors)
