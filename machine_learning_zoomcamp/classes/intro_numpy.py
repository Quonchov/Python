import numpy as np

a = np.array([1,2,3,4,5,10])
print(a)
d = np.array([9,7,6,5,42,2])
print(d)

# # change the values in the array
# a[2] = 12
# print(a)

# # creating a range of values 
# b = np.arange(start=2, stop=10) # the "start and stop" is optional
# b_1 = np.arange(5)
# print(b)

# # linspace creates an array filled with numbers between  the first parameter and the second parameter
# c = np.linspace(start=0, stop=10, num=50) # Again the "start, stop and num usually is the step" are optional arguements
# print(c)

# #  create multi-dimensional array
# d = np.zeros((5,2))
# print(d)

# e = np.array([[1,2,3],[13,4,5],[1,4,4]])
# print(e)

# # getting the indices
# print(e[0,1])

# # changing the indices above to 100
# e[0,1] = 100
# print(e)  # e is updated automatically

# #  getting the first row of the array
# print("This is the first row of the array: ",e[0])

# # replacing the row #2 with an array of numbers
# e[2] = [20,20,20]
# print(e)

# #  accessing the columns
# print("print the column here: ", e[:,1])

# # randomly generated array
# np.random.seed(2) # makes the randon seed fixed
# g = np.random.rand(5,2)
# print(g)

# np.random.seed(2) # makes the randon seed fixed
# h = np.random.randn(5,2)
# print(h)

# # two dimensional arrays of a random integer
# i = np.random.randint(0,100,(2,5))


# comparison gives a boolean result
a >= 2
a > d
print(a[a>d])  # returns the only the satisfied "True" conditions


# summarizing arrays
a.min()  # returns the minimum number in the array
a.sum()  # returns the sum of the array
a.mean()  # returns the average
a.std()  # computes the standard deviation


#  identity matrix
n = np.eye(3)
print(n)


# take the first two rows of matrix e
e2 = e[[0, 1]]
print(e2)

#  inverse of a matirx: must be a square matrix to achieve this
e_inverse = np.linalg.inv(e)
print(e_inverse)

e_identity = e_inverse.dot(e) # this gives back an identity matrix
print(e_identity)
