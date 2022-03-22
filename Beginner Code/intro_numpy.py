import numpy as np
# one_D_array = np.array([1,2,3,0,4,5])
# print(one_D_array)

# two_D_array = np.array([[1,2],[2,4],[4,56]])
# print (two_D_array)

# # Populate np with zeros and ones
# print(np.ones([3,3]))
# print(np.zeros([2,3]))


# # creates a range of values with a step
ran_ge = np.arange(2,10)
# print(ran_ge)

# creates a 6-element vector with low and high values random integer 
# six_matrix = np.random.randint(low=2,high=10,size=(6)) # creates a single vector
# six_matrix_1 = np.random.randint(low=2,high=10,size=([6,6])) # creates a matrix of 6-by-6
# print(six_matrix)
# print(six_matrix_1)

# # creates random floating points or numbers
# num_float = np.random.random(6)
# print(num_float)

# # broadcasting: enables you to carry out an mathematical operation on a matrix of unequal size
# num_float_add = num_float + 2.0
# print(num_float_add) 


# Adding noise to a number (this noise is a float num)
noise = (np.random.random([len(ran_ge)]) * 4) -2
ran_ge1 = ran_ge + noise
print(noise)