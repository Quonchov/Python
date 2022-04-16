#### This lesson discusses how machine learning works using mean squared error or cost function
### mse = (1/n) * sumation from i = 1 to n * (yi - (mxi + b)) ^ 2, where y = mx+b looks familiar
""" Let's say you have an input of x and y arrays, and you want to predict a future of
	variable using this arrays, you can find a function that acts as a formulae"""

""" Gradient descent is an algorithm that finds the best fit line for any given training data set"""

# Determining the cost function and creating a learning rate
""" Use m = m - learning rate * del/del(m)"""
""" Use b = b - learning rate * del/del(b)"""


import numpy as np
import pandas as pd
from sklearn import linear_model


 # Implement gradeint descent using x and y values to find m and b, cost should be reduced at each time
 # cost function = cost = mse = (1/n) * sum(yi - y_pred)^2

# y = mx + b: we want to find m and b that give sthe values of x and y
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

def grad_descent(x,y):

	 # first start with m and b current
	 m_curr = b_curr = 0

	 # define number of iteration: how many steps you're going to take
	 iteration = 1000   # vary this value to get optimum cost

	 # length of your array
	 n = len(x)
	 learning_rate = 0.08 # start with a small value, then gradually twerk it to fit properly, get the sweet spot where the cost func starts reducing

	 for i in range(iteration):
		 y_pred = m_curr * x + b_curr
		 cost = (1/n) * sum([val**2 for val in (y - y_pred)]) # formular can be found online for mse
		 md = -(2/n) * sum(x * (y-y_pred))   # m derivative
		 bd = -(2/n) * sum((y-y_pred))    # b derivative
		 m_curr = m_curr - learning_rate * md         # updated m
		 b_curr = b_curr - learning_rate * bd         # updated b

		 # Visualize the values
		 print(f"m: {m_curr}, b: {b_curr}, cost: {cost}, iteration: {i}")


""" Threshold comparison use: math.isclose(a,b,*, rel_tol=1e-09, abs_tol=0.0)  ===> returns true if the values of a and b are close to each other and false otherwise """

# consider y = mx + b where m = 2 and b = 3 from the np.array provided
# Aim: make m ===> 2 and b ===> 3 : To achieve this twerk learning rate at a low iteration point, watch the cost to determine if its going up or down
grad_descent(x,y)