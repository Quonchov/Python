""" For this assignment:
    Run gradient descent algorithm to find the values of m, b and apprpriate learning rate;
    For each iteration, compare previous cost with current cost, stop when costs are similar (use math.isclose function with 1e-20 threshold);
    Implement sklearn.linear_model to find coefficient m and intercept b;
    Compare the m and b generated by the linear regression using sklearn and gradient descent algorithm """

import numpy as np
import pandas as pd
from sklearn import linear_model
import math
import matplotlib.pyplot as plt


# consider math = x and cs = y
df = pd.read_csv("test_score.csv")
print(df)

""" First: Make plot to visualize the dataset """
# Plot to visulaize the dataset use y = mx + b
# plt.xlabel("math",  fontsize=20)
# plt.ylabel("cs", fontsize=20)
# plt.scatter(df.math, df.cs, color="red", marker="+")

""" Second: Apply Linear Regression """
reg = linear_model.LinearRegression()
reg.fit(df[["math"]], df.cs)

""" Get intercept and coefficient """
print(f"The coefficient of the Linear Regression is {reg.coef_}")
print(f"The intercept of the Linear Regression is {reg.intercept_}")

""" Third: Apply Gradient Descent """
x = np.array(df.math)
y = np.array(df.cs)

def grad_descent(x,y):

    m_curr = b_curr = 0     # current m and b
    iteration = 1000000        # number of iterations
    learning_rate = 0.0001   # Learning rate
    n = len(df.math)
    
    cost_prev = 0
    
    for i in range(iteration):
        y_pred = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y - y_pred)]) # formular can be found online for mse
        md = -(2/n) * sum(x * (y-y_pred))   # m derivative
        bd = -(2/n) * sum((y-y_pred))    # b derivative
        m_curr = m_curr - learning_rate * md         # updated m
        b_curr = b_curr - learning_rate * bd         # updated b
        
        if math.isclose(cost,cost_prev, rel_tol=1e-20, abs_tol=0.0):
            print(f"iteration number: {i}")
            break
        cost_prev = cost

        # Visualize the values
        print(f"m: {m_curr}, b: {b_curr}, cost: {cost}, iteration: {i}")
        
    return m_curr, b_curr



# if __name__ == "__main__":
#     df = pd.read_csv("test_score.csv")
#     x = np.array(df.math)
#     y = np.array(df.cs)
    
m, b = grad_descent(x,y)
print(f"coefficient and intecept using gradient descent is {m} and {b}")
print(f"coefficient and intecept using sklearn is {reg.coef_} and {reg.intercept_}")






