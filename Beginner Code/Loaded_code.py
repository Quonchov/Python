# Create a list to work on

class coordinate(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def distance(self,other):
        x_diff_sq = (self.x - other.x) ** 2
        y_diff_sq = (self.y - other.y) ** 2
        return(x_diff_sq + y_diff_sq) ** 0.5

c = coordinate(3,4)
zero = coordinate(0,0)
print(c.distance(zero))

import numpy as np
import matplotlib.pyplot as plot

# Get x values of the sine wave
time = np.arange(0, 10, 0.1);

# Amplitude of the sine wave is sine of a variable like time
amplitude = np.sin(time)

# Plot a sine wave using time and amplitude obtained for the sine wave
plot.plot(time, amplitude)

# Give a title for the sine wave plot
plot.title('Sine wave')

# Give x axis label for the sine wave plot
plot.xlabel('Time')

# Give y axis label for the sine wave plot
plot.ylabel('Amplitude = sin(time)')
plot.grid(True, which='both')
plot.axhline(y=0, color='k')
plot.show()

# Display the sine wave
plot.show()