##tkinter calculator


import tkinter as tk  # tkinter is used to build GUI's in python
import tkinter.ttk as ttk  # Builds the buttons of the calculator a little smoother

# Creates the name of the operator
name = input("What's your name: ")

# Assigns the alias to win
win = tk.Tk()

# Creates the title of the calculator
win.title('Simple Calculator')


# Setting up the attributes of the calculator

win.configure(width=300,height=400,background='green')
# win.geometry('300x400+0+0')  # Takes in a string format for the geometry, addition sign controls where it appears on the screen

# Adding labels
label = tk.Label(win,text = f'Hi {name}, calculate anything you want with me 😜😁')
label.pack()


# Adding Buttons this uses button.pack()
button = ttk.Button(win,text = 0)  # sample of the buttons
button = tk.Button(win,text = '1')
button.pack()

