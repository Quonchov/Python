##Scientific tkinter calculator


import tkinter as tk  # tkinter is used to build GUI's in python
import tkinter.ttk as ttk  # Builds the buttons of the calculator a little smoother

# Creates the name of the operator
# name = input("What's your name: ")

# Assigns the alias to win
win = tk.Tk()

# Creates the title of the calculator
win.title('Simple Calculator')


# Setting up the attributes of the calculator

# win.configure(width=300,height=400,background='green')
# win.geometry('300x400+0+0')  # Takes in a string format for the geometry, addition sign controls where it appears on the screen

# Adding labels
# label = tk.Label(win,text = f'Hi {name}, calculate anything you want with me 😜😁')
# label.pack()


# Adding Buttons this uses button.pack()
# button = ttk.Button(win,text = 0)  # sample of the buttons
# button = tk.Button(win,text = '1')
# button.pack()

expr = ''
text = tk.StringVar()

def press(num):
    global expr
    expr += str(num)
    text.set(expr)

def clc():
    global expr
    expr = ''
    text.set(expr)

def equal():
    global expr
    ttl = str(eval(expr))
    text.set(ttl)


def trig():
    global expr
    import math as m
    sin = str(m.sin(expr))
    text.set(sin)
    cos = str(m.cos(expr))
    text.set(cos)
    tan = str(m.tan(expr))
    text.set(tan)
    
display = ttk.Entry(win,justify='right',textvariable=text)  # Justify makes the typed display move from right to left
display.grid(row=0,columnspan=4,sticky='nwes')

button_7 = ttk.Button(win,text = '7', command=lambda:press(7))
button_7.grid(row=1,column=0)

button_8 = ttk.Button(win,text = '8', command=lambda:press(8))
button_8.grid(row=1,column=1)

button_9 = ttk.Button(win,text = '9', command=lambda:press(9))
button_9.grid(row=1,column=2)

button_d = ttk.Button(win,text = '/', command=lambda:press('/'))
button_d.grid(row=1,column=3)

button_6 = ttk.Button(win,text = '4', command=lambda:press(4))
button_6.grid(row=2,column=0)

button_5 = ttk.Button(win,text = '5', command=lambda:press(5))
button_5.grid(row=2,column=1)

button_4 = ttk.Button(win,text = '6', command=lambda:press(6))
button_4.grid(row=2,column=2)

button_m = ttk.Button(win,text = '*', command=lambda:press('*'))
button_m.grid(row=2,column=3)

button_3 = ttk.Button(win,text = '1', command=lambda:press(1))
button_3.grid(row=3,column=0)

button_2 = ttk.Button(win,text = '2', command=lambda:press(2))
button_2.grid(row=3,column=1)

button_1 = ttk.Button(win,text = '3', command=lambda:press(3))
button_1.grid(row=3,column=2)

button_s = ttk.Button(win,text = '-', command=lambda:press('-'))
button_s.grid(row=3,column=3)

button_0 = ttk.Button(win,text = '0', command=lambda:press(0))
button_0.grid(row=4,column=0)

button_dot = ttk.Button(win,text = '.', command=lambda:press('.'))
button_dot.grid(row=4,column=1)

button_c = ttk.Button(win,text = 'c', command = clc)
button_c.grid(row=4,column=2)

button_a = ttk.Button(win,text = '+', command=lambda:press('+'))
button_a.grid(row=4,column=3)

button_exp = ttk.Button(win,text = 'exp', command = lambda:press('**'))
button_exp.grid(row=5,column=0)

# button_sin = ttk.Button(win,text = 'sin', command = lambda:press('sin('))
# button_sin.grid(row=5,column=1)

# button_cos = ttk.Button(win,text = 'cos', command = lambda:press('cos('))
# button_cos.grid(row=5,column=2)

button_tan = ttk.Button(win,text = 'tan', command = trig)
button_tan.grid(row=5,column=3)

button_e = ttk.Button(win,text = '=', command = equal)
button_e.grid(row=6,columnspan=4,sticky='nsew')


win.mainloop()
