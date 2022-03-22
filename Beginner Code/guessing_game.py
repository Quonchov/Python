# Bisectional Search

import random as r

guess = r.randint(1,100)
print(guess)

# Start Guess Game

value = int(input('Put in a number: '))
''' Put in an integer value for the guessing game to work properly)'''
if value < guess:
    print('The value is less than the required value.')
    guess = guess / 2
    input(f'Try another value between {value} and {guess}: ')
elif value > guess:
    print('The value is greater than the required.')
    input(f'Try another value between 0 and {value}')