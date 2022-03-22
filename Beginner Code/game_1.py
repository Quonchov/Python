from random import choice
# from string import ascii_letters as letters
from string import ascii_uppercase as letters


a_f = list(letters[:6])

num = iter(range(1,7))
hidden = [choice(a_f) + str(choice(list(range(1,7)))) for i in range(4)]
arr = [['O' for i in range(6)] for i in range(6)]

Play = True
while Play:
    num = iter(range(1,7))
    print(hidden)
    print("  "  + ' '.join(a_f).upper()) 
    for i in arr:
        print(next(num),end=" ")  # Used next() here because the iter() was created earlier
        print(' '.join(i))
    move = input("Enter Q to quit or select a location (e.g A5) for another move: ")
    if move.lower() == 'q':
        Play = False
        print("Thanks for playing")
        
        
numbers = list(range(1,23))
print(numbers)
# print(numbers.pop())
print(numbers.index(10,2,22))