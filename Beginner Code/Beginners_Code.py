#  Function definition 
# def hello(name):
#     print(f'Hello!!! {name}, it was great catching up with you')

# name = "George"
# hello(name)



# Trying to modify a bit
# def greetings():
#     """ This code greets the user after inputing their name"""
#     name = input("Enter Name: ")
#     print(f'Hello!!! {name}, it was great catching up with you')
#     # return name

# # print(greetings())
# greetings()

# # Condtionals and Loops
# numbers = list(range(20))
# # The code evaluates numbers that are odd and even
# for number in numbers:
#     if number % 2 == 1:
#         print(f'{number} is odd!!!')
#     else:
#         print(f'{number} is even!!!!!')

# Prints numbers arranged in rows and columns
for row in range(5):
    # print(row)
    for col in range(5):
        print(col,end="")
    print()
    