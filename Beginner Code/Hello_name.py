# Calculates the square root of numbers

##from math import sqrt
##
##def square_root(x):
##    x = sqrt(x)
##    return x
##
##print(square_root(100))

##
##my_shopping_list = open('shopping_list','w')
##my_shopping_list.close()



def my_list():
    item = []
    while True:
        with open('shopping_cart.txt','a+') as file:
            item = input('Enter item: ')
            if item == "Exit":
                break
            elif item == 'Finish':
                    print(file.read())
            else:
                file.write(item + '\n')
            file.close()

my_list()
