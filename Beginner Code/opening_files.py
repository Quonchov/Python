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
##    item = []
    while True:
        with open('shopping_stuffs.txt','a+') as file:
##            print(file.tell())            
            item = input('Enter item: ')
            if item == 'EXIT':
                break
            elif item == 'LIST':
##                    print(file.tell())
##                    file.seek(0)
##                    print(file.read())
                items = list(enumerate(file.read().split(),1))
                for count, item in items:
                    print(f'{count:3d}) {item}')
            else:
                file.write(item + '\n')
##            file.close()

my_list()

