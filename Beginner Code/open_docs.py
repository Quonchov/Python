# Creating New Files

new_file = open("test_1.txt",'w')
new_file.write('Hi George!!!' + '\n')
new_file.close()

new_file = open('test_1.txt','a')
new_file.write('How are you doing today? ')
new_file.close()

new_file = open('test_1.txt')
print(new_file.read())
new_file.close()

# new_file= open('test_1.txt','a')
# new_file.write('Am doing well')
# print(new_file.read())
# new_file.close()






# #  Using context manager

# with open('test_1.txt') as new_file:
#     print(new_file.read())
    


#  Create a shopping list 
# def my_list():
#     with open('shopping_list', 'a') as shop_list:
#         item = input('Enter an item needed: ')
#         shop_list.write(item + '\n')
#         print(shop_list)
    
# my_list()