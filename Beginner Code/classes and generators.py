#### enumerate the list of shopping items
####
####Foods = ['Onions','Maggi','Ginger','Orange','Pizza']
######print(Foods)
####
####Food = enumerate(Foods,1)
######print(Food)
####
####for position,food in Food:
####    print(f'{position:3d}) {food}')
####    
##
##
##
##
##
####  Class Creation
##
##class rect():
##
##    """ Python uses the __init__ method """
##    
##    def __init__(self,length,width):
##        self.length = length
##        self.width = width
##
##
##    """ Let's define our own method """
##    
##    def __repr__(self):
##        """ Tells python what to put out on the screen """
##        return "rect({self.length},{self.width})".format(self=self)
##    
##    def __str__(self):
##        """ Displays the context in a string format """
##        return (f'The rectangle has a {self.length} by {self.width} dimensions')
##    
##    def area(self):
##        """ This code prints the area of a rectangle """
##        print(f'The area of your rectangle is {self.length * self.width} meters')
##    
##    def perimeter(self):
##        """ This code prints the perimeter of a rectangle """
##        print(f'The perimeter of a rectangle is {2 * (self.length + self.width)} meters')
##
##
##length = 10
##width = 5
##dimensions = rect(length,width)
##
##print(dimensions)
##area_rect = dimensions.area()
##perimeter_rect = dimensions.perimeter()



##Generators: Has a set of instructions and sitting out there waiting to be used.

##my_name = 'George'
##name = iter(my_name)
####print(name)
##
##i = 0
##while i < len(my_name):
##    print(next(name))
##    i += 1
##
##

##def series(num):
##    n = 0
##    while n < num:
##        yield n
##        n += 1
##
##five = series(5)
##print(five)


##Rounding up values 

def fives(x):
    if x%5 == 4 or x%5 == 3:
        x += (5 - x%5)
##    if x % 5 == 4:
##        x += 1
##    elif x % 5 == 3:
##        x += 2
    print(x)


grades = [72,25,46,64,77,23,21,100,98,66]

for grade in grades:
    print(grade, end=" "), fives(grade)



    
