####Create a class for calculating the area and perimeter of a rectangle
##
##class rect():
##
##    """ This is what python uses """
##    def __init__(self,length,width):
##        self.length = length
##        self.width = width
##
##
##    """" This is waht is displayed on the screen before the code starts """
##    def __str__(self):
##        return (f'Dimension of the rectangle is {self.length} by {self.width}')
##
##    """ Calculates the area of the rectangle in meters squared"""
##    def area(self):
##        print(f'Area of the rectangle is {self.length * self.width} meteres sqaured')
##
##    """ Calculates the perimenter of the rectangle in meters """
##    def perimeter(self):
##        print(f'Perimeter of the rectangle is {(self.length + self.width) * 2} meters')
##
##
##
##length_A = 100
##width_A = 80
##length_B = 120
##width_B = 100
##
##dimensions_A = rect(length_A,width_A)
##print(dimensions_A)
##dimensions_B = rect(length_B,width_B)
##print(dimensions_B)
##
##area_rect_A = dimensions_A.area()
##perimeter_rect_A = dimensions_A.perimeter()
##
##area_rect_B = dimensions_B.area()
##perimeter_rect_B = dimensions_B.perimeter()
##
#### Create the class of different races of humans with same attributes
##
##class humans():
##    
##    posture = "stands Erect".split()
##    kingdom = "Mammals"
##    Attributes = ["Walk","Run","Eat","Swim"]
##
##class Africans(humans):
##    pass
##
##class Americans(humans):
##    pass
##
##class Asians(humans):
##    pass
##
##class Europeans(humans):
##    pass
##
##



##   Another piece of example

name = 'Sam'
name = iter(name)
##print(name)


for i in name:
    print(i)
