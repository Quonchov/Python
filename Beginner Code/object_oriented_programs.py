# class coordinate(object):
#     """ A coordinate made of x and y value """
#     def __init__(self,x,y):
#         self.x = x
#         self.y = y
#     def distance(self,offset):
#         x_diff_sq = (self.x - offset.x) ** 2
#         y_diff_sq = (self.y - offset.y) ** 2
#         return (x_diff_sq + y_diff_sq) ** 0.5

        
# c = coordinate(3,4)
# zero = coordinate(0,0)
# print(c.distance(zero))
# print(coordinate.distance(c,zero))


# class diff_two_squares(object):
    
#     """ Calculates the difference of two squares, takes two values a and b. """
#     def __init__(self,a,b):
#         self.a = a
#         self.b = b
#     def __str__(self):
#         return "The difference of the two squares is thus: " + str((self.a + self.b) * (self.a - self.b)) 
    
# c = diff_two_squares(10, 3)
# print(c)

class Animal(object):
    def __init__(self,age):
        self.age = age
        self.name = None
        self.color = None
        self.weight = None
        self.length = None
        self.type = None
    def get_name(self):
        return self.name
    def get_age(self):
        return self.age
    def get_color(self):
        return self.color
    def get_weight(self):
        return self.weight
    def get_length(self):
        return self.length
    def get_type(self):
        return self.type
    def set_age(self,newage):
        self.age = newage
    def set_name(self,newname = ""):
        self.name = newname
    def set_color(self,newcolor):
        self.color = newcolor
    def set_weight(self,newweight):
        self.weight = newweight
    def set_length(self,newlength):
        self.length = newlength
    def set_type(self,newtype):
        self.type = newtype
    def __str__(self):
        return "The name of my pet is " + str(self.name) + ", he is " + str(self.age) + " years old, " + str(self.color) + " in color and weighs " + str(self.weight) + ", with a total length of " + str(self.length) 
        


print("\n............Animal Discussion...........")
a = Animal(1)    # 1 is the initial age that can be changed anytime
a.set_age(200)   # Sets the age to 200 years
a.set_name('Quonchi')
a.set_color('Indigo')
a.set_weight('300 Pounds')
a.set_length('30 meters')
print(a)
# print(a.get_name())


# a.set_type = str(input("Guess the animal:")).lower()
# if a.set_type == 'snake':
#   print('Yep you got that right')
# else: 
#   print("Nope you EGG HEaD")
