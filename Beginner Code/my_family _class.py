# Creating a class and a subclass

class ochieze_family(object):
    def __init__(self,population):
        self.population = population
        self.names = None
    def get_name(self):
        return self.names
    def set_name(self,names=""):
        self.names = names

names = ['Basil','Cali','Felix','James']

ochieze = ochieze_family(10)
ochieze.set_name(names)  
print(ochieze.get_name())     

class Basil(Ochieze_Family):
    def children(self):
        sons = ['Ugochukwu','Emeka']
        daughters = ['Ifeoma','Amarachi']
        return(sons,daughters)
  
#     def cars(self):
#         sports = ['Toyota','Mercedes','Porsche']
#         sedan = ['Toyota','Nissan']
#         return(sports,sedan)

