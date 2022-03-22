""" This code shows how to write the class and subclass in python"""
class Animal(object):
    
    def __init__(self,age):
        self.age = age
        self.name = None
    
    def get_name(self):
        return self.name
    def get_age(self):
        return self.age
    
    def set_name(self,newname=""):
        self.name = newname
    def set_age(self,new_age):
        self.age = new_age
        
    def __str__(self):
        # pet_name = input('Which type of pet do you have: ').lower()
        
  class Cat(Animal):
                def speak(self):
                    return("Meow")
                def name(self,cat_name=""):
                    self.name = cat_name
            return "My cat's name is " + str(self.name) + " and will be " + str(self.age) + " years old today, whenever she cries you hear " + str(self.speak())
             
        if pet_name == "cat":
          
        elif pet_name == "dog":
            class Dog(Animal):
                def speak(self):
                    return("Bark")
                def set_name(self,dog_name=""):
                    self.name = dog_name
            return "My dog's name is " + str(self.name) + " and will be " + str(self.age) + " years old today, whenever she cries you hear " + str(self.speak())
        
        else:
            print(f"The {pet_name} you typed in cannot be found try a cat or dog")
               
cat = Cat(8)
cat.set_name("Kitten")
print(cat)


# dog = Dog(input('Type in how old your dog is: '))
dog = Dog(11)
dog.set_name("Bingo")
print(dog)