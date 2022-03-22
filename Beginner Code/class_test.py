class Jedi:
    def __init__(self,name,age):
        self.jedi_name = name
        self.jedi_age = age

    

    def say_hi(self):
        print(f"Hello, my name is {self.jedi_name} and I am {self.jedi_age}")

j1 = Jedi("Obiwan",12)
j1.say_hi()

