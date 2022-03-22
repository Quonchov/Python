# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 05:44:16 2022

@author: ochie
"""

# This code calculates the total cost of phones, laptops and portable games

# class item:
#     all = []
#     def __init__(self):
        
        
        
#      def cost_calc(self,x,y):
#          return x * y
     

# phones = item()
# phones.names = ["samsung","iphone","motorola"]
# phones.prices = 200
# phones.quantity = 1200
# phoneCost = phones.cost_calc(phones.prices,phones.quantity)
# print(phoneCost)




# Calculation tem costs
class Item:
    
    # Item properties
    def __init__(self,name: str, prices: float, quantity=0):
        
        # Run validation for price and quantity
        assert prices >= 0, f"{price} must be a positive value"
        assert quantity >= 0, f"{quantity} must be a positive value"
        
        # Assign to self objects
        self.name = name
        self.prices = prices
        self.quantity = quantity
        
    # Calculates the cost of items
    def cost_calc(self):
        return self.prices * self.quantity

item1 = Item("Phone",52.99,10)
item2 = Item("Laptop",600.99,3)

print(item1.cost_calc())
print(item2.cost_calc())

