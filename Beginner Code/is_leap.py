def is_leap(year):
    """ This code accepts an integer calender year to confirm if it is a leap year
    or not """
    leap = False
    
    # Write your logic here
    return year % 4 == 0 and (year % 100 == 1 or year % 400 == 0)

year = int(input())
print(is_leap(year))
