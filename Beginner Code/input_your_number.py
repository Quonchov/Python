# Phone Contact

def phone_number(x):
  
    while len(x) <= 9:
        print(f'{x} is not complete')
        x = input('please check the number and try again: ')
        
    if len(x) == 10:
        print(f'({x[:3]}) {x[3:6]}-{x[6:]}')         
    
            
num = input('Put in your phone number: ')
phone_number(num)


