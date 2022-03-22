##Random Password Generator

import random, string


def password(length,num=False,strength='weak'):
    """ Strength of password :- weak, strong,very strong
        Length of password
        Number included to make it a bot complex """
    lower = string.ascii_lowercase
    upper = string.ascii_uppercase
    letters = lower + upper
    dig = string.digits
    punct = string.punctuation
    passwrd = ""
    
    if strength == 'weak':
        if num:
            length -= 2
            for i in range(2):
                passwrd += random.choice(dig)
        for i in range(length):
            passwrd += random.choice(lower)

    elif strength == 'strong':
        if num:
            length -= 2
            for i in range(2):
                passwrd += random.choice(dig)
        for i in range(length):
            passwrd += random.choice(letters)
            
    elif strength == 'very strong':
        ran = random.randint(2,4)
        if num:
            length -= ran
            for i in range(ran):
                passwrd += random.choice(dig)
        length -= ran
        for i in range(ran):
            passwrd += random.choice(punct)
        for i in range(length):
            passwrd += random.choice(letters)
        
            
    passwrd = list(passwrd)
    random.shuffle(passwrd)
    return ''.join(passwrd)

print(password(20,num=True,strength='very strong'))

      
