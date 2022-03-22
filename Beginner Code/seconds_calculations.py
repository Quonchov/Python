##  Time calculations, seconds conversion


##def period(x):
##    """ x is the number of hours, this code returns the numbers of seconds
##        for the hours provided"""
##    y = 60*60*x
##    return (f'{y}seconds are in {x}hours')
##
##
##print(period(200))


def sec(x):
    d = x//86400  # 86400 is the number of seconds in a day
    h = (x%86400)//3600
    m = (x%3600)//60
    s = (x%60)
    print(f'{d:02}:{h:02}:{m:02}:{s:02}')  # :02 creates two digits for the value placer

    
print('d:h:m:s')
sec(86400)
##sec(9000)


walk = ["n","e","s","w"]

length = 4


def is_valid_walk(time,walk):
    # Determine if walk is valid your code goes here
    north = walk.count('n')
    south = walk.count("s")
    east = walk.count("e")
    west = walk.count("w")

    dis = len(walk)

    return (north == south) and (east == west) and (time == dis)

print(is_valid_walk(4,walk))

