#  Chicken Nuggets

t,n,s = 20,9,6

def nuggets(y):
    global x
    x = {}
    for i in range(10):
        for j in range(10):
            for k in range(10):
                tot = i*t + j*n + k*s
                # print(tot)
                x[tot] = [i,j,k]
                # print(x)
                # print(y)
                if y == tot:
                    break
            if y == tot:
                break
        if y == tot:
            break
    print(f''' Total number of nuggets: {y}
          {x[y][2]}: 6 pieces
          {x[y][1]}: 9 pieces
          {x[y][0]}: 20 pieces''')
          
nuggets(60)