import random

a = 0.3
b = 2.3
c = 0.12

inferior = -10
superior = 10

def f1(x1,x2,x3):
    r = (a**x1) + (b*(x2**2)) - (x3**c)
    return r

random.seed()
for i in range(5):
    x1 = random.uniform(inferior,superior)
    x2 = random.uniform(inferior,superior)
    x3 = random.uniform(inferior,superior)

    # Mostramos el resultado:
    y = f1(x1,x2,x3)
    print(f'<{x1:.2f},{x2:.2f},{x3:.2f} --> {y}')