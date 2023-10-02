import math
from scipy.optimize import fsolve

def f(x):
    return x ** 3 + x + 1

def d1f(x):
    return 3 * x ** 2 + 1

def d2f(x):
    return 6*x

def second_order_get_next_xi(x):
     d0 = f(x)
     d1 = d1f(x)
     d2 = d2f(x)

     a = d2 / 2
     b = d1 - x * d2
     c = d0 - x * d1 + (d2 * x**2)/2

     discr = b**2 - 4*a*c
     #print(discr)

     sqt = math.sqrt(discr)
     neg = -(b + sqt)/(2*a)
     pos = -(b - sqt)/(2*a)

     return neg if abs(x-neg)<abs(x-pos) else pos


def first_order_get_next_xi(x):
    return x - f(x) / d1f(x)

def newton_steps(initialguess, numsteps):
    firstorderguess = [initialguess]
    secondorderguess = [initialguess]
    x,y = initialguess, initialguess
    for i in range(numsteps):
        x = second_order_get_next_xi(x)
        y = first_order_get_next_xi(y)
        secondorderguess.append(x)
        firstorderguess.append(y)
    return firstorderguess,secondorderguess


root = fsolve(f, 0)[0]
print('Approx solution: '+str(root))

firststeps, secondsteps = newton_steps(-0.7,10)
firsterror,seconderror = [],[]

for s in range(len(firststeps)):
    firsterror.append(abs(firststeps[s]-root))
    seconderror.append(abs(secondsteps[s] - root))
print('First order steps: ' + str(firststeps))
print('Second order steps: ' + str(secondsteps))
print('First order errors: ' + str(firsterror))
print('Second order errors: ' + str(seconderror))