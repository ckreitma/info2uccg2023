# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt

cantidad_puntos = 400
iteraciones = 8000

# Create random input and output data
x = np.linspace(-math.pi, math.pi, cantidad_puntos)
y = np.sin(x)
#y = 0 + 0.1*x + 0*x*x + 0*x*x*x

# Temporal. Solo para mostrar cómo se generan los puntos "correctos"
# Ahora vamos a la prediccion
#plt.figure(figsize=(60, 20))
#plt.plot(x, y)
#plt.xlabel('X')
#plt.ylabel('Y')

# Fijamos el límite y.
#plt.ylim(-2,2)
#plt.show()


#############################
### y = a + bx + cx^2 + dx^3
#############################
# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

print(f'Inicial: y = {a} + {b} x + {c} x^2 + {d} x^3')

# UCCG. Agregamos la lista de errores para ir visualizando.
error_history = []
epoch_list = []

# La predicción no se hace con capas de una red neuronal sino con un polinomio.
# Esta técnica se denomina regresión logística y se utiliza en cálculo numérico.
# prediction (1,2.3,-3.1,4.5,[0,1,2,3,4,5,6])
def prediction(a,b,c,d,x):
    # y = a + b x + c x^2 + d x^3
    return a + b * x + c * x ** 2 + d * x ** 3

learning_rate = 2e-6
for t in range(iteraciones):
    # Forward pass: compute predicted y
    y_pred = prediction(a,b,c,d,x)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()

    #Agregamos el control del historial de error.
    error_history.append(loss)
    epoch_list.append(t)

    #Cada veinte iteraciones vamos mostrando los avances.
    if t % 1000 == 0:
        print(t, loss)

        # Imprimimos el seno y la predicción
        plt.figure(figsize=(60, 20))
        plt.plot(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')

        seno_aproximado = []
        for p in x:
            seno_aproximado.append(prediction(a,b,c,d,p))
        plt.plot(x,seno_aproximado)

        # Fijamos el límite y.
        plt.ylim(-2,2)
        plt.show()

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

# Imprimimos el avance.
plt.figure(figsize=(30,5))
plt.plot(epoch_list,error_history)
plt.xlabel('Epoch')
plt.ylabel('Erro')
plt.show()
quit()