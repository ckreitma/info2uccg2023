""" Cacula el factor de ramificación para un árbol de 
12 nodos y la solución en una profundidad 4"
"""

# Encontrar un número b tal que b + b*b + b*b*b + b*b*b*b = 12

# Lo más bajo que puede ser es 1
delta = 0.001
b = 1.0
suma = b + b*b + b*b*b + b*b*b*b
while suma < 50:
    b += delta
    suma = b + b*b + b*b*b + b*b*b*b
print(f'b={b} suma={suma}')
