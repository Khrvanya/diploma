import sympy as sp
from sympy import Matrix, symbols


ex, ey, ez = symbols('ex ey ez')
e = Matrix([[ex, ey, ez]])
k = Matrix([[0, 0, 1]])

r1 = k.cross(k.cross(e))
r1_norm = r1 / sp.sqrt(r1.dot(r1))
r2 = k.cross(e)
r2_norm = r2 / sp.sqrt(r2.dot(r2))
R = Matrix([r1_norm, r2_norm, k])

x = r1.dot(e)
x_norm = r1_norm.dot(e)
G = Matrix([[1, 0, 0], [0, 1, 0], [-ez/x_norm, 0, 1]])

Rr = G * R

Ex = Matrix([[0, -ez, ey], [ez, 0, -ex], [-ey, ex, 0]])
print()
