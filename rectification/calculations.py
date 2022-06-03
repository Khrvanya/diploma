import sympy as sp
from sympy import Matrix, symbols, init_printing, latex


# 1
ex, ey = symbols('ex ey')
k = Matrix([[0, 0, 1]])
e = Matrix([[ex, ey, 1]])
d_origin, d = sp.sqrt(ex ** 2 + ey ** 2), symbols('d')
Ex = Matrix([[0, -1, ey],
             [1, 0, -ex],
             [-ey, ex, 0]])

# 2
Rx = k.cross(e.cross(k))
assert sp.sqrt(Rx.dot(Rx)) == d_origin
Ry = k.cross(e)
assert sp.sqrt(Ry.dot(Ry)) == d_origin

Rx_norm = Rx / d
Ry_norm = Ry / d
Rz_norm = k
R = Matrix([Rx_norm, Ry_norm, Rz_norm])

# 3
Rx_norm = Rx / d_origin
Rxe = Rx_norm.dot(e).simplify()
assert Rxe == d_origin
Rxe = d
G = Matrix([[1, 0, 0],
            [0, 1, 0],
            [-1/Rxe, 0, 1]])

# 4
Rr = G * R

# 5
# F = symbols('F')
M = Rr * Ex  #  * F

# 6
# Mx, My, Mz = symbols('Mx My Mz')
# M_x = My.cross(Mz)
# M_ = Matrix([M_x, My, Mz])

# 7
a, b, c = symbols('a b c')
A = Matrix([[a, b, c],
            [0, 1, 0],
            [0, 0, 1]])

# 8
# Rl = A * M_

# 9
print(latex(R))
