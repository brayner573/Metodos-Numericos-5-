import numpy as np
import matplotlib.pyplot as plt

# Definir la función y sus derivadas según sea necesario
f = lambda x: 5 * np.log(x) - 2 + 0.4 * x
df = lambda x: 5 / x + 0.4
g = lambda x: np.exp(0.4*x - 2)  # Suponiendo g(x) para el método de punto fijo

# Método de Bisección
def bisection_method(f, xl, xu, es=1e-6, max_iter=100):
    if f(xl) * f(xu) > 0:
        return None, []
    iter_count = 0
    xr = xl
    approximations = []
    while iter_count < max_iter:
        xr_old = xr
        xr = (xl + xu) / 2
        approximations.append(xr)
        if f(xr) == 0 or abs((xr - xr_old) / xr) < es:
            break
        if f(xl) * f(xr) < 0:
            xu = xr
        else:
            xl = xr
        iter_count += 1
    return xr, approximations

# Método de Falsa Posición
def false_position_method(f, xl, xu, es=1e-6, max_iter=100):
    if f(xl) * f(xu) > 0:
        return None, []
    iter_count = 0
    xr = xl
    approximations = []
    while iter_count < max_iter:
        xr_old = xr
        xr = xu - (f(xu) * (xl - xu)) / (f(xl) - f(xu))
        approximations.append(xr)
        if f(xr) == 0 or abs((xr - xr_old) / xr) < es:
            break
        if f(xl) * f(xr) < 0:
            xu = xr
        else:
            xl = xr
        iter_count += 1
    return xr, approximations

# Método de Newton-Raphson
def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    xn = x0
    approximations = [xn]
    for n in range(max_iter):
        fxn = f(xn)
        if abs(fxn) < tol:
            return xn, approximations
        dfxn = df(xn)
        if dfxn == 0:
            return None, approximations
        xn = xn - fxn / dfxn
        approximations.append(xn)
    return None, approximations

# Método de la Secante
def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    approximations = [x0, x1]
    for n in range(max_iter):
        if f(x0) == f(x1):
            return None, approximations
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        approximations.append(x2)
        if abs(x2 - x1) < tol:
            return x2, approximations
        x0, x1 = x1, x2
    return None, approximations

# Método de Punto Fijo
def fixed_point_method(g, x0, tol=1e-6, max_iter=100):
    approximations = [x0]
    for n in range(max_iter):
        x1 = g(x0)
        approximations.append(x1)
        if abs(x1 - x0) < tol:
            return x1, approximations
        x0 = x1
    return None, approximations

# Valores iniciales
xl, xu = 0.1, 10  # Para Bisección y Falsa Posición
x0 = 1  # Para Newton-Raphson y Punto Fijo
x0_secant, x1_secant = 0.1, 10  # Para Secante

# Ejecutar métodos
root_bisection, approximations_bisection = bisection_method(f, xl, xu)
root_false_position, approximations_false_position = false_position_method(f, xl, xu)
root_newton, approximations_newton = newton_raphson(f, df, x0)
root_secant, approximations_secant = secant_method(f, x0_secant, x1_secant)
root_fixed_point, approximations_fixed_point = fixed_point_method(g, x0)

# Graficar las aproximaciones
x = np.linspace(0.1, 10, 400)
y = f(x)

plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.plot(x, y, label='Función $f(x) = 5 \\log(x) - 2 + 0.4x$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.scatter(approximations_bisection, [f(x) for x in approximations_bisection], color='red')
for i, (xi, yi) in enumerate(zip(approximations_bisection, [f(x) for x in approximations_bisection])):
    plt.text(xi, yi, f'${i}$', fontsize=12)
plt.title('Aproximaciones del Método de Bisección')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(x, y, label='Función $f(x) = 5 \\log(x) - 2 + 0.4x$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.scatter(approximations_false_position, [f(x) for x in approximations_false_position], color='red')
for i, (xi, yi) in enumerate(zip(approximations_false_position, [f(x) for x in approximations_false_position])):
    plt.text(xi, yi, f'${i}$', fontsize=12)
plt.title('Aproximaciones del Método de Falsa Posición')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(x, y, label='Función $f(x) = 5 \\log(x) - 2 + 0.4x$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.scatter(approximations_newton, [f(x) for x in approximations_newton], color='red')
for i, (xi, yi) in enumerate(zip(approximations_newton, [f(x) for x in approximations_newton])):
    plt.text(xi, yi, f'${i}$', fontsize=12)
plt.title('Aproximaciones del Método de Newton-Raphson')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(x, y, label='Función $f(x) = 5 \\log(x) - 2 + 0.4x$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.scatter(approximations_secant, [f(x) for x in approximations_secant], color='red')
for i, (xi, yi) in enumerate(zip(approximations_secant, [f(x) for x in approximations_secant])):
    plt.text(xi, yi, f'${i}$', fontsize=12)
plt.title('Aproximaciones del Método de la Secante')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(x, y, label='Función $f(x) = 5 \\log(x) - 2 + 0.4x$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.scatter(approximations_fixed_point, [f(x) for x in approximations_fixed_point], color='red')
for i, (xi, yi) in enumerate(zip(approximations_fixed_point, [f(x) for x in approximations_fixed_point])):
    plt.text(xi, yi, f'${i}$', fontsize=12)
plt.title('Aproximaciones del Método de Punto Fijo')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
