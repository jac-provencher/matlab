import math
import matplotlib as mpl
"""
Méthode de la bissection
(1) Choisir un intervalle de départ [x1, x2] où la fonction f possède un changement de signe,c’est-à-dire où f(x1) × f(x2) < 0
(2) Approximer la racine par le point milieu xm = (x1+x2)/2
(3) Choisir entre [x1, xm] et [xm, x2] l’intervalle qui possède encore un changement de signe.
(4) Recommencer la deuxième étape avec ce nouvel intervalle

Exemple:
fonction = lambda x: x**3+x**2-3*x-3
intervalle = (1, 2)
tolerance = 10**-8
nmax = 1000
print(bissection(intervalle, fonction, tolerance, nmax))

Note:
Cette méthode fonctionne seulement si l'intervalle choisit est approprié,
c'est-à-dire que l'on a déjà une idée de la valeur de la raciné recherché.

Cas qui ne fonctionnent pas:
- si la fonction est tangente à l'abcisse (exemple: x²)
- si, dans l'intervalle choisie, il y a plus qu'une racine
"""
def bissection(intervalle, f, tolerance, nmax):
    x1, x2 = intervalle
    xmilieu = (x1 + x2)/2
    n = 0
    approximation = lambda x1, x2, xmilieu: abs(x2-x1)/(2*abs(xmilieu)+10**-7)
    while approximation(x1, x2, xmilieu) > tolerance and n <= nmax:

        if f(x1)*f(xmilieu) < 0:    # chg de signe est dans l'intervalle de gauche [x1, xmilieu]
            x2 = xmilieu
        elif f(xmilieu)*f(x2) < 0:  # chg de signe est dans l'intervalle de gauche [xmilieu, x2]
            x1 = xmilieu
        xmilieu = (x1+x2)/2         # On update l'intervalle de recherche
        n += 1                      # Incrémente de +1 itération

    return f"xm = {xmilieu}, f(xm) = {f(xmilieu)}"
