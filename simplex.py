"""
Núcleo del método simplex (método de la Gran M) para programación lineal
con dos variables de decisión y tres restricciones.

Todo el procedimiento está implementado explícitamente (sin linprog ni similares)
y usa aritmética exacta con fracciones. Los números de la fila Z tienen la forma
a + b·M, donde M es un número positivo arbitrariamente grande.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F
from itertools import combinations

SIGNOS_UNICODE = {"<=": "≤", ">=": "≥", "=": "="}
SUB = dict(zip("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def subindice(i):
    """3 -> '₃'"""
    return "".join(SUB[ch] for ch in str(i))


# ----------------------------------------------------------------------------
# Números de la forma a + bM
# ----------------------------------------------------------------------------
class BM:
    """Número a + b·M (M = constante positiva muy grande). Inmutable."""

    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = F(a)
        self.b = F(b)

    def __add__(self, o):
        o = _bm(o)
        return BM(self.a + o.a, self.b + o.b)

    def __sub__(self, o):
        o = _bm(o)
        return BM(self.a - o.a, self.b - o.b)

    def __neg__(self):
        return BM(-self.a, -self.b)

    def __mul__(self, k):  # solo por escalares (Fraction/int)
        k = F(k)
        return BM(self.a * k, self.b * k)

    def __lt__(self, o):  # orden lexicográfico: primero el coeficiente de M
        o = _bm(o)
        return (self.b, self.a) < (o.b, o.a)

    def __eq__(self, o):
        o = _bm(o)
        return self.a == o.a and self.b == o.b

    def __hash__(self):
        return hash((self.a, self.b))

    def es_cero(self):
        return self.a == 0 and self.b == 0


def _bm(x):
    return x if isinstance(x, BM) else BM(x)


# ----------------------------------------------------------------------------
# Formato de números
# ----------------------------------------------------------------------------
def fnum(x, dec=False, nd=4):
    """Formatea una fracción: como fracción exacta o como decimal."""
    if not dec:
        return str(F(x))
    v = round(float(x), nd)
    if v == 0:
        v = 0.0
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def fbm(z, dec=False):
    """Formatea a + bM."""
    z = _bm(z)
    if z.b == 0:
        return fnum(z.a, dec)
    mb = fnum(abs(z.b), dec)
    termino = "M" if mb == "1" else f"{mb}M"
    if z.a == 0:
        return ("-" if z.b < 0 else "") + termino
    return f"{fnum(z.a, dec)} {'-' if z.b < 0 else '+'} {termino}"


def _lin_texto(coefs, nombres, dec=False):
    """Texto de una combinación lineal  c1 n1 + c2 n2 + ..."""
    out = ""
    for v, nom in zip(coefs, nombres):
        v = _bm(v)
        if v.es_cero():
            continue
        if v.a != 0 and v.b != 0:
            neg, mag, mixto = False, v, True
        else:
            neg = (v.b < 0) or (v.b == 0 and v.a < 0)
            mag, mixto = (-v if neg else v), False
        s = fbm(mag, dec)
        if mixto:
            t = f"({s}){nom}"
        elif mag.b != 0:
            t = ("M" if s == "M" else s) + f"·{nom}"
        else:
            t = nom if s == "1" else f"{s}{nom}"
        if not out:
            out = ("-" if neg else "") + t
        else:
            out += (" - " if neg else " + ") + t
    return out or "0"


def texto_lineal(a1, a2, dec=False):
    """a1 x₁ + a2 x₂ en texto unicode."""
    return _lin_texto([F(a1), F(a2)], ["x₁", "x₂"], dec)


def texto_restriccion(a1, a2, signo, b, dec=False):
    return f"{texto_lineal(a1, a2, dec)} {SIGNOS_UNICODE[signo]} {fnum(b, dec)}"


# ----------------------------------------------------------------------------
# Modelo del problema
# ----------------------------------------------------------------------------
@dataclass
class Problema:
    sentido: str                      # "max" o "min"
    c: tuple                          # (c1, c2)
    restricciones: list               # [(a1, a2, signo, b), ...]  signo en {"<=", ">=", "="}

    def __post_init__(self):
        self.c = tuple(F(v) for v in self.c)
        self.restricciones = [(F(a1), F(a2), s, F(b)) for a1, a2, s, b in self.restricciones]


def valor_objetivo(p: Problema, x1, x2):
    return p.c[0] * x1 + p.c[1] * x2


def hay_region_factible(p: Problema) -> bool:
    """Comprueba (con fracciones exactas) si la región factible es no vacía.
    Como x1, x2 >= 0, una región no vacía siempre tiene al menos un vértice."""
    rectas = [(a1, a2, b) for a1, a2, _, b in p.restricciones]
    rectas += [(F(1), F(0), F(0)), (F(0), F(1), F(0))]
    for (a1, a2, b1), (c1, c2, b2) in combinations(rectas, 2):
        det = a1 * c2 - a2 * c1
        if det == 0:
            continue
        x1 = (b1 * c2 - a2 * b2) / det
        x2 = (a1 * b2 - b1 * c1) / det
        if x1 < 0 or x2 < 0:
            continue
        ok = True
        for r1, r2, s, b in p.restricciones:
            v = r1 * x1 + r2 * x2
            if (s == "<=" and v > b) or (s == ">=" and v < b) or (s == "=" and v != b):
                ok = False
                break
        if ok:
            return True
    return False


# ----------------------------------------------------------------------------
# Tabla simplex
# ----------------------------------------------------------------------------
@dataclass
class Tabla:
    z: list                 # fila Z: n+1 elementos BM (última = lado derecho)
    filas: list             # m filas de n+1 Fractions
    base: list              # índice de la variable básica de cada fila

    def copia(self):
        return Tabla(list(self.z), [list(f) for f in self.filas], list(self.base))


@dataclass
class Estandar:
    nombres: list
    A: list
    b: list
    tipos: list                         # por restricción: "holgura", "superavit+art", "art"
    restr_norm: list                    # restricciones tras hacer b >= 0
    notas: list
    artificiales: list                  # índices de columnas artificiales
    sentido: str
    c_orig: tuple
    T_previa: Tabla | None              # tabla inicial antes de hacer ceros las columnas de las artificiales
    T0: Tabla                           # tabla inicial lista para iterar


def estandarizar(p: Problema) -> Estandar:
    """Convierte el problema a forma estándar (maximización, restricciones de igualdad, b >= 0)."""
    inv = {"<=": ">=", ">=": "<=", "=": "="}
    norm, notas = [], []
    for i, (a1, a2, s, b) in enumerate(p.restricciones, start=1):
        if b < 0:
            a1, a2, b, s = -a1, -a2, -b, inv[s]
            notas.append(f"R{i}: el lado derecho era negativo; se multiplicó la restricción por −1 "
                         f"(el sentido de la desigualdad se invierte).")
        norm.append((a1, a2, s, b))

    m = len(norm)
    nombres = ["x₁", "x₂"]
    cols_extra = []     # columnas de holgura / superávit
    cols_art = []       # columnas artificiales
    tipos = []
    for i, (_, _, s, _) in enumerate(norm, start=1):
        sub = subindice(i)
        if s == "<=":
            cols_extra.append((i - 1, f"s{sub}", +1))
            tipos.append("holgura")
        elif s == ">=":
            cols_extra.append((i - 1, f"e{sub}", -1))
            cols_art.append((i - 1, f"a{sub}"))
            tipos.append("superávit + artificial")
        else:
            cols_art.append((i - 1, f"a{sub}"))
            tipos.append("artificial")

    for _, nom, _ in cols_extra:
        nombres.append(nom)
    for _, nom in cols_art:
        nombres.append(nom)
    n = len(nombres)

    A = [[F(0)] * n for _ in range(m)]
    b = [r[3] for r in norm]
    base = [None] * m
    for i, (a1, a2, _, _) in enumerate(norm):
        A[i][0], A[i][1] = a1, a2
    for k, (i, _, signo) in enumerate(cols_extra):
        A[i][2 + k] = F(signo)
        if signo == +1:
            base[i] = 2 + k
    art_idx = []
    for k, (i, _) in enumerate(cols_art):
        j = 2 + len(cols_extra) + k
        A[i][j] = F(1)
        base[i] = j
        art_idx.append(j)

    # Fila Z del problema en forma de maximización:  Z - c·x + M·(artificiales) = 0
    c = p.c if p.sentido == "max" else (-p.c[0], -p.c[1])
    z = [BM(0) for _ in range(n + 1)]
    z[0], z[1] = BM(-c[0]), BM(-c[1])
    for j in art_idx:
        z[j] = BM(0, 1)

    filas = [A[i] + [b[i]] for i in range(m)]
    T_prev = Tabla(list(z), [list(f) for f in filas], list(base)) if art_idx else None

    # Eliminar M de las columnas de las variables artificiales básicas
    for i in range(m):
        if base[i] in art_idx:
            z = [z[j] - BM(0, filas[i][j]) for j in range(n + 1)]
    T0 = Tabla(z, filas, base)

    return Estandar(nombres, A, b, tipos, norm, notas, art_idx, p.sentido, p.c, T_prev, T0)


def sistema_texto(sd: Estandar, dec=False):
    """Líneas de texto con el sistema de ecuaciones aumentado."""
    n = len(sd.nombres)
    lineas = []
    if sd.sentido == "max":
        lineas.append("Maximizar Z")
        cz = [-sd.c_orig[0], -sd.c_orig[1]]
        titulo = "Z"
    else:
        lineas.append("Minimizar Z  ⇔  Maximizar Z' = −Z")
        cz = [sd.c_orig[0], sd.c_orig[1]]
        titulo = "Z'"
    for j in range(2, n):
        cz.append(BM(0, 1) if j in sd.artificiales else BM(0))
    resto = _lin_texto(cz, sd.nombres, dec)
    lineas.append(f"{titulo} {'+ ' + resto if resto != '0' and not resto.startswith('-') else ('- ' + resto[1:] if resto.startswith('-') else '')} = 0".replace("  ", " "))
    for i in range(len(sd.A)):
        lineas.append(f"{_lin_texto(sd.A[i], sd.nombres, dec)} = {fnum(sd.b[i], dec)}")
    return lineas


# ----------------------------------------------------------------------------
# Algoritmo simplex
# ----------------------------------------------------------------------------
@dataclass
class Resultado:
    estado: str                          # "optimo", "no_acotado", "infactible"
    estandar: Estandar
    pasos: list = field(default_factory=list)
    valores: dict = field(default_factory=dict)      # nombre -> Fraction
    z: F | None = None                   # valor óptimo de la función objetivo original
    optimos_alternativos: bool = False
    degenerada: bool = False
    mensaje: str = ""


def _pivotar(T: Tabla, r: int, c: int, dec):
    """Operaciones de Gauss–Jordan. Devuelve el texto de las operaciones."""
    ops = []
    p = T.filas[r][c]
    if p != 1:
        ops.append(f"F{r + 1} ← F{r + 1} ÷ {fnum(p, dec)}")
    T.filas[r] = [v / p for v in T.filas[r]]
    for i in range(len(T.filas)):
        if i == r:
            continue
        f = T.filas[i][c]
        if f != 0:
            ops.append(f"F{i + 1} ← F{i + 1} − ({fnum(f, dec)})·F{r + 1}")
            T.filas[i] = [v - f * w for v, w in zip(T.filas[i], T.filas[r])]
    f = T.z[c]
    if not f.es_cero():
        ops.append(f"F_Z ← F_Z − ({fbm(f, dec)})·F{r + 1}")
        T.z = [v - f * w for v, w in zip(T.z, T.filas[r])]
    T.base[r] = c
    return ops


def resolver(p: Problema, max_iter: int = 100, dec: bool = False) -> Resultado:
    sd = estandarizar(p)
    T = sd.T0.copia()
    n = len(sd.nombres)
    m = len(T.filas)
    res = Resultado("optimo", sd)
    estado = None

    for it in range(max_iter):
        # 1) Variable de entrada: coeficiente más negativo en la fila Z
        col, mejor = None, BM(0)
        for j in range(n):
            if T.z[j] < mejor:
                mejor, col = T.z[j], j
        if col is None:
            estado = "optimo"
            res.pasos.append(dict(tabla=T.copia(), col=None, fila=None, razones=None, pivote=None, ops=[]))
            break

        # 2) Prueba de la razón mínima
        razones = [(T.filas[i][n] / T.filas[i][col]) if T.filas[i][col] > 0 else None for i in range(m)]
        validas = [(r, i) for i, r in enumerate(razones) if r is not None]
        if not validas:
            estado = "no_acotado"
            res.pasos.append(dict(tabla=T.copia(), col=col, fila=None, razones=razones, pivote=None, ops=[]))
            break
        rmin = min(r for r, _ in validas)
        candidatas = [i for r, i in validas if r == rmin]
        fila = min(candidatas, key=lambda i: T.base[i])      # desempate: menor índice de variable

        paso = dict(tabla=T.copia(), col=col, fila=fila, razones=razones,
                    pivote=T.filas[fila][col], sale=T.base[fila])
        paso["ops"] = _pivotar(T, fila, col, dec)
        res.pasos.append(paso)
    else:
        estado = "max_iter"

    res.estado = estado
    T_fin = T

    if estado == "optimo":
        vals = [F(0)] * n
        for i, bv in enumerate(T_fin.base):
            vals[bv] = T_fin.filas[i][n]
        if any(vals[j] > 0 for j in sd.artificiales):
            res.estado = "infactible"
            res.mensaje = ("El óptimo del problema aumentado conserva al menos una variable artificial con valor "
                           "positivo: el sistema de restricciones no tiene solución.")
            return res
        res.valores = {nom: vals[j] for j, nom in enumerate(sd.nombres)}
        zmax = T_fin.z[n].a
        res.z = zmax if p.sentido == "max" else -zmax
        # soluciones alternativas: variable no básica (no artificial) con coeficiente 0 en la fila Z
        res.optimos_alternativos = any(
            T_fin.z[j].es_cero() for j in range(n) if j not in T_fin.base and j not in sd.artificiales)
        res.degenerada = any(T_fin.filas[i][n] == 0 for i in range(m))
    elif estado == "no_acotado":
        vals = [F(0)] * n
        for i, bv in enumerate(T_fin.base):
            vals[bv] = T_fin.filas[i][n]
        if any(vals[j] > 0 for j in sd.artificiales) and not hay_region_factible(p):
            res.estado = "infactible"
            res.mensaje = "No existe ningún punto que cumpla todas las restricciones."
        else:
            res.mensaje = ("La variable que entra tiene todos los coeficientes de su columna ≤ 0: puede crecer "
                           "indefinidamente sin violar restricciones y la función objetivo mejora sin límite.")
    else:
        res.mensaje = f"Se alcanzó el máximo de {max_iter} iteraciones (posible ciclaje)."
    return res


def clasificar(p: Problema, res: Resultado) -> list:
    """Líneas de texto que describen a qué categoría pertenece el problema."""
    sd = res.estandar
    n_h = sd.tipos.count("holgura")
    n_s = sd.tipos.count("superávit + artificial")
    n_art = len(sd.artificiales)

    tipo = "Maximización" if p.sentido == "max" else "Minimización"
    if n_art == 0:
        forma = "solo variables de holgura"
    else:
        partes = []
        if n_h:
            partes.append(f"{n_h} holgura{'s' if n_h > 1 else ''}")
        if n_s:
            partes.append(f"{n_s} superávit")
        partes.append(f"{n_art} artificial{'es' if n_art > 1 else ''}")
        forma = ", ".join(partes) + " (método de la M)"

    if res.estado == "optimo":
        if res.optimos_alternativos:
            resultado = "óptimos alternativos (infinitas soluciones óptimas)"
        else:
            resultado = "solución óptima única"
        if res.degenerada:
            resultado += ", solución degenerada"
    elif res.estado == "infactible":
        resultado = "infactible (región factible vacía)"
    elif res.estado == "no_acotado":
        resultado = "no acotado (Z mejora sin límite)"
    else:
        resultado = "no se alcanzó el óptimo (posible ciclaje)"

    return [f"**Tipo:** {tipo}", f"**Variables añadidas:** {forma}", f"**Resultado:** {resultado}"]


# ----------------------------------------------------------------------------
# Verificación rápida desde la terminal:  python simplex.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ejemplo = Problema("max", (3, 5), [(1, 0, "<=", 4), (0, 2, "<=", 12), (3, 2, "<=", 18)])
    r = resolver(ejemplo)
    print(r.estado, r.z, {k: str(v) for k, v in r.valores.items()})