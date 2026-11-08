"""
Método gráfico para programación lineal con dos variables.

analizar(p)  -> vértices factibles, estado del problema y óptimo (aritmética exacta).
dibujar(p, a) -> figura de matplotlib con rectas, región factible, vértices y óptimo.
"""
from __future__ import annotations

import math
from fractions import Fraction as F
from itertools import combinations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simplex import Problema, texto_restriccion

COLORES = ["#1f77b4", "#d6604d", "#2ca02c"]


# ----------------------------------------------------------------------------
# Análisis (exacto, con fracciones)
# ----------------------------------------------------------------------------
def _cumple(p: Problema, x1, x2) -> bool:
    if x1 < 0 or x2 < 0:
        return False
    for a1, a2, s, b in p.restricciones:
        v = a1 * x1 + a2 * x2
        if (s == "<=" and v > b) or (s == ">=" and v < b) or (s == "=" and v != b):
            return False
    return True


def _vertices(p: Problema) -> list:
    """Intersecciones de pares de rectas (restricciones y ejes) que son factibles."""
    rectas = [(a1, a2, b) for a1, a2, _, b in p.restricciones]
    rectas += [(F(1), F(0), F(0)), (F(0), F(1), F(0))]
    puntos = set()
    for (a1, a2, b1), (c1, c2, b2) in combinations(rectas, 2):
        det = a1 * c2 - a2 * c1
        if det == 0:
            continue
        x1 = (b1 * c2 - a2 * b2) / det
        x2 = (a1 * b2 - b1 * c1) / det
        if _cumple(p, x1, x2):
            puntos.add((x1, x2))
    return list(puntos)


def _rayos(p: Problema) -> list:
    """Direcciones d >= 0 en las que la región factible no tiene límite."""
    candidatos = [(F(1), F(0)), (F(0), F(1))]
    for a1, a2, _, _ in p.restricciones:
        candidatos += [(a2, -a1), (-a2, a1)]
    rayos = []
    for d1, d2 in candidatos:
        if d1 < 0 or d2 < 0 or (d1 == 0 and d2 == 0):
            continue
        ok = True
        for a1, a2, s, _ in p.restricciones:
            v = a1 * d1 + a2 * d2
            if (s == "<=" and v > 0) or (s == ">=" and v < 0) or (s == "=" and v != 0):
                ok = False
                break
        if ok:
            rayos.append((d1, d2))
    return rayos


def analizar(p: Problema) -> dict:
    """Devuelve un diccionario con: estado ('optimo', 'no_acotado', 'infactible'),
    vertices, optimos (listas de tuplas float), z_opt (float) y rayos."""
    vs = _vertices(p)
    if not vs:
        return dict(estado="infactible", vertices=[], optimos=[], z_opt=None, rayos=[])

    cx = sum(float(v[0]) for v in vs) / len(vs)
    cy = sum(float(v[1]) for v in vs) / len(vs)
    vs.sort(key=lambda v: math.atan2(float(v[1]) - cy, float(v[0]) - cx))

    signo = 1 if p.sentido == "max" else -1
    rayos = _rayos(p)
    mejora = any(signo * (p.c[0] * d1 + p.c[1] * d2) > 0 for d1, d2 in rayos)
    vertices = [(float(x), float(y)) for x, y in vs]
    if mejora:
        return dict(estado="no_acotado", vertices=vertices, optimos=[], z_opt=None, rayos=rayos)

    z = [p.c[0] * x + p.c[1] * y for x, y in vs]
    mejor = max(signo * v for v in z)
    optimos = [vertices[i] for i, v in enumerate(z) if signo * v == mejor]
    return dict(estado="optimo", vertices=vertices, optimos=optimos,
                z_opt=float(signo * mejor), rayos=rayos)


# ----------------------------------------------------------------------------
# Dibujo
# ----------------------------------------------------------------------------
def _limites(p: Problema, a: dict):
    xs, ys = [1.0], [1.0]
    for x, y in a["vertices"]:
        xs.append(x)
        ys.append(y)
    for a1, a2, _, b in p.restricciones:
        if a1 != 0 and b / a1 > 0:
            xs.append(float(b / a1))
        if a2 != 0 and b / a2 > 0:
            ys.append(float(b / a2))
    if a["estado"] == "no_acotado":   # más espacio para ver hacia dónde se abre la región
        lado = max(max(xs), max(ys)) * 1.8
        return lado, lado
    return max(xs) * 1.25, max(ys) * 1.25


def dibujar(p: Problema, a: dict):
    xmax, ymax = _limites(p, a)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # región factible (máscara sobre una malla; sirve también si no está acotada)
    X, Y = np.meshgrid(np.linspace(0, xmax, 500), np.linspace(0, ymax, 500))
    mask = np.ones_like(X, dtype=bool)
    tol = 1e-9
    for a1, a2, s, b in p.restricciones:
        v = float(a1) * X + float(a2) * Y
        if s == "<=":
            mask &= v <= float(b) + tol
        elif s == ">=":
            mask &= v >= float(b) - tol
        else:
            mask &= np.abs(v - float(b)) <= 0.01 * max(xmax, ymax)
    if mask.any() and not all(s == "=" for _, _, s, _ in p.restricciones):
        ax.contourf(X, Y, mask.astype(int), levels=[0.5, 1.5], colors=["#9ecae1"], alpha=0.45)

    # rectas de las restricciones
    t = np.linspace(0, max(xmax, ymax) * 2, 2)
    for i, (a1, a2, s, b) in enumerate(p.restricciones):
        color = COLORES[i % len(COLORES)]
        etiqueta = f"R{i + 1}: {texto_restriccion(a1, a2, s, b)}"
        estilo = "--" if s == "=" else "-"
        if a2 != 0:
            xs = np.array([0.0, xmax])
            ax.plot(xs, (float(b) - float(a1) * xs) / float(a2), estilo, color=color, lw=1.6, label=etiqueta)
        elif a1 != 0:
            ax.axvline(float(b / a1), ls=estilo, color=color, lw=1.6, label=etiqueta)

    # vértices
    for k, (x, y) in enumerate(a["vertices"], start=1):
        es_opt = any(abs(x - o[0]) < 1e-9 and abs(y - o[1]) < 1e-9 for o in a["optimos"])
        ax.plot(x, y, "o", color="#222", ms=5, zorder=5)
        ax.annotate(f"V{k}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9,
                    weight="bold" if es_opt else "normal")

    # óptimo y recta de nivel
    if a["estado"] == "optimo":
        for j, (x, y) in enumerate(a["optimos"]):
            ax.plot(x, y, "*", color="#e08a00", ms=16, zorder=6, mec="#222",
                    label=f"Óptimo  Z = {a['z_opt']:.6g}" if j == 0 else None)
        c1, c2 = float(p.c[0]), float(p.c[1])
        z = a["z_opt"]
        if c2 != 0:
            xs = np.array([0.0, xmax])
            ax.plot(xs, (z - c1 * xs) / c2, ":", color="#444", lw=1.4, label="Recta de nivel de Z")
        elif c1 != 0:
            ax.axvline(z / c1, ls=":", color="#444", lw=1.4, label="Recta de nivel de Z")
    elif a["estado"] == "no_acotado":
        ax.text(0.5, 0.95, "Región no acotada: Z mejora sin límite", transform=ax.transAxes,
                ha="center", va="top", fontsize=10, color="#a33")
    else:
        ax.text(0.5, 0.5, "Región factible vacía", transform=ax.transAxes,
                ha="center", va="center", fontsize=13, color="#a33")

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.grid(alpha=0.25)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    return fig
