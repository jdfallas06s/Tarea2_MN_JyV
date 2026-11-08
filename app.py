"""
Herramienta de programación lineal (2 variables, 3 restricciones)
Método simplex (Gran M) y método gráfico.

IF0017 – Métodos Numéricos y Análisis Computacional · Tarea II · II Ciclo 2026
Ejecutar con:   streamlit run app.py
"""
from fractions import Fraction as F

import pandas as pd
import streamlit as st

import grafico
from simplex import (Problema, fbm, fnum, resolver, sistema_texto, subindice,
                     texto_restriccion, valor_objetivo)

st.set_page_config(page_title="Programación lineal", layout="centered")

# Estilo mínimo: contenedor más angosto, títulos discretos y sin barra superior de Streamlit.
st.markdown("""
<style>
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 1000px; padding-top: 2rem; }
h1 { font-size: 1.7rem; font-weight: 600; }
h2 { font-size: 1.3rem; font-weight: 600; border-bottom: 1px solid #e3e5e8; padding-bottom: .3rem; }
h3 { font-size: 1.05rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Ejemplos (sirven también como problemas de verificación)
# ----------------------------------------------------------------------------
EJEMPLOS = {
    "— Problema propio —": None,
    "1. Maximización (solo holguras)": dict(
        sentido="Maximizar", c=(3, 5),
        r=[(1, 0, "≤", 4), (0, 2, "≤", 12), (3, 2, "≤", 18)]),
    "2. Minimización (superávit + artificiales)": dict(
        sentido="Minimizar", c=(2, 3),
        r=[(1, 1, "≥", 4), (1, 3, "≥", 6), (1, 0, "≤", 5)]),
    "3. Variables artificiales (≤, = y ≥)": dict(
        sentido="Minimizar", c=(0.4, 0.5),
        r=[(0.3, 0.1, "≤", 2.7), (0.5, 0.5, "=", 6), (0.6, 0.4, "≥", 6)]),
    "4. Infactible (maximizar)": dict(
        sentido="Maximizar", c=(1, 1),
        r=[(1, 1, "≤", 2), (1, 1, "≥", 5), (1, 0, "≤", 10)]),
    "5. No acotado (maximizar)": dict(
        sentido="Maximizar", c=(1, 1),
        r=[(1, -1, "≤", 2), (0, 1, "≥", 1), (1, 1, "≥", 1)]),
    "6. Óptimos alternativos (maximizar)": dict(
        sentido="Maximizar", c=(2, 4),
        r=[(1, 2, "≤", 8), (1, 0, "≤", 6), (0, 1, "≤", 3)]),
    "7. Infactible (minimizar)": dict(
        sentido="Minimizar", c=(2, 3),
        r=[(1, 1, "≤", 4), (1, 1, "≥", 6), (1, 0, "≤", 10)]),
    "8. No acotado (minimizar)": dict(
        sentido="Minimizar", c=(-2, -1),
        r=[(1, -1, "≤", 3), (1, 0, "≥", 1), (0, 1, "≥", 0)]),
    "9. Óptimos alternativos (minimizar)": dict(
        sentido="Minimizar", c=(1, 1),
        r=[(1, 1, "≥", 4), (1, 0, "≤", 3), (0, 1, "≤", 3)]),
    # --- Presentación de clase ---
    "Clase 1. Procesamiento de gas": dict(
        sentido="Maximizar", c=(150, 175),
        r=[(7, 11, "≤", 77), (10, 8, "≤", 80), (0, 1, "≤", 6)]),          # sin x₁ ≤ 9 (redundante)
    "Clase 2. Dieta Ozark Farms": dict(
        sentido="Minimizar", c=(0.3, 0.9),
        r=[(1, 1, "≥", 800), (21, -30, "≤", 0), (-3, 1, "≤", 0)]),
    "Clase 3. Método M adicional": dict(
        sentido="Minimizar", c=(4, 1),
        r=[(3, 1, "=", 3), (4, 3, "≥", 6), (1, 2, "≤", 4)]),
    "Clase 4. Dualidad: primal del Ejemplo 3": dict(
        sentido="Maximizar", c=(3, 5),
        r=[(1, 2, "≤", 8), (3, 1, "≤", 9), (1, 1, "≤", 10)]),             # 3.ª restricción redundante
    "Clase 5. Dualidad: dual del Ejemplo 3": dict(
        sentido="Minimizar", c=(8, 9),
        r=[(1, 3, "≥", 3), (2, 1, "≥", 5), (1, 1, "≤", 10)]),             # 3.ª restricción redundante
    "Práctica 1": dict(
        sentido="Maximizar", c=(2, 3),
        r=[(1, 3, "≤", 6), (3, 2, "≤", 6), (1, 1, "≤", 10)]),             # 3.ª restricción redundante
    "Práctica 4. Productos A y B": dict(
        sentido="Maximizar", c=(45, 20),
        r=[(20, 5, "≤", 9500), (0.04, 0.12, "≤", 40), (1, 1, "≤", 550)]),
    "Práctica 5": dict(
        sentido="Maximizar", c=(6, 8),
        r=[(5, 2, "≤", 40), (6, 6, "≤", 60), (2, 4, "≤", 32)]),
    "Práctica 7. Método M (maximizar)": dict(
        sentido="Maximizar", c=(4, 1),
        r=[(3, 1, "=", 3), (4, 3, "≥", 6), (1, 2, "≤", 4)]),
    "Práctica 8a. Lado derecho negativo": dict(
        sentido="Maximizar", c=(-5, 2),
        r=[(-1, 1, "≤", -2), (2, 3, "≤", 5), (1, 1, "≤", 10)]),          # 3.ª restricción redundante
    # --- Libro (Taha) ---
    "Libro. Reddy Mikks": dict(
        sentido="Maximizar", c=(5, 4),
        r=[(6, 4, "≤", 24), (1, 2, "≤", 6), (-1, 1, "≤", 1)]),            # sin x₂ ≤ 2 (redundante)
}
SIGNO = {"≤": "<=", "≥": ">=", "=": "="}


def cargar_ejemplo(nombre):
    ej = EJEMPLOS[nombre]
    if ej is None:
        return
    st.session_state["sentido"] = ej["sentido"]
    st.session_state["c1"], st.session_state["c2"] = float(ej["c"][0]), float(ej["c"][1])
    for i, (a1, a2, s, b) in enumerate(ej["r"], start=1):
        st.session_state[f"a{i}1"], st.session_state[f"a{i}2"] = float(a1), float(a2)
        st.session_state[f"s{i}"], st.session_state[f"b{i}"] = s, float(b)


PROPIO = "— Problema propio —"


def a_propio():
    """Si el usuario modifica un dato, el problema deja de ser el ejemplo cargado."""
    st.session_state["ejemplo"] = PROPIO


if "sentido" not in st.session_state:
    cargar_ejemplo("1. Maximización (solo holguras)")


def frac(v):
    """float del widget -> Fraction exacta (0.3 -> 3/10)."""
    return F(str(round(float(v), 10)))


def mostrar_df(df):
    """st.dataframe a todo el ancho (la opción cambió de nombre entre versiones de Streamlit)."""
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


# ----------------------------------------------------------------------------
# LaTeX del modelo
# ----------------------------------------------------------------------------
def lx(v):
    v = F(v)
    if v.denominator == 1:
        return str(v.numerator)
    return ("-" if v < 0 else "") + rf"\frac{{{abs(v.numerator)}}}{{{v.denominator}}}"


def lx_lineal(a1, a2):
    out = ""
    for a, nom in ((a1, "x_1"), (a2, "x_2")):
        a = F(a)
        if a == 0:
            continue
        mag = abs(a)
        t = f"{'' if mag == 1 else lx(mag)}{nom}"
        out += (("-" if a < 0 else "") + t) if not out else ((" - " if a < 0 else " + ") + t)
    return out or "0"


def latex_modelo(p: Problema):
    sg = {"<=": r"\leq", ">=": r"\geq", "=": "="}
    obj = r"\max" if p.sentido == "max" else r"\min"
    lineas = [rf"{obj}\ Z = {lx_lineal(*p.c)}", r"\text{sujeto a:}"]
    for a1, a2, s, b in p.restricciones:
        lineas.append(rf"{lx_lineal(a1, a2)} {sg[s]} {lx(b)}")
    lineas.append(r"x_1,\ x_2 \geq 0")
    return r"\begin{aligned}" + r" \\ ".join(lineas) + r"\end{aligned}"


# ----------------------------------------------------------------------------
# Tablas simplex
# ----------------------------------------------------------------------------
def tabla_df(T, nombres, dec, razones=None, col=None, fila=None):
    datos = [[fbm(v, dec) for v in T.z]] + [[fnum(v, dec) for v in f] for f in T.filas]
    indice = ["Z"] + [f"F{i + 1}: {nombres[b]}" for i, b in enumerate(T.base)]
    df = pd.DataFrame(datos, index=indice, columns=list(nombres) + ["LD"])
    if razones is not None:
        df["Razón"] = ["—"] + [fnum(r, dec) if r is not None else "—" for r in razones]

    def estilo(_):
        css = pd.DataFrame("", index=df.index, columns=df.columns)
        if col is not None:
            css.iloc[:, col] = "background-color: #e8eef6"
        if fila is not None:
            css.iloc[fila + 1, :] = "background-color: #f6ecd9"
        if col is not None and fila is not None:
            css.iloc[fila + 1, col] = "background-color: #cfe5d3; font-weight: bold"
        return css

    return df.style.apply(estilo, axis=None)


# ----------------------------------------------------------------------------
# Encabezado y datos de entrada
# ----------------------------------------------------------------------------
st.title("Programación lineal: simplex y método gráfico")
st.caption("IF0017 · Métodos Numéricos y Análisis Computacional · Tarea II · II Ciclo 2026")

st.header("1. Datos y formulación")
cab = st.columns([3, 2])
cab[0].selectbox("Cargar un ejemplo", list(EJEMPLOS), key="ejemplo",
                 on_change=lambda: cargar_ejemplo(st.session_state["ejemplo"]))
cab[1].radio("Tipo de problema", ["Maximizar", "Minimizar"], key="sentido", horizontal=True,
                 on_change=a_propio)

st.markdown("**Función objetivo**: Z = c₁x₁ + c₂x₂")
f1, f2, _ = st.columns([1, 1, 3])
f1.number_input("c₁", key="c1", step=1.0, format="%g", on_change=a_propio)
f2.number_input("c₂", key="c2", step=1.0, format="%g", on_change=a_propio)

st.markdown("**Restricciones**: a₁x₁ + a₂x₂ (≤, ≥, =) b")
for i in range(1, 4):
    k = st.columns([1, 1, 1, 1])
    k[0].number_input(f"a₁ (R{i})", key=f"a{i}1", step=1.0, format="%g", on_change=a_propio)
    k[1].number_input(f"a₂ (R{i})", key=f"a{i}2", step=1.0, format="%g", on_change=a_propio)
    k[2].selectbox(f"Signo (R{i})", ["≤", "≥", "="], key=f"s{i}", on_change=a_propio)
    k[3].number_input(f"b (R{i})", key=f"b{i}", step=1.0, format="%g", on_change=a_propio)

problema = Problema(
    "max" if st.session_state["sentido"] == "Maximizar" else "min",
    (frac(st.session_state["c1"]), frac(st.session_state["c2"])),
    [(frac(st.session_state[f"a{i}1"]), frac(st.session_state[f"a{i}2"]),
      SIGNO[st.session_state[f"s{i}"]], frac(st.session_state[f"b{i}"])) for i in range(1, 4)],
)

st.markdown("**Modelo matemático**")
st.latex(latex_modelo(problema))

dec = st.radio("Mostrar números como", ["Fracciones exactas", "Decimales"], horizontal=True) == "Decimales"

resultado = resolver(problema, dec=dec)
analisis = grafico.analizar(problema)
sd = resultado.estandar

tab_simplex, tab_grafico, tab_resultados = st.tabs(
    ["2. Simplex", "3. Método gráfico", "4. Resultados"])

# ----------------------------------------------------------------------------
# 2. Simplex
# ----------------------------------------------------------------------------
with tab_simplex:
    st.subheader("Forma estándar")
    st.markdown(
        "- **≤**: se suma una variable de **holgura** sᵢ, que es la básica inicial.\n"
        "- **≥**: se resta una variable de **superávit** eᵢ y se suma una **artificial** aᵢ.\n"
        "- **=**: se suma una variable **artificial** aᵢ.\n"
        "- **Minimizar**: se resuelve como max Z′ = −Z.\n"
        "- **Método de la M**: cada artificial se penaliza con −M en la función objetivo "
        "(M positivo y muy grande), por lo que en la fila Z aparece +M.")
    for nota in sd.notas:
        st.caption(nota)

    st.table(pd.DataFrame({
        "Restricción": [f"R{i + 1}: " + texto_restriccion(a1, a2, s, b, dec)
                        for i, (a1, a2, s, b) in enumerate(sd.restr_norm)],
        "Variables añadidas": sd.tipos}).set_index("Restricción"))

    st.markdown("**Sistema de ecuaciones**")
    st.code("\n".join(sistema_texto(sd, dec)), language=None)

    if sd.T_previa is not None:
        st.markdown("**Tabla inicial** (con +M en las columnas de las artificiales)")
        mostrar_df(tabla_df(sd.T_previa, sd.nombres, dec))
        st.markdown("Las artificiales son básicas, así que su coeficiente en Z debe ser 0: "
                    "se resta M veces cada fila donde hay una artificial básica.")
        st.markdown("**Tabla inicial lista para iterar**")
    else:
        st.markdown("**Tabla inicial** (no hay artificiales)")
    mostrar_df(tabla_df(sd.T0, sd.nombres, dec))

    st.subheader("Iteraciones")
    st.caption("Columna de entrada en azul, fila de salida en beige, pivote en verde.")
    for k, paso in enumerate(resultado.pasos):
        T, col, fila = paso["tabla"], paso["col"], paso["fila"]
        if k == len(resultado.pasos) - 1 and col is None:
            st.markdown(f"### Tabla final (iteración {k})")
            mostrar_df(tabla_df(T, sd.nombres, dec))
            st.markdown("Todos los coeficientes de la fila Z son ≥ 0: se cumple la condición de optimalidad.")
            break
        st.markdown(f"### Iteración {k}")
        mostrar_df(tabla_df(T, sd.nombres, dec, paso["razones"], col, fila))
        st.markdown(f"- **Entra:** {sd.nombres[col]} (coeficiente más negativo en Z: {fbm(T.z[col], dec)}).")
        if fila is None:
            st.warning(f"No hay coeficientes positivos en la columna de {sd.nombres[col]}: no se puede "
                       f"calcular ninguna razón. El problema es no acotado (o infactible, ver resultados).")
            break
        st.markdown(f"- **Razones** (LD ÷ coeficiente, solo coeficientes > 0): la menor está en F{fila + 1}.")
        st.markdown(f"- **Sale:** {sd.nombres[paso['sale']]} · **Pivote:** {fnum(paso['pivote'], dec)} "
                    f"(F{fila + 1}, columna {sd.nombres[col]}).")
        if paso["ops"]:
            st.markdown("- **Operaciones de Gauss–Jordan:**")
            st.code("\n".join(paso["ops"]), language=None)

    if resultado.estado == "max_iter":
        st.error(resultado.mensaje)

# ----------------------------------------------------------------------------
# 3. Método gráfico
# ----------------------------------------------------------------------------
with tab_grafico:
    st.subheader("Región factible y solución óptima")
    if analisis["estado"] == "infactible":
        st.error("La región factible es vacía: ningún punto cumple todas las restricciones.")
    st.pyplot(grafico.dibujar(problema, analisis))

    if analisis["vertices"]:
        st.markdown("**Vértices factibles y valor de Z**")
        filas = []
        for k, (x1, x2) in enumerate(analisis["vertices"], start=1):
            es_opt = any(abs(x1 - o[0]) < 1e-9 and abs(x2 - o[1]) < 1e-9 for o in analisis["optimos"])
            filas.append({"Vértice": f"V{k}", "x₁": round(x1, 6), "x₂": round(x2, 6),
                          "Z": round(float(valor_objetivo(problema, x1, x2)), 6),
                          "Óptimo": "★" if es_opt else ""})
        mostrar_df(pd.DataFrame(filas).set_index("Vértice"))

    st.markdown("**Comparación con el simplex**")
    if analisis["estado"] == "optimo":
        if len(analisis["optimos"]) > 1:
            st.markdown("Varios vértices dan el mismo valor óptimo: hay infinitas soluciones óptimas "
                        "(todo el segmento que los une).")
        if resultado.estado == "optimo":
            zg, zs = analisis["z_opt"], float(resultado.z)
            st.markdown(f"- Método gráfico: Z = **{zg:.6g}**\n- Método simplex: Z = **{zs:.6g}**")
            if abs(zs - zg) < 1e-7 * max(1, abs(zs)):
                st.markdown("Ambos métodos coinciden.")
            else:
                st.error("Los resultados no coinciden: revisar el modelo.")
        else:
            st.error("Los métodos no coinciden en el estado del problema; revisar el modelo.")
    elif analisis["estado"] == "no_acotado":
        st.markdown("La región factible no está acotada y Z mejora indefinidamente: no hay óptimo finito.")
        if resultado.estado != "no_acotado":
            st.error("El simplex no reporta el mismo estado; revisar el modelo.")
    else:
        if resultado.estado == "infactible":
            st.markdown("Ambos métodos coinciden: el problema es infactible.")
        else:
            st.error("El simplex no reporta el mismo estado; revisar el modelo.")

# ----------------------------------------------------------------------------
# 4. Resultados
# ----------------------------------------------------------------------------
with tab_resultados:
    st.subheader("Resultado")
    if resultado.estado == "optimo":
        v = resultado.valores
        c = st.columns(3)
        c[0].metric("x₁", fnum(v["x₁"], True))
        c[1].metric("x₂", fnum(v["x₂"], True))
        c[2].metric("Z óptimo", fnum(resultado.z, True))
        if not dec:
            st.caption(f"Valores exactos: x₁ = {fnum(v['x₁'])}, x₂ = {fnum(v['x₂'])}, Z = {fnum(resultado.z)}")

        st.markdown("**Variables de holgura y superávit**")
        filas = []
        for i in range(len(sd.restr_norm)):
            sub = subindice(i + 1)
            nombre = next((n for n in (f"s{sub}", f"e{sub}") if n in v), None)
            if nombre is None:
                filas.append({"Restricción": f"R{i + 1}", "Variable": "—", "Valor": "0",
                              "Interpretación": "igualdad: siempre activa"})
                continue
            val = v[nombre]
            if val == 0:
                inter = "activa (se cumple con igualdad)"
            elif nombre[0] == "s":
                inter = f"sobran {fnum(val, True)} unidades del recurso"
            else:
                inter = f"se supera el mínimo exigido en {fnum(val, True)} unidades"
            filas.append({"Restricción": f"R{i + 1}", "Variable": nombre,
                          "Valor": fnum(val, True), "Interpretación": inter})
        st.table(pd.DataFrame(filas).set_index("Restricción"))

        st.markdown("**Interpretación**")
        sentido = "crecimiento" if problema.sentido == "max" else "decrecimiento"
        st.markdown(
            f"- El óptimo está en el vértice (x₁, x₂) = ({fnum(v['x₁'], True)}, {fnum(v['x₂'], True)}), "
            f"último punto de la región que toca la recta de nivel de Z en la dirección de {sentido}.")
        if resultado.optimos_alternativos:
            st.markdown("- Una variable no básica tiene coeficiente 0 en la fila Z final: hay "
                        "**soluciones óptimas alternativas**.")
        if resultado.degenerada:
            st.markdown("- Una variable básica vale 0: la solución es **degenerada**.")
        if sd.artificiales:
            st.markdown("- Las artificiales salieron de la base (valen 0): la solución es factible "
                        "para el problema original.")
    elif resultado.estado == "infactible":
        st.error("El problema es infactible.")
        st.markdown(f"- {resultado.mensaje}\n"
                    "- La intersección de los semiplanos de las restricciones con x₁, x₂ ≥ 0 es vacía.")
    elif resultado.estado == "no_acotado":
        st.warning("El problema es no acotado.")
        st.markdown(f"- {resultado.mensaje}\n"
                    "- La región factible no está acotada y existe una dirección a lo largo de la cual "
                    "Z mejora sin límite, así que no hay óptimo finito.")
    else:
        st.error(resultado.mensaje)