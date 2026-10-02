import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from modelo import (
    FEATURES,
    TARGET,
    cargar_datos,
    entrenar_modelo,
    predecir,
    preparar_datos,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="UTSmart IA",
    page_icon=":material/landslide:",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
GEOJSON_FILE = BASE_DIR / "datos" / "santander_municipios.geojson"

MESES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>
    .main { background-color: #F5F7FA; }
    .block-container { padding-top: 1.5rem; }

    .kpi {
        background: white;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 3px 12px rgba(0,0,0,.08);
        text-align: center;
        min-height: 125px;
        border: 1px solid #E2E8F0;
    }

    .kpi-title {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 29px;
        font-weight: 700;
        color: #2563EB;
    }

    .danger { color: #DC2626; }

    .card {
        background: white;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CARGA Y MODELO
# ============================================================

@st.cache_data
def cargar():
    train, test = cargar_datos()
    return preparar_datos(train, test)


@st.cache_resource
def obtener_modelo(train):
    return entrenar_modelo(train)


try:
    train, test = cargar()
    modelo = obtener_modelo(train)
except Exception as error:
    st.error("No fue posible cargar los datos o entrenar el modelo.")
    st.exception(error)
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("UTSmart IA")
st.sidebar.markdown("### Sistema de análisis de movimientos en masa")
st.sidebar.divider()

threshold = st.sidebar.slider(
    "Umbral de predicción",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)

mes_seleccionado = st.sidebar.selectbox(
    "Mes para analizar",
    options=list(MESES.keys()),
    index=9,  # Octubre: requisito de la prueba
    format_func=lambda x: MESES[x],
)

st.sidebar.divider()
st.sidebar.markdown("**Entrenamiento:** 2015–2024")
st.sidebar.markdown("**Evaluación:** 2025")


# ============================================================
# PREDICCIONES RANDOM FOREST
# ============================================================

resultado = predecir(
    modelo,
    test,
    threshold=threshold,
).copy()

# El modelo original devuelve la probabilidad y una predicción.
# Se recalcula la predicción aquí para que responda al slider.
resultado["prediccion"] = (
    resultado["probabilidad"] >= threshold
).astype(int)


y_true = resultado[TARGET].astype(int)
y_pred = resultado["prediccion"].astype(int)

recall = recall_score(y_true, y_pred, zero_division=0)
precision = precision_score(y_true, y_pred, zero_division=0)
accuracy = accuracy_score(y_true, y_pred)


# ============================================================
# LÍNEA BASE 2024
# ============================================================

# Requisito: predecir en 2025 lo mismo que ocurrió en el mismo
# municipio y mismo mes del año anterior (2024).
base_2024 = train.loc[
    train["anio"] == 2024,
    ["codigo_dane", "mes", TARGET],
].copy()

base_2024 = base_2024.rename(
    columns={TARGET: "prediccion_base"}
)

# Seguridad: una fila por municipio y mes.
base_2024 = (
    base_2024
    .groupby(["codigo_dane", "mes"], as_index=False)["prediccion_base"]
    .max()
)

evaluacion = resultado.merge(
    base_2024,
    on=["codigo_dane", "mes"],
    how="left",
)

evaluacion["prediccion_base"] = (
    evaluacion["prediccion_base"]
    .fillna(0)
    .astype(int)
)

y_base = evaluacion["prediccion_base"]

recall_base = recall_score(y_true, y_base, zero_division=0)
precision_base = precision_score(y_true, y_base, zero_division=0)
accuracy_base = accuracy_score(y_true, y_base)


# ============================================================
# FUNCIONES DE INTERFAZ
# ============================================================

def kpi(titulo, valor, clase=""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-title">{titulo}</div>
            <div class="kpi-value {clase}">{valor}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cargar_geojson():
    if not GEOJSON_FILE.exists():
        return None
    try:
        with open(GEOJSON_FILE, "r", encoding="utf-8") as archivo:
            return json.load(archivo)
    except Exception:
        return None


def detectar_campo_municipio(geojson):
    if not geojson or not geojson.get("features"):
        return None

    propiedades = geojson["features"][0].get("properties", {})

    candidatos = [
        "municipio", "MUNICIPIO", "Municipio",
        "nombre", "NOMBRE", "name", "NAME",
        "nom_mpio", "MPIO_CNMBR", "MPIO_NOMBR",
        "nombre_municipio",
    ]

    for campo in candidatos:
        if campo in propiedades:
            return campo

    return None


# ============================================================
# ENCABEZADO
# ============================================================

st.title("UTSmart IA")
st.subheader("Sistema de predicción y visualización de movimientos en masa")

st.markdown(
    "El sistema utiliza un **Random Forest** entrenado con datos de "
    "2015–2024 y evaluado con datos de 2025."
)


# ============================================================
# KPIs
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    kpi("Municipios", resultado["municipio"].nunique())
with c2:
    kpi("Registros entrenamiento", f"{len(train):,}")
with c3:
    kpi("Recall", f"{recall:.2%}")
with c4:
    kpi("Precisión", f"{precision:.2%}")
with c5:
    kpi("Predicciones positivas", int(y_pred.sum()), "danger")

st.write("")


# ============================================================
# TABS
# ============================================================

tab_dashboard, tab_municipios, tab_evaluacion, tab_analisis, tab_modelo, tab_mapa = st.tabs(
    [
        ":material/dashboard: Dashboard",
        ":material/location_city: Municipios",
        ":material/compare_arrows: Evaluación",
        ":material/analytics: Análisis",
        ":material/model_training: Modelo",
        ":material/map: Mapa",
    ]
)


# ============================================================
# TAB 1 - DASHBOARD
# ============================================================

with tab_dashboard:
    st.header(f"Resumen — {MESES[mes_seleccionado]}")

    c1, c2 = st.columns(2)

    with c1:
        distribucion = train[TARGET].value_counts().reset_index()
        distribucion.columns = ["hubo_mm", "cantidad"]
        distribucion["hubo_mm"] = distribucion["hubo_mm"].map({
            0: "No ocurrió",
            1: "Ocurrió",
        })

        fig = px.pie(
            distribucion,
            names="hubo_mm",
            values="cantidad",
            hole=0.45,
            title="Distribución histórica",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="dashboard_distribucion",
        )

    with c2:
        datos_mes = resultado[
            resultado["mes"] == mes_seleccionado
        ].sort_values("probabilidad").tail(10)

        fig = px.bar(
            datos_mes,
            x="probabilidad",
            y="municipio",
            orientation="h",
            text=datos_mes["probabilidad"].map(lambda x: f"{x:.1%}"),
            title=f"Top 10 — {MESES[mes_seleccionado]} 2025",
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="dashboard_top10",
        )

    lluvia = (
        train.groupby("mes")["lluvia_mm"]
        .mean()
        .reset_index()
    )
    lluvia["nombre_mes"] = lluvia["mes"].map(MESES)

    fig = px.line(
        lluvia,
        x="nombre_mes",
        y="lluvia_mm",
        markers=True,
        title="Precipitación promedio por mes",
    )
    fig.update_layout(
        xaxis_title="Mes",
        yaxis_title="Lluvia (mm)",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="dashboard_lluvia",
    )


# ============================================================
# TAB 2 - MUNICIPIOS
# ============================================================

with tab_municipios:
    st.header(f"Municipios — {MESES[mes_seleccionado]} 2025")

    if mes_seleccionado == 10:
        st.success(
            "Requisito 4: se muestran los diez municipios con mayor "
            "probabilidad estimada para octubre de 2025."
        )
    else:
        st.info(
            "La prueba solicita octubre de 2025. El selector permite "
            "explorar otros meses de 2025."
        )

    st.write(f"Umbral actual: **{threshold:.0%}**")

    datos_mes = resultado[
        resultado["mes"] == mes_seleccionado
    ].sort_values("probabilidad", ascending=False)

    top10 = datos_mes.head(10).copy()

    st.subheader("Top 10 municipios")

    tabla = top10[
        [
            "municipio",
            "probabilidad",
            "prediccion",
            "lluvia_mm_mes_anterior",
            "eventos_12m",
            "altitud_m",
        ]
    ].copy()

    tabla["probabilidad"] = tabla["probabilidad"].map(lambda x: f"{x:.2%}")
    tabla["prediccion"] = tabla["prediccion"].map({0: "No", 1: "Sí"})

    tabla.insert(0, "Posición", range(1, len(tabla) + 1))

    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Descargar Top 10 CSV",
        data=top10.to_csv(index=False).encode("utf-8"),
        file_name=f"top10_{MESES[mes_seleccionado].lower()}_2025.csv",
        mime="text/csv",
    )

    st.subheader("Todos los municipios del mes")
    st.dataframe(
        datos_mes,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 3 - EVALUACIÓN
# ============================================================

with tab_evaluacion:
    st.header("Evaluación del modelo")

    st.markdown(
        """
        <div class="card">
        <b>Requisito 1 — Separación temporal</b><br>
        El Random Forest se entrena exclusivamente con 2015–2024 y se
        evalúa exclusivamente con 2025. No se mezclan los años.
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b = st.columns(2)
    with a:
        kpi("Periodo de entrenamiento", "2015–2024")
    with b:
        kpi("Periodo de evaluación", "2025")

    st.divider()

    st.markdown(
        """
        <div class="card">
        <b>Requisito 2 — Línea base</b><br>
        La línea base predice que cada municipio tendrá en 2025 el mismo
        resultado que tuvo en el mismo mes de 2024.
        </div>
        """,
        unsafe_allow_html=True,
    )

    comparacion = pd.DataFrame({
        "Modelo": ["Random Forest", "Línea base 2024"],
        "Recall": [recall, recall_base],
        "Precisión": [precision, precision_base],
        "Accuracy": [accuracy, accuracy_base],
    })

    tabla_comp = comparacion.copy()
    for columna in ["Recall", "Precisión", "Accuracy"]:
        tabla_comp[columna] = tabla_comp[columna].map(lambda x: f"{x:.2%}")

    st.dataframe(
        tabla_comp,
        use_container_width=True,
        hide_index=True,
    )

    grafico_comp = comparacion.melt(
        id_vars="Modelo",
        value_vars=["Recall", "Precisión"],
        var_name="Métrica",
        value_name="Valor",
    )

    fig = px.bar(
        grafico_comp,
        x="Modelo",
        y="Valor",
        color="Métrica",
        barmode="group",
        text=grafico_comp["Valor"].map(lambda x: f"{x:.1%}"),
        title="Random Forest vs línea base 2024",
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="evaluacion_comparacion",
    )

    st.divider()

    st.markdown(
        """
        <div class="card">
        <b>Requisito 3 — Recall y precisión</b><br>
        Se reportan ambas métricas para el Random Forest y para la línea base.
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b = st.columns(2)
    with a:
        kpi("Recall — Random Forest", f"{recall:.2%}")
    with b:
        kpi("Precisión — Random Forest", f"{precision:.2%}")

    st.info(
        "Métrica priorizada: Recall. Se prioriza porque interesa detectar "
        "la mayor cantidad posible de casos reales de movimientos en masa. "
        "Un falso negativo significa no detectar un caso que realmente ocurrió."
    )

    st.subheader("Matriz de confusión — Random Forest")

    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predijo NO", "Predijo SÍ"],
        y=["Real NO", "Real SÍ"],
        title="Matriz de confusión",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="evaluacion_confusion",
    )

    st.subheader("Cómo se calcula la línea base")
    st.code(
        "Predicción 2025 = resultado del mismo municipio + mismo mes en 2024",
        language="text",
    )


# ============================================================
# TAB 4 - ANÁLISIS
# ============================================================

with tab_analisis:
    st.header("Análisis de los datos")

    fig = px.histogram(
        resultado,
        x="probabilidad",
        nbins=20,
        title="Distribución de probabilidades",
    )
    fig.update_xaxes(tickformat=".0%", title="Probabilidad")
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="analisis_histograma",
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.scatter(
            resultado,
            x="lluvia_mm_mes_anterior",
            y="probabilidad",
            color="prediccion",
            hover_name="municipio",
            title="Lluvia del mes anterior vs probabilidad",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="analisis_lluvia",
        )

    with c2:
        fig = px.scatter(
            resultado,
            x="altitud_m",
            y="probabilidad",
            color="prediccion",
            hover_name="municipio",
            title="Altitud vs probabilidad",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="analisis_altitud",
        )

    fig = px.scatter(
        resultado,
        x="eventos_12m",
        y="probabilidad",
        size="lluvia_mm_mes_anterior",
        color="prediccion",
        hover_name="municipio",
        title="Eventos últimos 12 meses vs probabilidad",
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="analisis_eventos",
    )


# ============================================================
# TAB 5 - MODELO
# ============================================================

with tab_modelo:
    st.header("Información del modelo")

    st.markdown(
        """
        ### Random Forest

        Se conserva el modelo original definido en `modelo.py`.

        - Algoritmo: Random Forest Classifier
        - `n_estimators`: 300
        - `class_weight`: balanced
        - `random_state`: 42
        - `n_jobs`: -1
        """
    )

    st.subheader("Variables utilizadas")

    variables = pd.DataFrame({
        "Variable": FEATURES,
        "Descripción": [
            "Lluvia registrada en el mes anterior",
            "Cantidad de eventos en los últimos 12 meses",
            "Altitud del municipio",
            "Mes del año",
        ],
    })

    st.dataframe(
        variables,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Importancia de las variables")

    importancia = pd.DataFrame({
        "Variable": FEATURES,
        "Importancia": modelo.feature_importances_,
    }).sort_values("Importancia")

    fig = px.bar(
        importancia,
        x="Importancia",
        y="Variable",
        orientation="h",
        text=importancia["Importancia"].map(lambda x: f"{x:.2%}"),
        title="Importancia de variables del Random Forest",
    )
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="modelo_importancia",
    )

    st.subheader("Matriz de confusión")
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predijo NO", "Predijo SÍ"],
        y=["Real NO", "Real SÍ"],
        title="Matriz de confusión",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="modelo_confusion",
    )


# ============================================================
# TAB 6 - MAPA
# ============================================================

with tab_mapa:
    st.header(f"Mapa — {MESES[mes_seleccionado]} 2025")

    geojson = cargar_geojson()

    if geojson is None:
        st.warning("No se encontró `datos/santander_municipios.geojson`.")
    else:
        campo_municipio = detectar_campo_municipio(geojson)

        if campo_municipio is None:
            st.warning(
                "Se encontró el GeoJSON, pero no se identificó automáticamente "
                "el campo con el nombre del municipio."
            )
            if geojson.get("features"):
                st.json(geojson["features"][0].get("properties", {}))
        else:
            try:
                import folium
                from streamlit_folium import st_folium
            except ImportError:
                st.error(
                    "Falta streamlit-folium. Ejecuta: "
                    "python -m pip install streamlit-folium"
                )
                st.stop()

            mapa_datos = resultado[
                resultado["mes"] == mes_seleccionado
            ][["municipio", "probabilidad", "prediccion"]].copy()

            mapa_datos["municipio"] = (
                mapa_datos["municipio"].astype(str).str.strip().str.upper()
            )

            mapa = folium.Map(
                location=[6.9, -73.1],
                zoom_start=8,
                tiles="OpenStreetMap",
            )

            for feature in geojson.get("features", []):
                nombre_geo = str(
                    feature.get("properties", {}).get(campo_municipio, "")
                ).strip().upper()

                fila = mapa_datos[mapa_datos["municipio"] == nombre_geo]
                if fila.empty:
                    continue

                probabilidad = float(fila.iloc[0]["probabilidad"])
                prediccion = int(fila.iloc[0]["prediccion"])

                if probabilidad >= 0.80:
                    color = "red"
                elif probabilidad >= 0.60:
                    color = "orange"
                elif probabilidad >= 0.40:
                    color = "yellow"
                else:
                    color = "green"

                popup = folium.Popup(
                    f"<b>{nombre_geo}</b><br><br>"
                    f"Probabilidad: <b>{probabilidad:.2%}</b><br>"
                    f"Predicción: <b>{'Movimiento en masa' if prediccion else 'No movimiento'}</b>",
                    max_width=300,
                )

                folium.GeoJson(
                    feature,
                    style_function=lambda _feature, color=color: {
                        "fillColor": color,
                        "color": "#333333",
                        "weight": 1,
                        "fillOpacity": 0.65,
                    },
                    popup=popup,
                    tooltip=nombre_geo,
                ).add_to(mapa)

            st_folium(mapa, width=None, height=650)

            st.markdown(
                """
                **Interpretación del mapa**

                - 0%–39%: probabilidad baja
                - 40%–59%: probabilidad moderada
                - 60%–79%: probabilidad alta
                - 80%–100%: probabilidad muy alta
                """
            )


# ============================================================
# PIE
# ============================================================

st.divider()
st.caption("UTSmart IA Challenge 2026 | Random Forest + línea base 2024")
