# UTSmart IA --- Dashboard de Predicción y Visualización

## 1. Descripción del proyecto

**UTSmart IA** es una aplicación web desarrollada en Python para
analizar y visualizar datos relacionados con movimientos en masa en
municipios de Santander.

La aplicación utiliza un modelo de **Random Forest Classifier** para
estimar la probabilidad de que ocurra un movimiento en masa y presenta
los resultados mediante un dashboard interactivo construido con
**Streamlit** y **Plotly**.

El proyecto está diseñado para cumplir los requerimientos de evaluación
temporal y comparación solicitados para el reto:

1.  Entrenar el modelo utilizando los años **2015--2024**.
2.  Evaluar el modelo utilizando exclusivamente **2025**.
3.  No mezclar los años de entrenamiento con los años de evaluación.
4.  Comparar el Random Forest contra una línea base que repite lo
    ocurrido en el **mismo municipio y mismo mes de 2024**.
5.  Reportar **Recall** y **Precisión**.
6.  Explicar la métrica priorizada.
7.  Mostrar los **10 municipios con mayor probabilidad para octubre de
    2025**, que corresponde al primer mes de evaluación considerado en
    la prueba proporcionada.
8.  Presentar los resultados mediante una aplicación visual con tablas,
    gráficos y mapa.

------------------------------------------------------------------------

# 2. Tecnologías utilizadas

El proyecto utiliza las siguientes tecnologías:

  Tecnología         Uso
  ------------------ ----------------------------------------------
  Python             Lenguaje principal
  Streamlit          Construcción del dashboard web
  Pandas             Lectura, limpieza y manipulación de datos
  NumPy              Operaciones numéricas y soporte científico
  Scikit-learn       Entrenamiento y evaluación del Random Forest
  Plotly             Gráficos interactivos
  Folium             Visualización geográfica
  streamlit-folium   Integración del mapa Folium con Streamlit
  GeoJSON            Geometrías de los municipios de Santander

------------------------------------------------------------------------

# 3. Requisitos previos

Antes de ejecutar el proyecto se recomienda tener instalado:

-   Windows 10 u 11.
-   Python 3.10, 3.11 o 3.12.
-   `pip`.
-   CMD, PowerShell o una terminal equivalente.
-   Los archivos CSV y GeoJSON del proyecto.

### Comprobar Python

Abrir CMD y ejecutar:

``` cmd
python --version
```

Debe aparecer una versión de Python, por ejemplo:

``` text
Python 3.11.9
```

También se puede comprobar `pip`:

``` cmd
python -m pip --version
```

Si `python` no funciona, en algunas instalaciones de Windows puede
utilizarse:

``` cmd
py --version
```

------------------------------------------------------------------------

# 4. Estructura del proyecto

La estructura recomendada es:

``` text
Reto/
│
├── app.py
├── modelo.py
├── requirements.txt
├── README.md
│
├── resultados_predicciones.csv
├── top10_octubre_2025.csv
│
└── datos/
    ├── panel_entrenamiento_2015_2024.csv
    ├── panel_prueba_2025.csv
    └── santander_municipios.geojson
```

## Función de cada archivo

### `app.py`

Es la aplicación visual.

Se encarga de:

-   Crear el dashboard.
-   Cargar el modelo.
-   Mostrar indicadores.
-   Permitir modificar el umbral.
-   Mostrar tablas.
-   Crear gráficos.
-   Comparar Random Forest con la línea base.
-   Mostrar Recall y Precisión.
-   Mostrar el Top 10 de municipios.
-   Mostrar la importancia de las variables.
-   Mostrar la matriz de confusión.
-   Mostrar el mapa.

**No contiene un modelo diferente. Utiliza el modelo definido en
`modelo.py`.**

------------------------------------------------------------------------

### `modelo.py`

Contiene la lógica principal del modelo de Machine Learning.

Aquí se:

1.  Cargan los datos.
2.  Preparan las variables.
3.  Se crea el Random Forest.
4.  Se entrena.
5.  Se calculan probabilidades.
6.  Se generan predicciones.

La configuración del modelo es:

``` python
RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

Esto significa:

-   `n_estimators=300`: utiliza 300 árboles.
-   `class_weight="balanced"`: compensa el desbalance entre clases.
-   `random_state=42`: permite reproducir el entrenamiento.
-   `n_jobs=-1`: utiliza los núcleos disponibles del procesador.

------------------------------------------------------------------------

### `requirements.txt`

Contiene las librerías necesarias para ejecutar el proyecto.

Contenido recomendado:

``` txt
streamlit
pandas
numpy
scikit-learn
plotly
folium
streamlit-folium
```

------------------------------------------------------------------------

### `datos/panel_entrenamiento_2015_2024.csv`

Contiene los datos utilizados para entrenar el modelo.

El período utilizado es:

``` text
2015 → 2024
```

Este archivo **no debe mezclarse con el conjunto de prueba de 2025**.

------------------------------------------------------------------------

### `datos/panel_prueba_2025.csv`

Contiene los registros utilizados para evaluar el modelo.

Período:

``` text
2025
```

------------------------------------------------------------------------

### `datos/santander_municipios.geojson`

Contiene las geometrías de los municipios utilizadas para construir el
mapa.

------------------------------------------------------------------------

# 5. Variables utilizadas por el modelo

El Random Forest utiliza estas cuatro variables:

``` python
FEATURES = [
    "lluvia_mm_mes_anterior",
    "eventos_12m",
    "altitud_m",
    "mes",
]
```

La variable objetivo es:

``` python
TARGET = "hubo_mm"
```

## Descripción

  -----------------------------------------------------------------------
  Variable                            Descripción
  ----------------------------------- -----------------------------------
  `lluvia_mm_mes_anterior`            Precipitación registrada durante el
                                      mes anterior

  `eventos_12m`                       Cantidad de eventos registrados
                                      durante los últimos 12 meses

  `altitud_m`                         Altitud del municipio en metros

  `mes`                               Número del mes

  `hubo_mm`                           Variable objetivo: indica si hubo o
                                      no movimiento en masa
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 6. ¿Por qué no se utilizan algunas variables?

El conjunto de datos contiene variables como:

``` text
lluvia_mm
eventos_mes
```

Sin embargo, para realizar una predicción sin utilizar información del
mismo mes que se pretende predecir, el modelo trabaja con información
disponible previamente:

``` text
lluvia_mm_mes_anterior
eventos_12m
altitud_m
mes
```

Esto evita introducir información del mismo período que podría generar
fuga de información (*data leakage*).

------------------------------------------------------------------------

# 7. Instalación del proyecto

## Paso 1 --- Abrir CMD

Presionar:

``` text
Windows + R
```

Escribir:

``` text
cmd
```

y presionar Enter.

------------------------------------------------------------------------

## Paso 2 --- Entrar a la carpeta del proyecto

Si el proyecto está ubicado en:

``` text
D:\PC\Escritorio\Reto
```

ejecutar:

``` cmd
cd /d D:\PC\Escritorio\Reto
```

Comprobar que los archivos estén allí:

``` cmd
dir
```

Deberían aparecer, entre otros:

``` text
app.py
modelo.py
requirements.txt
datos
```

------------------------------------------------------------------------

# 8. Crear un entorno virtual

Se recomienda utilizar un entorno virtual para no mezclar las
dependencias de este proyecto con otras instalaciones de Python.

Crear el entorno:

``` cmd
python -m venv .venv
```

Esto crea:

``` text
Reto/
└── .venv/
```

------------------------------------------------------------------------

# 9. Activar el entorno virtual

En Windows CMD:

``` cmd
.venv\Scripts\activate
```

Si se activó correctamente, la terminal mostrará algo parecido a:

``` text
(.venv) D:\PC\Escritorio\Reto>
```

Mientras aparezca `(.venv)`, las instalaciones se realizarán dentro del
entorno del proyecto.

------------------------------------------------------------------------

# 10. Actualizar pip

Ejecutar:

``` cmd
python -m pip install --upgrade pip
```

No es obligatorio, pero ayuda a evitar problemas con instalaciones
antiguas de `pip`.

------------------------------------------------------------------------

# 11. Instalar las dependencias

Ejecutar:

``` cmd
python -m pip install -r requirements.txt
```

Este comando instala todas las librerías necesarias.

Entre ellas:

``` text
streamlit
pandas
numpy
scikit-learn
plotly
folium
streamlit-folium
```

La instalación puede tardar unos minutos.

------------------------------------------------------------------------

# 12. Verificar Streamlit

Ejecutar:

``` cmd
python -m streamlit --version
```

Si funciona, aparecerá la versión instalada de Streamlit.

Por ejemplo:

``` text
Streamlit, version 1.x.x
```

------------------------------------------------------------------------

# 13. Ejecutar la aplicación

No se debe iniciar el dashboard con:

``` cmd
python app.py
```

La forma correcta es:

``` cmd
python -m streamlit run app.py
```

Después de ejecutar el comando, Streamlit iniciará un servidor local.

Normalmente aparecerá una dirección similar a:

``` text
Local URL: http://localhost:8501
```

Abrir esa dirección en el navegador.

También se puede utilizar:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

# 14. Cómo detener la aplicación

En la ventana de CMD donde está ejecutándose Streamlit:

``` text
Ctrl + C
```

Esto detiene el servidor.

Para volver a ejecutarlo:

``` cmd
python -m streamlit run app.py
```

------------------------------------------------------------------------

# 15. Cómo volver a iniciar el proyecto otro día

No es necesario crear nuevamente el entorno virtual.

Abrir CMD y ejecutar:

``` cmd
cd /d D:\PC\Escritorio\Reto
```

Activar el entorno:

``` cmd
.venv\Scripts\activate
```

Y ejecutar:

``` cmd
python -m streamlit run app.py
```

------------------------------------------------------------------------

# 16. Funcionamiento general

El flujo de la aplicación es:

``` text
                    DATOS
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   ENTRENAMIENTO              PRUEBA
   2015–2024                    2025
          │                       │
          ▼                       │
    RANDOM FOREST                 │
          │                       │
          └───────────┬───────────┘
                      ▼
                PREDICCIONES
                      │
          ┌───────────┼────────────┐
          ▼           ▼            ▼
       Recall     Precisión     Probabilidad
                                  │
                                  ▼
                           Dashboard visual
```

------------------------------------------------------------------------

# 17. Separación temporal

Este es uno de los requisitos principales.

El proyecto utiliza:

``` text
ENTRENAMIENTO
2015 ───────────────── 2024
             │
             ▼
       RANDOM FOREST
             │
             ▼
EVALUACIÓN
2025
```

Los registros de 2025 no se utilizan para entrenar el modelo.

El código carga los archivos separados:

``` python
TRAIN_FILE = DATA_DIR / "panel_entrenamiento_2015_2024.csv"
TEST_FILE = DATA_DIR / "panel_prueba_2025.csv"
```

Y posteriormente:

``` python
modelo = entrenar_modelo(train)
resultado = predecir(modelo, test)
```

------------------------------------------------------------------------

# 18. Línea base

La línea base sirve para saber qué tan útil es el modelo frente a una
estrategia sencilla.

La estrategia utilizada es:

> Para un municipio y mes de 2025, repetir el resultado observado en ese
> mismo municipio y mes durante 2024.

Ejemplo:

``` text
Municipio: Vélez

Mayo 2024
hubo_mm = 1

        ↓

Mayo 2025
predicción línea base = 1
```

Otro ejemplo:

``` text
Municipio: Curití

Mayo 2024
hubo_mm = 0

        ↓

Mayo 2025
predicción línea base = 0
```

La aplicación calcula:

-   Accuracy de la línea base.
-   Recall de la línea base.
-   Precisión de la línea base.

Después los compara con el Random Forest.

------------------------------------------------------------------------

# 19. Recall

El **Recall** mide qué proporción de los casos positivos reales logró
identificar el modelo.

La fórmula es:

``` text
Recall = Verdaderos Positivos /
         (Verdaderos Positivos + Falsos Negativos)
```

En términos sencillos:

> De todos los casos que realmente ocurrieron, ¿cuántos consiguió
> detectar el modelo?

El dashboard muestra el Recall en la parte superior y en la pestaña
**Evaluación**.

------------------------------------------------------------------------

# 20. Precisión

La **Precisión** mide qué proporción de las predicciones positivas
realizadas por el modelo fueron realmente positivas.

La fórmula es:

``` text
Precisión = Verdaderos Positivos /
            (Verdaderos Positivos + Falsos Positivos)
```

En términos sencillos:

> De todos los casos que el modelo marcó como positivos, ¿cuántos
> realmente lo fueron?

------------------------------------------------------------------------

# 21. Métrica priorizada

En la aplicación se prioriza:

``` text
RECALL
```

La razón es que el objetivo es detectar la mayor cantidad posible de
casos reales de movimientos en masa.

Un falso negativo significa:

``` text
Ocurrió un movimiento
        ↓
El modelo dijo "No"
```

Por lo tanto, el dashboard explica que el Recall es especialmente
importante para este problema.

Esta es una decisión metodológica del proyecto y no significa que la
precisión deje de ser importante. Ambas métricas se reportan.

------------------------------------------------------------------------

# 22. Umbral de predicción

En la barra lateral aparece:

``` text
Umbral de predicción
```

Por defecto:

``` text
0.50
```

El Random Forest genera una probabilidad entre 0 y 1.

Por ejemplo:

``` text
Municipio A → 0.82
Municipio B → 0.61
Municipio C → 0.34
```

Con un umbral de 0.50:

``` text
0.82 >= 0.50 → Positivo
0.61 >= 0.50 → Positivo
0.34 <  0.50 → Negativo
```

Por eso el umbral modifica la cantidad de predicciones positivas.

### Si se baja el umbral

Por ejemplo:

``` text
0.30
```

más registros pueden clasificarse como positivos.

### Si se sube el umbral

Por ejemplo:

``` text
0.70
```

se necesitará una probabilidad mayor para clasificar un registro como
positivo.

El dashboard permite modificar el umbral sin volver a entrenar el
modelo.

------------------------------------------------------------------------

# 23. Dashboard

La aplicación está dividida en seis secciones.

## Dashboard

Presenta:

-   Número de municipios.
-   Registros de entrenamiento.
-   Recall.
-   Precisión.
-   Número de predicciones positivas.
-   Distribución histórica.
-   Top 10 del mes seleccionado.
-   Precipitación promedio mensual.

------------------------------------------------------------------------

# 24. Sección Municipios

Permite seleccionar un mes de 2025.

Muestra:

-   Top 10 municipios.
-   Probabilidad estimada.
-   Predicción positiva o negativa.
-   Lluvia del mes anterior.
-   Eventos de los últimos 12 meses.
-   Altitud.

También permite descargar los resultados en CSV.

------------------------------------------------------------------------

# 25. Top 10 de octubre de 2025

El requisito del reto solicita los municipios con mayor probabilidad
para **octubre de 2025**.

Por eso el selector inicia en:

``` text
Octubre
```

La aplicación ordena los municipios de mayor a menor probabilidad:

``` text
1. Municipio A — XX%
2. Municipio B — XX%
3. Municipio C — XX%
...
10. Municipio J — XX%
```

El usuario puede cambiar el mes para realizar análisis adicionales, pero
octubre queda como selección inicial para cumplir el requisito.

------------------------------------------------------------------------

# 26. Sección Evaluación

Esta sección reúne los requisitos metodológicos principales.

Presenta:

### Requisito 1

``` text
Entrenamiento: 2015–2024
Evaluación: 2025
```

### Requisito 2

Comparación:

``` text
Random Forest
vs
Línea base 2024
```

### Requisito 3

Métricas:

``` text
Recall
Precisión
Accuracy
```

También muestra:

-   Gráfico comparativo.
-   Matriz de confusión.
-   Explicación de la línea base.
-   Explicación de la métrica priorizada.

------------------------------------------------------------------------

# 27. Matriz de confusión

La matriz de confusión muestra cuatro resultados:

``` text
                    PREDICCIÓN
                  NO          SÍ

REAL NO           TN          FP

REAL SÍ           FN          TP
```

Donde:

-   **TN:** verdadero negativo.
-   **FP:** falso positivo.
-   **FN:** falso negativo.
-   **TP:** verdadero positivo.

Esta matriz permite entender de dónde salen las métricas Recall y
Precisión.

------------------------------------------------------------------------

# 28. Sección Análisis

Incluye gráficos para explorar la relación entre las variables y las
probabilidades generadas por el modelo.

Se muestran:

### Distribución de probabilidades

Permite observar cómo se distribuyen las probabilidades generadas.

### Lluvia vs probabilidad

Compara:

``` text
lluvia del mes anterior
```

contra:

``` text
probabilidad estimada
```

### Altitud vs probabilidad

Compara:

``` text
altitud
```

contra:

``` text
probabilidad
```

### Eventos 12 meses vs probabilidad

Compara:

``` text
eventos de los últimos 12 meses
```

contra:

``` text
probabilidad
```

Estos gráficos son herramientas de exploración y no representan por sí
solos una relación causal.

------------------------------------------------------------------------

# 29. Sección Modelo

Presenta la configuración del Random Forest:

``` text
Random Forest Classifier

n_estimators = 300
class_weight = balanced
random_state = 42
n_jobs = -1
```

También muestra:

-   Variables utilizadas.
-   Descripción de las variables.
-   Importancia de cada variable.
-   Matriz de confusión.

------------------------------------------------------------------------

# 30. Importancia de variables

Random Forest permite obtener una estimación de la importancia de las
variables mediante:

``` python
modelo.feature_importances_
```

El dashboard representa estos valores mediante un gráfico.

Las variables utilizadas son:

``` text
lluvia_mm_mes_anterior
eventos_12m
altitud_m
mes
```

La importancia indica cuánto contribuye cada variable al proceso de
decisión de los árboles del modelo.

No debe interpretarse automáticamente como causalidad.

------------------------------------------------------------------------

# 31. Sección Mapa

La aplicación puede mostrar las probabilidades sobre un mapa de
Santander.

El mapa necesita:

``` text
datos/santander_municipios.geojson
```

El programa intenta identificar automáticamente el campo que contiene el
nombre del municipio.

Se contemplan nombres de campo comunes como:

``` text
municipio
MUNICIPIO
Municipio
nombre
NOMBRE
name
NAME
```

El mapa utiliza rangos visuales:

``` text
0%–39%   → baja
40%–59%  → moderada
60%–79%  → alta
80%–100% → muy alta
```

Estos rangos son categorías visuales del dashboard y no representan
categorías oficiales de amenaza.

------------------------------------------------------------------------

# 32. Archivos de salida

Cuando se ejecuta `modelo.py`, el proyecto puede generar:

``` text
resultados_predicciones.csv
```

y:

``` text
top10_octubre_2025.csv
```

El dashboard también permite descargar el Top 10 correspondiente al mes
seleccionado.

------------------------------------------------------------------------

# 33. Ejecutar solamente el modelo

Si se quiere ejecutar el modelo sin abrir Streamlit:

``` cmd
python modelo.py
```

Esto permite comprobar el funcionamiento del modelo directamente desde
la consola.

La aplicación visual, en cambio, se ejecuta con:

``` cmd
python -m streamlit run app.py
```

------------------------------------------------------------------------

# 34. Diferencia entre `modelo.py` y `app.py`

Es importante no confundirlos.

``` text
modelo.py
    │
    ├── Carga datos
    ├── Prepara datos
    ├── Entrena Random Forest
    └── Genera predicciones
             │
             ▼
app.py
    │
    ├── Usa el modelo
    ├── Calcula métricas
    ├── Calcula línea base
    ├── Crea tablas
    ├── Crea gráficos
    └── Muestra el dashboard
```

La interfaz no reemplaza el modelo.

La interfaz utiliza el modelo.

------------------------------------------------------------------------

# 35. No modificar los nombres de los archivos de datos

Los nombres esperados son exactamente:

``` text
panel_entrenamiento_2015_2024.csv
panel_prueba_2025.csv
santander_municipios.geojson
```

Deben estar dentro de:

``` text
datos/
```

Si se cambia el nombre, el programa puede mostrar un error de archivo no
encontrado.

------------------------------------------------------------------------

# 36. Errores comunes

## Error: `No module named streamlit`

Ejecutar:

``` cmd
python -m pip install streamlit
```

O instalar todo:

``` cmd
python -m pip install -r requirements.txt
```

------------------------------------------------------------------------

## Error: `No module named plotly`

Ejecutar:

``` cmd
python -m pip install plotly
```

------------------------------------------------------------------------

## Error: `No module named sklearn`

Ejecutar:

``` cmd
python -m pip install scikit-learn
```

------------------------------------------------------------------------

## Error: `No module named pandas`

Ejecutar:

``` cmd
python -m pip install pandas
```

------------------------------------------------------------------------

## Error: `No module named folium`

Ejecutar:

``` cmd
python -m pip install folium
```

------------------------------------------------------------------------

## Error: `No module named streamlit_folium`

Ejecutar:

``` cmd
python -m pip install streamlit-folium
```

------------------------------------------------------------------------

## Error: `StreamlitDuplicateElementId`

Los gráficos Plotly deben tener claves únicas.

El `app.py` corregido utiliza:

``` python
st.plotly_chart(
    fig,
    use_container_width=True,
    key="nombre_unico",
)
```

Cada gráfico tiene una clave diferente para evitar conflictos de
componentes.

------------------------------------------------------------------------

## Error: no encuentra `modelo`

Comprobar que:

``` text
app.py
modelo.py
```

estén en la misma carpeta.

Correcto:

``` text
Reto/
├── app.py
└── modelo.py
```

Incorrecto:

``` text
Reto/
├── app.py
└── otra_carpeta/
    └── modelo.py
```

------------------------------------------------------------------------

## Error: no encuentra los CSV

Comprobar:

``` text
Reto/
└── datos/
    ├── panel_entrenamiento_2015_2024.csv
    └── panel_prueba_2025.csv
```

------------------------------------------------------------------------

# 37. Comando completo desde cero

Si el proyecto está en:

``` text
D:\PC\Escritorio\Reto
```

los comandos son:

``` cmd
cd /d D:\PC\Escritorio\Reto
```

Crear entorno:

``` cmd
python -m venv .venv
```

Activar:

``` cmd
.venv\Scripts\activate
```

Actualizar pip:

``` cmd
python -m pip install --upgrade pip
```

Instalar dependencias:

``` cmd
python -m pip install -r requirements.txt
```

Ejecutar:

``` cmd
python -m streamlit run app.py
```

Abrir:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

# 38. Flujo recomendado para una demostración

Para presentar el proyecto:

### Paso 1

Ejecutar:

``` cmd
python -m streamlit run app.py
```

### Paso 2

Mostrar la pantalla principal.

Explicar:

-   87 municipios.
-   Registros de entrenamiento.
-   Recall.
-   Precisión.
-   Predicciones positivas.

### Paso 3

Entrar en:

``` text
Municipios
```

Seleccionar:

``` text
Octubre
```

Mostrar el Top 10 de municipios.

### Paso 4

Entrar en:

``` text
Evaluación
```

Mostrar:

``` text
2015–2024 → entrenamiento
2025       → evaluación
```

### Paso 5

Mostrar:

``` text
Random Forest
vs
Línea base 2024
```

### Paso 6

Explicar:

``` text
Recall
Precisión
```

y por qué el proyecto prioriza Recall.

### Paso 7

Mostrar:

``` text
Matriz de confusión
```

### Paso 8

Mostrar:

``` text
Modelo
```

para explicar:

-   Random Forest.
-   300 árboles.
-   Variables.
-   Importancia.

### Paso 9

Finalmente mostrar:

``` text
Mapa
```

para visualizar espacialmente las probabilidades.

------------------------------------------------------------------------

# 39. Resumen de comandos

### Instalación inicial

``` cmd
cd /d D:\PC\Escritorio\Reto
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Ejecución

``` cmd
python -m streamlit run app.py
```

### Ejecutar solamente el modelo

``` cmd
python modelo.py
```

### Comprobar Streamlit

``` cmd
python -m streamlit --version
```

### Detener Streamlit

``` text
Ctrl + C
```

------------------------------------------------------------------------

# 40. Resultado final

Al ejecutar correctamente la aplicación se obtiene un dashboard web
interactivo que integra:

``` text
                 UTSMART IA
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
   DATOS          MODELO       EVALUACIÓN
       │             │             │
       │       Random Forest    Recall
       │       300 árboles      Precisión
       │             │          Línea base
       │             │          2024
       │             │             │
       └─────────────┼─────────────┘
                     ▼
               DASHBOARD
                     │
       ┌─────────────┼──────────────┐
       ▼             ▼              ▼
     Tablas        Gráficos        Mapa
       │             │              │
       └─────────────┼──────────────┘
                     ▼
             Top 10 octubre 2025
```

El proyecto conserva el modelo Random Forest original y utiliza
Streamlit como capa de presentación y análisis visual.
