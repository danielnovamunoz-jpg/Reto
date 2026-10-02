
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datos"

TRAIN_FILE = DATA_DIR / "panel_entrenamiento_2015_2024.csv"
TEST_FILE = DATA_DIR / "panel_prueba_2025.csv"

FEATURES = [
    "lluvia_mm_mes_anterior",
    "eventos_12m",
    "altitud_m",
    "mes",
]

TARGET = "hubo_mm"


def cargar_datos():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    return train, test


def preparar_datos(train, test):
    train = train.copy()
    test = test.copy()

    columnas = FEATURES + [TARGET]

    for columna in columnas:
        train[columna] = pd.to_numeric(train[columna], errors="coerce")
        test[columna] = pd.to_numeric(test[columna], errors="coerce")

    train = train.dropna(subset=columnas)
    test = test.dropna(subset=columnas)

    return train, test


def crear_modelo():
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def entrenar_modelo(train):
    modelo = crear_modelo()

    X_train = train[FEATURES]
    y_train = train[TARGET].astype(int)

    modelo.fit(X_train, y_train)

    return modelo


def predecir(modelo, test, threshold=0.50):
    resultado = test.copy()

    resultado["probabilidad"] = modelo.predict_proba(
        resultado[FEATURES]
    )[:, 1]

    resultado["prediccion"] = (
        resultado["probabilidad"] >= threshold
    ).astype(int)

    return resultado


def evaluar(resultado):
    y_true = resultado[TARGET].astype(int)
    y_pred = resultado["prediccion"].astype(int)

    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)

    return recall, precision


def obtener_top10(resultado, mes=10):
    datos = resultado[resultado["mes"] == mes].copy()

    return datos.sort_values(
        "probabilidad",
        ascending=False
    ).head(10)


if __name__ == "__main__":
    train, test = cargar_datos()
    train, test = preparar_datos(train, test)

    print("Datos de entrenamiento:", train.shape)
    print("Datos de prueba:", test.shape)

    print("\nColumnas:")
    print(train.columns.tolist())

    print("\nEntrenando modelo...")

    modelo = entrenar_modelo(train)

    print("Modelo entrenado correctamente.")

    resultado = predecir(modelo, test)

    recall, precision = evaluar(resultado)

    print("\n==============================")
    print("RESULTADOS RANDOM FOREST")
    print("==============================")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    top10 = obtener_top10(resultado, mes=10)

    print("\n==========================================")
    print("TOP 10 MUNICIPIOS - OCTUBRE 2025")
    print("==========================================")

    for posicion, (_, fila) in enumerate(
        top10.iterrows(), start=1
    ):
        print(
            f"{posicion}. {fila['municipio']} - "
            f"{fila['probabilidad'] * 100:.2f}%"
        )

    resultado.to_csv(
        BASE_DIR / "resultados_predicciones.csv",
        index=False
    )

    top10.to_csv(
        BASE_DIR / "top10_octubre_2025.csv",
        index=False
    )

    print("\nArchivos generados:")
    print("resultados_predicciones.csv")
    print("top10_octubre_2025.csv")
