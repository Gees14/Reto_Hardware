# pipeline_digitalizacion_airquality.py
# -------------------------------------
# Digitalización (muestreo + cuantización) y filtrado (Luenberger + Kalman)
# sobre el dataset AirQualityUCI (Kaggle/UCI).
#
# Requiere: numpy, pandas, matplotlib
# Ejecuta:  python pipeline_digitalizacion_airquality.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =========================
# PARÁMETROS PRINCIPALES
# =========================
CSV_PATH   = r"Modulo_Hardware\AirQualityUCI.csv"  # <-- ruta a tu CSV
SENSOR_COL = "T"                             # "T" (Temperatura) o "RH" (Humedad)
DATE_COL   = None
TIME_COL   = None

# Rejilla uniforme de referencia (el dataset es cada minuto)
RAW_RESAMPLE_RULE = "1min"

# Digitalización simulada
DOWNSAMPLE_FACTOR = 10   # toma 1 de cada N (simula menor frecuencia)
QUANT_BITS        = 6    # bits de cuantización (menos que el raw)

# Observador de Luenberger (modelo AR(1) estimado)
LAMBDA_OBS = 0.75        # polo deseado del error del observador (0.6-0.9 típico)

# Kalman escalar (mismo modelo AR(1))
Q_PROCESS  = 1e-3        # varianza del proceso
R_MEASURE  = 1e-2        # varianza de medición

# Ventana de visualización (None para todo)
PLOT_N_SAMPLES = 2000

# Guardar resultados a CSV (opcional)
SAVE_OUTPUTS = False
OUTPUT_PREFIX = "airquality_results"  # se usarán sufijos _raw/_digitized/_luen/_kal


# =========================
# CARGA Y LIMPIEZA DEL CSV
# =========================
import re
import numpy as np
import pandas as pd

def _norm(s: str) -> str:
    """Normaliza un nombre de columna: quita BOM/espacios/símbolos y pasa a lower."""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\ufeff", "")          # BOM
    s = s.strip()
    s = re.sub(r"[^a-zA-Z0-9]+", "", s)  # quita todo salvo a-zA-Z0-9
    return s.lower()

def _find_col_by_alias(df: pd.DataFrame, aliases):
    """Busca en df.columns el primer nombre que, normalizado, coincida con alguno de aliases."""
    norm_map = {c: _norm(c) for c in df.columns}
    aliases  = [_norm(a) for a in aliases]
    for real, n in norm_map.items():
        if n in aliases:
            return real
    return None

def load_timeseries_csv(path, value_col, date_col=None, time_col=None):
    # 1) Leer CSV con formato UCI/Kaggle
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="latin1",
        engine="python",
        na_values=[-200, "-200", "", "NA", "NaN"]
    )

    # 2) Quitar columnas “Unnamed”
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]

    # 3) Si no nos dan Date/Time, autodetectar (robusto)
    if date_col is None or date_col not in df.columns:
        date_col = _find_col_by_alias(df, ["date", "fecha"])
    if time_col is None or time_col not in df.columns:
        time_col = _find_col_by_alias(df, ["time", "hora"])

    # Si no hay par date+time, intentar combinadas
    combined_col = None
    if date_col is None or time_col is None:
        combined_col = _find_col_by_alias(df, ["datetime", "datetimestamp", "datetimeunix", "fechahora", "datetime"])
    # 4) Construir índice de tiempo
    if date_col and time_col:
        ts = pd.to_datetime(
            df[date_col].astype(str).str.replace("\ufeff", "", regex=False).str.strip() + " " +
            df[time_col].astype(str).str.replace("\ufeff", "", regex=False).str.strip(),
            errors="coerce",
            dayfirst=True,
        )
    elif combined_col:
        ts = pd.to_datetime(
            df[combined_col].astype(str).str.replace("\ufeff", "", regex=False).str.strip(),
            errors="coerce",
            dayfirst=True,
        )
    else:
        # último recurso: usar la primera columna como timestamp
        first_col = df.columns[0]
        ts = pd.to_datetime(df[first_col], errors="coerce", dayfirst=True)

    # 5) Validar/seleccionar columna del sensor (T o RH)
    if value_col not in df.columns:
        # intenta localizar por alias
        value_col = _find_col_by_alias(df, [value_col, "t", "rh"])
        if value_col is None:
            raise KeyError("No se encontró la columna del sensor (p.ej. 'T' o 'RH').")

    # 6) Construir serie
    s = pd.Series(pd.to_numeric(df[value_col], errors="coerce").values, index=ts)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")].dropna()
    return s


def resample_uniform(series, rule):
    """Resamplea a una rejilla uniforme e interpola huecos."""
    s = series.resample(rule).mean()
    s = s.interpolate(method="time").ffill().bfill()
    return s


# =========================
# DIGITALIZACIÓN
# =========================
def downsample(series, factor):
    """Toma 1 de cada N muestras (simula menor frecuencia de muestreo)."""
    return series.iloc[::factor].copy()

def uniform_quantize(x, bits):
    """
    Cuantización uniforme en el rango [min, max] de la señal.
    Devuelve: (señal cuantizada float, índices de nivel int).
    """
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if xmax == xmin:
        return x.copy(), np.zeros_like(x, dtype=int)
    L = 2**bits
    # Normaliza a [0, 1]
    xn = (x - xmin) / (xmax - xmin)
    # Índices 0..L-1
    idx = np.clip(np.round(xn * (L - 1)), 0, L - 1).astype(int)
    # Reconstrucción al centro del nivel (mid-rise)
    xr = idx / (L - 1) * (xmax - xmin) + xmin
    return xr, idx


# =========================
# ESTIMACIÓN DEL MODELO AR(1)
# =========================
def estimate_a_ar1(x):
    """Estima 'a' de x_{k+1} ≈ a x_k por mínimos cuadrados."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 1.0
    xk, xk1 = x[:-1], x[1:]
    denom = np.dot(xk, xk)
    if denom == 0:
        return 1.0
    a = float(np.dot(xk1, xk) / denom)
    # Limita a un rango estable y razonable
    return max(min(a, 1.2), 0.6)


# =========================
# OBSERVADOR DE LUENBERGER
# =========================
def luenberger_observer(y, a, lambda_obs, x0=None):
    """
    Sistema:
        x_{k+1} = a x_k
        y_k     = x_k
    Observador:
        x̂_{k+1} = a x̂_k + L (y_k - x̂_k),  L = a - lambda_obs
    """
    L = a - lambda_obs
    y = np.asarray(y, dtype=float)
    xhat = np.zeros_like(y)
    xhat[0] = y[0] if x0 is None else x0
    for k in range(len(y)-1):
        xhat[k+1] = a * xhat[k] + L * (y[k] - xhat[k])
    return xhat


# =========================
# FILTRO DE KALMAN (ESCALAR)
# =========================
@dataclass
class KalmanScalar:
    a: float
    q: float
    r: float
    x: float
    P: float

    def step(self, y):
        # Predicción
        x_pred = self.a * self.x
        P_pred = self.a * self.P * self.a + self.q
        # Actualización (H=1)
        K = P_pred / (P_pred + self.r)
        self.x = x_pred + K * (y - x_pred)
        self.P = (1 - K) * P_pred
        return self.x

def kalman_filter(y, a, q, r, x0=None, P0=1.0):
    y = np.asarray(y, dtype=float)
    x0 = y[0] if x0 is None else x0
    kf = KalmanScalar(a=a, q=q, r=r, x=float(x0), P=float(P0))
    xhat = np.zeros_like(y)
    for i, yi in enumerate(y):
        xhat[i] = kf.step(yi)
    return xhat


# =========================
# MÉTRICAS
# =========================
def mse(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.mean((a - b)**2)

def mae(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.mean(np.abs(a - b))


# =========================
# PROGRAMA PRINCIPAL
# =========================
def main():
    # 1) Carga del raw y rejilla uniforme
    s_raw = load_timeseries_csv(CSV_PATH, SENSOR_COL, DATE_COL, TIME_COL)
    s_raw = resample_uniform(s_raw, RAW_RESAMPLE_RULE)
    s_raw.name = "data_raw"

    # 2) Digitalización: muestreo + cuantización
    s_sampled = downsample(s_raw, DOWNSAMPLE_FACTOR)
    s_quant, _ = uniform_quantize(s_sampled.values, QUANT_BITS)
    s_digitized = pd.Series(s_quant, index=s_sampled.index, name="data_digitized")

    # 3) Estimar 'a' (AR(1)) desde el raw
    a_est = estimate_a_ar1(s_raw.values)
    print(f"[INFO] a estimado (AR(1)) = {a_est:.4f}")

    # 4) Luenberger sobre la señal digitalizada
    xhat_luen = luenberger_observer(s_digitized.values, a=a_est, lambda_obs=LAMBDA_OBS)
    s_luen = pd.Series(xhat_luen, index=s_digitized.index, name="xhat_luenberger")

    # 5) Kalman sobre la señal digitalizada
    xhat_kal = kalman_filter(s_digitized.values, a=a_est, q=Q_PROCESS, r=R_MEASURE)
    s_kal = pd.Series(xhat_kal, index=s_digitized.index, name="xhat_kalman")

    # 6) Métricas contra el raw alineado a los timestamps digitalizados
    s_raw_aligned = s_raw.reindex(s_digitized.index).interpolate().ffill().bfill()
    m_luen_mse = mse(s_luen.values, s_raw_aligned.values)
    m_kal_mse  = mse(s_kal.values,  s_raw_aligned.values)
    m_luen_mae = mae(s_luen.values, s_raw_aligned.values)
    m_kal_mae  = mae(s_kal.values,  s_raw_aligned.values)

    print("\n=== Métricas vs data_raw (alineado) ===")
    print(f"Luenberger: MSE={m_luen_mse:.6f}  MAE={m_luen_mae:.6f}")
    print(f"Kalman    : MSE={m_kal_mse:.6f}  MAE={m_kal_mae:.6f}")

    # 7) Guardado opcional
    if SAVE_OUTPUTS:
        s_raw.to_csv(f"{OUTPUT_PREFIX}_raw.csv", header=True)
        s_digitized.to_csv(f"{OUTPUT_PREFIX}_digitized.csv", header=True)
        s_luen.to_csv(f"{OUTPUT_PREFIX}_luen.csv", header=True)
        s_kal.to_csv(f"{OUTPUT_PREFIX}_kal.csv", header=True)
        print(f"[INFO] Resultados guardados con prefijo: {OUTPUT_PREFIX}_*.csv")

    # 8) Gráficas
    if PLOT_N_SAMPLES is not None:
        end = min(PLOT_N_SAMPLES, len(s_digitized))
        sidx = slice(0, end)
    else:
        sidx = slice(0, len(s_digitized))

    plt.figure()
    s_raw_aligned.iloc[sidx].plot(label="raw (alineado)")
    s_digitized.iloc[sidx].plot(label="digitized (downsample+quantize)")
    s_luen.iloc[sidx].plot(label="Luenberger")
    s_kal.iloc[sidx].plot(label="Kalman")
    plt.title(f"Digitalización y filtros sobre {SENSOR_COL}")
    plt.xlabel("tiempo")
    plt.ylabel(SENSOR_COL)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
