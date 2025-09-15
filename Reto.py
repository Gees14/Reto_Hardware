# -*- coding: utf-8 -*-
"""
Práctica: Digitalización y filtrado de señales con datos NOAA (vía Meteostat)
- Señal: temperatura diaria promedio (tavg) en °C
- Dataset: NOAA/GHCN a través de la librería Meteostat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from meteostat import Point, Daily

# ================
# 1) DESCARGA DATA
# ================
# Ubicación (puedes cambiarla por tu ciudad)
ciudad = Point(19.4326, -99.1332)  # Ciudad de México
inicio = datetime(2023, 1, 1)
fin    = datetime(2024, 12, 31)

# Descarga diaria
df = Daily(ciudad, inicio, fin).fetch()

# Tomamos la temperatura promedio diaria (tavg). Puede tener NaN; los eliminamos.
serie = df['tavg'].dropna().astype(float)

# Señal cruda (data_raw) como vector numpy
data_raw = serie.values
fs_raw = 1.0  # 1 muestra/día (frecuencia original estimada para tavg)

print(f"Observaciones crudas: {len(data_raw)} días, "
      f"desde {serie.index.min().date()} hasta {serie.index.max().date()}")

# =====================================================
# 2) DIGITALIZACIÓN: SAMPLING + QUANTIZATION (data_digitized)
# =====================================================
# --- Sampling: reducir frecuencia (tomar 1 de cada N) ---
N = 3  # por ejemplo, de 1/día a 1 cada 3 días
data_sampled = data_raw[::N]
fs_sampled = fs_raw / N

# --- Quantization: reducir niveles de amplitud ---
levels = 128  # número de niveles (puedes probar 4, 8, 16, ...)
xmin, xmax = np.min(data_raw), np.max(data_raw)
q_step = (xmax - xmin) / levels
# cuantizamos sobre la señal muestreada para obtener "data_digitized"
data_digitized = np.round((data_sampled - xmin) / q_step) * q_step + xmin

# =======================================
# 3) FILTROS: Kalman 1D y Observador tipo Luenberger (EMA)
# =======================================
def kalman_filter_1d(z, R=0.25, Q=0.01):
    """
    Filtro de Kalman 1D para una señal escalar.
    z : mediciones (np.array)
    R : varianza del ruido de medición
    Q : varianza del proceso (suavizado)
    Devuelve: x_hat (estimado filtrado)
    """
    x_hat = np.zeros_like(z, dtype=float)
    P = 1.0  # varianza inicial del error
    x_hat[0] = z[0]  # estado inicial

    for k in range(1, len(z)):
        # Predicción
        x_pred = x_hat[k-1]
        P_pred = P + Q

        # Actualización
        K = P_pred / (P_pred + R)  # Ganancia de Kalman
        x_hat[k] = x_pred + K * (z[k] - x_pred)
        P = (1 - K) * P_pred

    return x_hat

# Parámetros de ruido: ajusta R (ruido de medición) y Q (suavizado del modelo)
kalman_R = 0.3**2   # varianza ~ (0.3°C)^2
kalman_Q = 0.05**2  # varianza del proceso
data_kalman = kalman_filter_1d(data_raw, R=kalman_R, Q=kalman_Q)

def luenberger_like_ema(y, alpha=0.2):
    """
    Observador tipo Luenberger de 1er orden (equivalente a un EMA).
    x_hat_k = x_hat_{k-1} + alpha*(y_k - x_hat_{k-1})
    """
    xh = np.zeros_like(y, dtype=float)
    xh[0] = y[0]
    for k in range(1, len(y)):
        xh[k] = xh[k-1] + alpha * (y[k] - xh[k-1])
    return xh

alpha_obs = 0.15
data_luenberger = luenberger_like_ema(data_raw, alpha=alpha_obs)

# ==================
# 4) VISUALIZACIONES
# ==================
plt.figure(figsize=(12, 6))
plt.plot(data_raw, label='data_raw (tavg diaria)', linewidth=1.2)
plt.plot(np.arange(0, len(data_raw), N), data_sampled, 'o-', label=f'sampled (cada {N} días)', linewidth=1.0)
plt.plot(np.arange(0, len(data_raw), N), data_digitized, 's--', label=f'quantized (niveles={levels})', linewidth=1.0)
plt.title('Digitalización: Sampling & Quantization (Temperatura diaria)')
plt.xlabel('muestra (día)')
plt.ylabel('°C')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(data_raw, label='data_raw', linewidth=1.0)
plt.plot(data_kalman, label='Kalman 1D', linewidth=1.6)
plt.plot(data_luenberger, label=f'Luenberger/EMA (α={alpha_obs})', linewidth=1.6)
plt.title('Filtrado: Kalman vs Luenberger (EMA)')
plt.xlabel('muestra (día)')
plt.ylabel('°C')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================
# 5) RESULTADOS PARA EL REPORTE
# ============================
resultados = {
    "fs_raw (Hz)": fs_raw,
    "fs_sampled (Hz)": fs_sampled,
    "N_sampling": N,
    "quant_levels": levels,
    "xmin": float(xmin),
    "xmax": float(xmax),
    "q_step": float(q_step),
    "Kalman_R": kalman_R,
    "Kalman_Q": kalman_Q,
    "Luenberger_alpha": alpha_obs,
    "n_raw": len(data_raw),
    "n_sampled": len(data_sampled)
}

print("\nResumen de parámetros:")
for k, v in resultados.items():
    print(f" - {k}: {v}")

# Si quieres exportar las series para tu informe:
out = pd.DataFrame({
    "raw_tavg_C": data_raw,
    "kalman_tavg_C": data_kalman,
    "luenberger_tavg_C": data_luenberger
})
# Para alinear con el muestreo/cuantización (más esporádicos):
sample_idx = np.arange(0, len(data_raw), N)
out_sample = pd.DataFrame({
    "idx": sample_idx,
    "sampled_C": data_sampled,
    "digitized_C": data_digitized
})

# Guardado opcional
# out.to_csv("noaa_temp_filtrado.csv", index=False)
# out_sample.to_csv("noaa_temp_digitalizado.csv", index=False)
