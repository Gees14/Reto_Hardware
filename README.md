# Digitalización y Filtrado de Señales en Series de Tiempo

Este repositorio contiene la práctica de **procesamiento de señales** aplicada a datos reales de temperatura.  
Se abordan las etapas de **digitalización (sampling y quantization)** y **filtrado** mediante dos métodos:  
- **Observador de Luenberger (EMA)**
- **Filtro de Kalman (1D)**

Los resultados se presentan tanto en gráficas como en un **reporte en PDF** que explica la implementación, parámetros empleados y conclusiones.

---

## 📂 Contenido del repositorio

- `digitalizacion_filtros.py` → Código principal en Python para:
  - Descargar el dataset NOAA (temperatura diaria en Ciudad de México).
  - Aplicar *sampling* y *quantization*.
  - Filtrar la señal con Kalman y Luenberger.
  - Generar gráficas comparativas.

- `reporte.pdf` → Documento con la explicación teórica y técnica de la práctica, resultados y conclusiones.

- `README.md` → Este archivo.

---

## 📊 Dataset

- Fuente: [NOAA / Meteostat](https://meteostat.net/)  
- Variable: Temperatura diaria promedio (°C).  
- Periodo: Enero 2023 – Diciembre 2024.  
- Ubicación: Ciudad de México.  

---

## ⚙️ Requisitos

Asegúrate de tener **Python 3.11+** y las siguientes librerías instaladas:

```bash
pip install numpy pandas matplotlib meteostat
