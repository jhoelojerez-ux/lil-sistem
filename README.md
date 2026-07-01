# 📊 Analizador Financiero VAR-GARCH (EUR/USD y VIX)

Este proyecto es una aplicación web interactiva desarrollada en **Streamlit** que implementa un pipeline econométrico avanzado para el análisis y pronóstico de series de tiempo financieras. Combina un modelo de **Vectores Autorregresivos (VAR)** para capturar las interdependencias dinámicas entre el par EUR/USD y el índice VIX, junto con un modelo **GARCH(1,1)** para modelar la volatilidad condicional y generar bandas de probabilidad sobre el precio.

---

## 🛠️ Arquitectura y Pasos del Sistema

La aplicación está estructurada en 5 fases secuenciales que representan el flujo de trabajo del análisis cuantitativo:

### 1. Ingesta y Preprocesamiento de Datos
* **Descarga Automática:** Conexión en tiempo real con la API de Yahoo Finance (`yfinance`) para extraer el historial diario del EUR/USD (`EURUSD=X`) y el VIX (`^VIX`).
* **Transformación Estacionaria:** Cálculo de retornos logarítmicos individuales para estabilizar la varianza:
  $$R_t = \ln(P_t / P_{t-1})$$
* **Sincronización:** Limpieza de datos faltantes ($NaN$) y alineación temporal estricta de ambas series en un único DataFrame de retornos corregidos.

### 2. Modelado de Interdependencia (VAR)
* **Ajuste del Modelo:** Implementación de un modelo VAR($p$) donde el número de rezagos ($lags$) es parametrizable por el usuario.
* **Análisis de Impacto:** Generación de **Funciones de Respuesta al Impulso (IRF)** para evaluar visualmente cómo un choque exógeno en la volatilidad global (VIX) afecta dinámicamente a los retornos del EUR/USD a lo largo del tiempo.

### 3. Diagnóstico e Identificación ARCH/GARCH
* **Filtro de Media:** Ajuste inicial mediante un modelo de media constante para extraer los residuos puros de la serie de retornos.
* **Prueba de Heterocedasticidad:** Aplicación del test **ARCH-LM** (`het_arch`) sobre los residuos para validar estadísticamente la presencia de efectos ARCH (volatilidad agrupada o *volatility clustering*).
* **Modelado de Volatilidad:** Si se detectan efectos ARCH, se ajusta un modelo **GARCH(1,1)** para estimar la varianza condicional de la próxima sesión.

### 4. Pronóstico Estadístico en Niveles de Precio
* **Extrapolación:** Proyección de los retornos futuros combinando la media estimada y la varianza condicional pronosticada por el GARCH.
* **Conversión a Precios:** Reconstrucción de los retornos proyectados hacia niveles de precios reales aplicando la capitalización continua sobre el último cierre disponible:
  $$P_{t+h} = P_t \cdot \exp\left(\sum R_{t+h}\right)$$
* **Bandas de Confianza:** Construcción de intervalos de predicción dinámica al 95% y 60% utilizando los percentiles de la distribución normal estándar ($z$-score) multiplicados por la raíz de la varianza condicional acumulada.

### 5. Motor de Señales & Simulador de Backtesting
* **Generación de Alertas:** Evaluación del retorno de la última sesión frente a los umbrales estadísticos del GARCH. Si el precio quiebra una banda de volatilidad y el filtro de tendencia del modelo VAR confirma la dirección, el sistema emite señales operativas (*Compra/Venta* con distintos niveles de fuerza).
* **Backtesting:** Simulación histórica con gestión de riesgo integrada. Ejecuta operaciones con tamaño fijo, registrando el impacto de un **Stop Loss dinámico en pips** y graficando la curva de capital (*Equity Curve*) junto a métricas críticas como el *Win Rate*.

---

## 🚀 Instalación y Uso Local

### 1. Clonar el repositorio
```bash
git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
cd tu-repositorio
