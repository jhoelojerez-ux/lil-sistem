import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
from scipy.stats import norm

# Configuración de la página de Streamlit
st.set_page_config(page_title="Analizador Financiero VAR-GARCH", layout="wide")

st.title("📊 Análisis de Series de Tiempo: EUR/USD y VIX")
st.markdown("Esta aplicación replica el pipeline de modelado econométrico utilizando **VAR** y **GARCH(1,1)**.")

# --- SIDEBAR: CONTROLES ---
st.sidebar.header("Configuración de Datos")
periodo = st.sidebar.selectbox("Periodo de datos históricos", ["1y", "2y", "6mo"], index=0)
horizonte = st.sidebar.slider("Horizonte de Pronóstico (Días)", min_value=5, max_value=20, value=10)

# --- DESCARGA DE DATOS ---
@st.cache_data
def descargar_datos(periodo_input):
    eurusd_ticker = 'EURUSD=X'
    vix_ticker = '^VIX'
    eurusd = yf.download(eurusd_ticker, period=periodo_input, interval='1d')
    vix = yf.download(vix_ticker, period=periodo_input, interval='1d')
    return eurusd, vix

eurusd_data, vix_data = descargar_datos(periodo)

# Extraer Cierre y calcular retornos logarítmicos
eurusd_close = eurusd_data['Close']
vix_close = vix_data['Close']

eurusd_log_returns = np.log(eurusd_close / eurusd_close.shift(1))
vix_log_returns = np.log(vix_close / vix_close.shift(1))

# Unir dataframes limpiando NaNs
combined_returns = pd.concat([
    eurusd_log_returns.rename(columns={'EURUSD=X': 'EURUSD_Log_Returns'}),
    vix_log_returns.rename(columns={'^VIX': 'VIX_Log_Returns'})
], axis=1).dropna()

# Pestañas principales
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Datos & Modelo VAR", 
    "🔬 Diagnóstico ARCH/GARCH", 
    "🔮 Pronóstico de Precios",
    "🚨 Señales de Trading",
    "💰 Simulador Backtesting"
])

with tab1:
    st.header("1. Vector Autorregresivo (VAR)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Últimos datos combinados (Retornos Log)")
        st.dataframe(combined_returns.tail())
    with col2:
        lags_var = st.slider("Lags para el modelo VAR", min_value=1, max_value=10, value=3)
        model_var = VAR(combined_returns)
        results_var = model_var.fit(lags_var)
        st.success(f"Modelo VAR({lags_var}) adjusted successfully.")

    if st.checkbox("Mostrar Resumen Estadístico del VAR"):
        st.text(results_var.summary())

    st.subheader("Funciones de Respuesta al Impulso (IRF)")
    irf_lags = st.slider("Pasos hacia adelante para IRF", min_value=3, max_value=10, value=5)
    irf = results_var.irf(irf_lags)
    fig_irf = irf.plot(orth=False)
    fig_irf.set_size_inches(12, 6)
    st.pyplot(fig_irf)

with tab2:
    st.header("2. Modelado de Volatilidad (ARIMA + GARCH)")
    
    eurusd_log_returns_cleaned = eurusd_log_returns.dropna()
    
    arima_model = ARIMA(eurusd_log_returns_cleaned, order=(0, 0, 0))
    arima_results = arima_model.fit()
    arima_residuals = arima_results.resid
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Test ARCH-LM (Heterocedasticidad)")
        arch_lm_test = het_arch(arima_residuals, nlags=12)
        st.metric(label="Valor p del Test ARCH-LM", value=f"{arch_lm_test[1]:.4f}")
        if arch_lm_test[1] < 0.05:
            st.warning("Resultado: Hay evidencia de efectos ARCH (Volatilidad agrupada).")
        else:
            st.info("Resultado: No hay evidencia fuerte de efectos ARCH.")

    with col_g2:
        st.subheader("Ajuste GARCH(1,1)")
        if isinstance(eurusd_log_returns_cleaned, pd.DataFrame):
            residuals_for_garch = eurusd_log_returns_cleaned.iloc[:, 0]
        else:
            residuals_for_garch = eurusd_log_returns_cleaned
            
        garch_model = arch_model(residuals_for_garch, vol='Garch', p=1, q=1, dist='normal')
        garch_results = garch_model.fit(disp='off')
        st.success("Modelo GARCH(1,1) entrenado.")
        
    if st.checkbox("Mostrar Resumen del GARCH"):
        st.text(garch_results.summary())

with tab3:
    st.header("3. Pronóstico en Niveles de Precio con Bandas GARCH")
    
    # --- CORRECCIÓN INTEGRAL DEL BUG DE LA LÍNEA 117 ---
    if isinstance(eurusd_close, pd.DataFrame):
        ultimo_precio = float(eurusd_close.iloc[-1, 0])
    else:
        ultimo_precio = float(eurusd_close.iloc[-1])
    
    garch_forecast = garch_results.forecast(horizon=horizonte)
    media_arima_forecast = arima_results.params['const']
    
    retornos_pronosticados_arima = np.full(horizonte, media_arima_forecast)
    varianza_shocks_predicha = garch_forecast.variance.iloc[-1].values
    desviacion_shocks_predicha = np.sqrt(varianza_shocks_predicha)
    
    retornos_acumulados_central = np.cumsum(retornos_pronosticados_arima)
    precios_proyectados = ultimo_precio * np.exp(retornos_acumulados_central)
    
    z_95 = norm.ppf(0.975)
    z_60 = norm.ppf(0.80)
    
    banda_sup_ret_95 = np.cumsum(retornos_pronosticados_arima + z_95 * desviacion_shocks_predicha)
    banda_inf_ret_95 = np.cumsum(retornos_pronosticados_arima - z_95 * desviacion_shocks_predicha)
    banda_sup_ret_60 = np.cumsum(retornos_pronosticados_arima + z_60 * desviacion_shocks_predicha)
    banda_inf_ret_60 = np.cumsum(retornos_pronosticados_arima - z_60 * desviacion_shocks_predicha)
    
    precio_superior_95 = ultimo_precio * np.exp(banda_sup_ret_95)
    precio_inferior_95 = ultimo_precio * np.exp(banda_inf_ret_95)
    precio_superior_60 = ultimo_precio * np.exp(banda_sup_ret_60)
    precio_inferior_60 = ultimo_precio * np.exp(banda_inf_ret_60)
    
    pasos_futuros = [f"Día {i}" for i in range(1, horizonte + 1)]
    
    fig_forecast, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pasos_futuros, precios_proyectados, label='Pronóstico Central EUR/USD', color='blue', marker='o')
    ax.fill_between(pasos_futuros, precio_inferior_95, precio_superior_95, color='gray', alpha=0.15, label='95% Confianza')
    ax.fill_between(pasos_futuros, precio_inferior_60, precio_superior_60, color='lightgreen', alpha=0.3, label='60% Confianza')
    ax.axhline(ultimo_precio, color='red', linestyle='--', label=f'Último Cierre Real ({ultimo_precio:.5f})')
    
    ax.set_title('Forecast de EUR/USD con Bandas de Volatilidad GARCH')
    ax.set_ylabel('Precio (EUR/USD)')
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=':')
    ax.set_ylim(ultimo_precio - 0.015, ultimo_precio + 0.015)
    st.pyplot(fig_forecast)
    
    df_forecast_resumen = pd.DataFrame({
        "Día": pasos_futuros,
        "Límite Inferior (95%)": precio_inferior_95,
        "Proyección Central": precios_proyectados,
        "Límite Superior (95%)": precio_superior_95
    }).set_index("Día")
    
    st.subheader("Valores Estimados de la Proyección")
    st.dataframe(df_forecast_resumen.style.format("{:.5f}"))

with tab4:
    st.header("🚨 Generador de Señales Cuantitativas Avanzado")
    st.markdown("Las señales se calculan transformando la volatilidad condicional del modelo GARCH en bandas de precio reales sobre el último cierre.")

    if isinstance(eurusd_close, pd.DataFrame):
        ultimo_precio_sig = float(eurusd_close.iloc[-1, 0])
    else:
        ultimo_precio_sig = float(eurusd_close.iloc[-1])
        
    ultimo_retorno_eurusd = combined_returns['EURUSD_Log_Returns'].iloc[-1]
    
    historia_vol = garch_results.conditional_volatility
    ultima_vol_garch = historia_vol.iloc[-1]
    
    prediccion_var = results_var.forecast(combined_returns.values[-lags_var:], steps=1)
    tendencia_var_eurusd = prediccion_var[0, 0]
    filtro_bullish = tendencia_var_eurusd > 0
    filtro_bearish = tendencia_var_eurusd < 0

    z_95 = 1.96
    z_50 = 0.67
    z_30 = 0.39

    precio_sup_95 = ultimo_precio_sig * np.exp(media_arima_forecast + z_95 * ultima_vol_garch)
    precio_inf_95 = ultimo_precio_sig * np.exp(media_arima_forecast - z_95 * ultima_vol_garch)
    precio_sup_50 = ultimo_precio_sig * np.exp(media_arima_forecast + z_50 * ultima_vol_garch)
    precio_inf_50 = ultimo_precio_sig * np.exp(media_arima_forecast - z_50 * ultima_vol_garch)
    precio_sup_30 = ultimo_precio_sig * np.exp(media_arima_forecast + z_30 * ultima_vol_garch)
    precio_inf_30 = ultimo_precio_sig * np.exp(media_arima_forecast - z_30 * ultima_vol_garch)

    st.subheader("📊 Umbrales de Activación en Puntos de Precio")
    st.markdown(f"""
    * **Nivel Fuerte (95%):** Cortos arriba de **{precio_sup_95:.5f}** | Largos abajo de **{precio_inf_95:.5f}**
    * **Nivel Moderado (50%):** Cortos arriba de **{precio_sup_50:.5f}** | Largos abajo de **{precio_inf_50:.5f}**
    * **Nivel Débil (30%):** Cortos arriba de **{precio_sup_30:.5f}** | Largos abajo de **{precio_inf_30:.5f}**
    """)

    st.subheader("Estado Actual del Par")
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Último Cierre Real", f"{ultimo_precio_sig:.5f}")
    col_s2.metric("Techo Dinámico (95%)", f"{precio_sup_95:.5f}")
    col_s3.metric("Soporte Dinámico (95%)", f"{precio_inf_95:.5f}")

    clase_senal = "NEUTRAL (Mantener)"
    detalles_senal = "El precio cotiza dentro de las zonas de equilibrio estadístico o el filtro VAR no valida el movimiento."
    color_alert = st.info

    if ultimo_retorno_eurusd > (media_arima_forecast + z_95 * ultima_vol_garch) and filtro_bearish:
        clase_senal = "VENTA FUERTE (Bearish)"
        detalles_senal = f"El precio perforó el techo crítico de {precio_sup_95:.5f}. El modelo VAR confirma presión bajista para la próxima sesión."
        color_alert = st.error
    elif ultimo_retorno_eurusd > (media_arima_forecast + z_50 * ultima_vol_garch) and filtro_bearish:
        clase_senal = "VENTA MODERADA (Bearish)"
        detalles_senal = f"Precio por encima de {precio_sup_50:.5f}. Zona de distribución estadística validada por el filtro VAR."
        color_alert = st.warning
    elif ultimo_retorno_eurusd > (media_arima_forecast + z_30 * ultima_vol_garch) and filtro_bearish:
        clase_senal = "VENTA DÉBIL (Bearish)"
        detalles_senal = f"El precio cotiza en {ultimo_precio_sig:.5f}, superando la resistencia leve de {precio_sup_30:.5f}."
        color_alert = st.warning

    elif ultimo_retorno_eurusd < (media_arima_forecast - z_95 * ultima_vol_garch) and filtro_bullish:
        clase_senal = "COMPRA FUERTE (Bullish)"
        detalles_senal = f"Pánico estadístico. El precio cayó por debajo del soporte mayor de {precio_inf_95:.5f}. El VAR proyecta reversión alcista."
        color_alert = st.success
    elif ultimo_retorno_eurusd < (media_arima_forecast - z_50 * ultima_vol_garch) and filtro_bullish:
        clase_senal = "COMPRA MODERADA (Bullish)"
        detalles_senal = f"Precio por debajo de {precio_inf_50:.5f}. Zona de acumulación respaldada por el vector autorregresivo."
        color_alert = st.success
    elif ultimo_retorno_eurusd < (media_arima_forecast - z_30 * ultima_vol_garch) and filtro_bullish:
        clase_senal = "COMPRA DÉBIL (Bullish)"
        detalles_senal = f"Soporte menor en {precio_inf_30:.5f} vulnerado con sesgo alcista en el VAR."
        color_alert = st.success

    st.subheader("🎯 Orden Operativa Sugerida:")
    color_alert(f"**Recomendación: {clase_senal}**")
    st.write(detalles_senal)
    st.caption(f"*Dirección matemática esperada por el VAR para mañana: {'Alcista' if filtro_bullish else 'Bajista'} ({tendencia_var_eurusd:.5f})*")

    st.subheader("📋 Historial de Precios de Activación Pasados")
    df_signals = pd.DataFrame(index=combined_returns.index)
    if isinstance(eurusd_close, pd.DataFrame):
        df_signals['Precio_Cierre'] = eurusd_close.iloc[:, 0].reindex(combined_returns.index)
    else:
        df_signals['Precio_Cierre'] = eurusd_close.reindex(combined_returns.index)
    df_signals['Precio_Anterior'] = df_signals['Precio_Cierre'].shift(1)
    df_signals['Retorno'] = combined_returns['EURUSD_Log_Returns']
    df_signals['Vol_GARCH'] = historia_vol
    
    df_signals['Soporte_95'] = df_signals['Precio_Anterior'] * np.exp(media_arima_forecast - z_95 * df_signals['Vol_GARCH'])
    df_signals['Techo_95'] = df_signals['Precio_Anterior'] * np.exp(media_arima_forecast + z_95 * df_signals['Vol_GARCH'])
    
    df_signals['Señal'] = "NEUTRAL"
    df_signals.loc[df_signals['Retorno'] > (media_arima_forecast + z_30 * df_signals['Vol_GARCH']), 'Señal'] = "VENTA DÉBIL"
    df_signals.loc[df_signals['Retorno'] < (media_arima_forecast - z_30 * df_signals['Vol_GARCH']), 'Señal'] = "COMPRA DÉBIL"
    df_signals.loc[df_signals['Retorno'] > (media_arima_forecast + z_50 * df_signals['Vol_GARCH']), 'Señal'] = "VENTA MODERADA"
    df_signals.loc[df_signals['Retorno'] < (media_arima_forecast - z_50 * df_signals['Vol_GARCH']), 'Señal'] = "COMPRA MODERADA"
    df_signals.loc[df_signals['Retorno'] > (media_arima_forecast + z_95 * df_signals['Vol_GARCH']), 'Señal'] = "VENTA FUERTE"
    df_signals.loc[df_signals['Retorno'] < (media_arima_forecast - z_95 * df_signals['Vol_GARCH']), 'Señal'] = "COMPRA FUERTE"
    
    alertas_efectivas = df_signals[df_signals['Señal'] != "NEUTRAL"].tail(12).dropna()
    
    if not alertas_efectivas.empty:
        def color_map(val):
            if 'COMPRA' in str(val): return 'background-color: rgba(46, 204, 113, 0.2); color: #2ecc71; font-weight: bold;'
            if 'VENTA' in str(val): return 'background-color: rgba(231, 76, 60, 0.2); color: #e74c3c; font-weight: bold;'
            return ''
            
        st.dataframe(
            alertas_efectivas[['Precio_Cierre', 'Soporte_95', 'Techo_95', 'Señal']]
            .style.format("{:.5f}", subset=['Precio_Cierre', 'Soporte_95', 'Techo_95'])
            .map(color_map, subset=['Señal'])
        )
    else:
        st.write("No se registraron anomalías de precio en las últimas sesiones.")

with tab5:
    st.header("💰 Simulador de Backtesting Operativo")
    st.markdown("Evaluación histórica ejecutando operaciones de **50 USD** con un capital inicial de **100 USD** basado en el umbral de confianza seleccionado.")

    st.subheader("Configuración de Señales y Confianza")
    col_conf1, col_conf2 = st.columns(2)
    
    with col_conf1:
        confianza_pct = st.selectbox(
            "Selecciona el Nivel de Confianza para los Umbrales:",
            options=[30, 50, 65, 95],
            index=1
        )
    
    mapeo_z = {30: 0.39, 50: 0.67, 65: 0.93, 95: 1.96}
    z_seleccionado = mapeo_z[confianza_pct]

    capital_inicial = 100.0
    tamano_operacion = 50.0
    
    with col_conf2:
        stop_loss_pips = st.slider("Stop Loss Moderado (en Pips)", min_value=10, max_value=100, value=25, step=5)
        sl_distancia = stop_loss_pips / 10000.0 

    df_bt = pd.DataFrame(index=combined_returns.index)
    if isinstance(eurusd_close, pd.DataFrame):
        df_bt['Precio'] = eurusd_close.iloc[:, 0].reindex(combined_returns.index)
    else:
        df_bt['Precio'] = eurusd_close.reindex(combined_returns.index)
    df_bt['Precio_Anterior'] = df_bt['Precio'].shift(1)
    df_bt['Retorno'] = combined_returns['EURUSD_Log_Returns']
    df_bt['Vol_GARCH'] = garch_results.conditional_volatility
    
    df_bt['Soporte_Dinamico'] = df_bt['Precio_Anterior'] * np.exp(media_arima_forecast - z_seleccionado * df_bt['Vol_GARCH'])
    df_bt['Techo_Dinamico'] = df_bt['Precio_Anterior'] * np.exp(media_arima_forecast + z_seleccionado * df_bt['Vol_GARCH'])
    df_bt.dropna(inplace=True)

    capital_actual = capital_inicial
    historial_capital = []
    operaciones = []

    for i in range(len(df_bt)):
        fecha = df_bt.index[i]
        precio_entrada = df_bt['Precio'].iloc[i]
        retorno = df_bt['Retorno'].iloc[i]
        vol_actual = df_bt['Vol_GARCH'].iloc[i]
        
        historial_capital.append(capital_actual)
        
        if capital_actual < tamano_operacion:
            continue
            
        if retorno < (media_arima_forecast - z_seleccionado * vol_actual):
            if i + 1 < len(df_bt):
                precio_salida = df_bt['Precio'].iloc[i+1]
                precio_sl = precio_entrada - sl_distancia
                
                if precio_salida <= precio_sl:
                    precio_salida = precio_sl
                    resultado_pnl = -tamano_operacion * (sl_distancia / precio_entrada)
                    tipo_cierre = "Stop Loss 🔴"
                else:
                    resultado_pnl = tamano_operacion * ((precio_salida - precio_entrada) / precio_entrada)
                    tipo_cierre = "Cierre Regular 🟢" if resultado_pnl > 0 else "Cierre con Pérdida 🟡"
                
                capital_actual += resultado_pnl
                operaciones.append({"Fecha": fecha, "Tipo": "COMPRA (Long)", "Entrada": precio_entrada, "Salida": precio_salida, "PnL USD": resultado_pnl, "Estado": tipo_cierre})

        elif retorno > (media_arima_forecast + z_seleccionado * vol_actual):
            if i + 1 < len(df_bt):
                precio_salida = df_bt['Precio'].iloc[i+1]
                precio_sl = precio_entrada + sl_distancia
                
                if precio_salida >= precio_sl:
                    precio_salida = precio_sl
                    resultado_pnl = -tamano_operacion * (sl_distancia / precio_entrada)
                    tipo_cierre = "Stop Loss 🔴"
                else:
                    resultado_pnl = tamano_operacion * ((precio_entrada - precio_salida) / precio_entrada)
                    tipo_cierre = "Cierre Regular 🟢" if resultado_pnl > 0 else "Cierre con Pérdida 🟡"
                
                capital_actual += resultado_pnl
                operaciones.append({"Fecha": fecha, "Tipo": "VENTA (Short)", "Entrada": precio_entrada, "Salida": precio_salida, "PnL USD": resultado_pnl, "Estado": tipo_cierre})

    df_ops = pd.DataFrame(operaciones)
    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric("Capital Final", f"{capital_actual:.2f} USD", f"{capital_actual - capital_inicial:.2f} USD")
    
    if not df_ops.empty:
        win_rate = (df_ops['PnL USD'] > 0).sum() / len(df_ops) * 100
        col_b2.metric("Total Operaciones", f"{len(df_ops)}")
        col_b3.metric("Efectividad (Win Rate)", f"{win_rate:.1f}%")
        
        st.subheader(f"📈 Rendimiento de la Cuenta (Equity Curve - Confianza {confianza_pct}%)")
        fig_cap, ax_cap = plt.subplots(figsize=(10, 3.5))
        ax_cap.plot(df_bt.index, historial_capital, color="#2ecc71", linewidth=2)
        ax_cap.axhline(capital_inicial, color="white", linestyle="--", alpha=0.3)
        ax_cap.set_ylabel("Balance en USD")
        ax_cap.grid(True, linestyle=":")
        st.pyplot(fig_cap)
        
        st.subheader("📋 Libro de Órdenes Ejecutadas")
        
        def color_status(val):
            if "🟢" in str(val): return "color: #2ecc71; font-weight: bold;"
            if "🔴" in str(val): return "color: #e74c3c; font-weight: bold;"
            return "color: #f1c40f;"

        st.dataframe(
            df_ops.set_index("Fecha")
            .style.format("{:.5f}", subset=["Entrada", "Salida"])
            .format("{:.2f} USD", subset=["PnL USD"])
            .map(color_status, subset=["Estado"])
        )
    else:
        st.info(f"El mercado se mantuvo dentro de las bandas. No se registraron rupturas del {confianza_pct}% de confianza en este periodo.")
