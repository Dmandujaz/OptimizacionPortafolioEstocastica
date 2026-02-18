"""
Script de simulación completa:
1. Ajuste ARCH/GARCH en datos históricos
2. Simulación de 8000 trayectorias
3. Sampleo de 5 casos diferentes
4. Optimización de cada caso con Lagrangiano
5. Comparación con rendimientos reales
"""

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURACIÓN GLOBAL
# ==========================================

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'XOM', 'JNJ', 'KO']
GAMMA = 20.0  # Penalización ALTA a la varianza
LIMITE_ACTIVO = 0.30
N_TRAYECTORIAS = 8000
HORIZONTE_SEMANAS = 13  # 3 meses
N_CASOS_SAMPLEO = 5

# ==========================================
# PASO 1: OBTENER DATOS HISTÓRICOS
# ==========================================

def obtener_datos_historicos(tickers, años=2):
    """Obtiene datos semanales del periodo 2024-2025"""
    fecha_fin = datetime.now()
    fecha_inicio = datetime(2024, 1, 1)  # Desde enero 2024
    
    print(f"\n{'='*80}")
    print("DESCARGANDO DATOS HISTÓRICOS")
    print(f"{'='*80}")
    print(f"Período de entrenamiento: {fecha_inicio.date()} a {fecha_fin.date()}")
    
    datos = yf.download(
        tickers=' '.join(tickers),
        start=fecha_inicio,
        end=fecha_fin,
        interval='1wk',
        auto_adjust=False,
        progress=True
    )
    
    # Extraer precios
    precios = pd.DataFrame()
    if isinstance(datos.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                if ('Adj Close', ticker) in datos.columns:
                    precios[ticker] = datos[('Adj Close', ticker)]
                elif ('Close', ticker) in datos.columns:
                    precios[ticker] = datos[('Close', ticker)]
            except:
                print(f"  ⚠ No se pudo obtener {ticker}")
    else:
        if 'Adj Close' in datos.columns:
            precios = datos['Adj Close']
        elif 'Close' in datos.columns:
            precios = datos['Close']
    
    precios = precios.dropna()
    
    # Separar: últimas 13 semanas (3 meses) para comparación, resto para entrenamiento
    precios_entrenamiento = precios.iloc[:-13]
    precios_prueba = precios.iloc[-13:]
    
    print(f"\n✓ Datos de entrenamiento: {len(precios_entrenamiento)} semanas")
    print(f"✓ Datos de prueba (últimos 3 meses): {len(precios_prueba)} semanas")
    
    return precios_entrenamiento, precios_prueba, precios

# ==========================================
# PASO 2: AJUSTAR MODELOS GARCH
# ==========================================

def ajustar_modelos_garch(precios_df):
    """Ajusta GARCH(1,1) a cada acción"""
    parametros = {}
    
    print(f"\n{'='*80}")
    print("AJUSTANDO MODELOS GARCH(1,1)")
    print(f"{'='*80}")
    
    for ticker in precios_df.columns:
        print(f"\n{ticker}...", end=" ")
        
        precios = precios_df[ticker].dropna()
        retornos = 100 * np.log(precios / precios.shift(1)).dropna()
        
        try:
            modelo = arch_model(retornos, vol='Garch', p=1, q=1, 
                              mean='Constant', dist='Normal')
            resultado = modelo.fit(disp='off')
            
            params = resultado.params
            parametros[ticker] = {
                'mu': params['mu'],
                'omega': params['omega'],
                'alpha': params['alpha[1]'],
                'beta': params['beta[1]'],
                'precio_inicial': float(precios.iloc[-1]),
                'volatilidad_actual': float(np.sqrt(resultado.conditional_volatility.iloc[-1]))
            }
            
            print(f"✓ μ={params['mu']:.4f}, α={params['alpha[1]']:.4f}, β={params['beta[1]']:.4f}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n{'='*80}")
    print(f"✓ Modelos ajustados: {len(parametros)}/{len(precios_df.columns)}")
    
    return parametros

# ==========================================
# PASO 3: SIMULAR 8000 TRAYECTORIAS
# ==========================================

def simular_8000_trayectorias(parametros_garch, horizonte=52, semilla=42):
    """Simula 8000 trayectorias completas del portafolio"""
    
    np.random.seed(semilla)
    tickers = list(parametros_garch.keys())
    n_activos = len(tickers)
    n_trayectorias = N_TRAYECTORIAS
    
    print(f"\n{'='*80}")
    print("GENERANDO 8000 TRAYECTORIAS COMPLETAS")
    print(f"{'='*80}")
    print(f"Activos: {n_activos}")
    print(f"Trayectorias: {n_trayectorias}")
    print(f"Horizonte: {horizonte} semanas")
    
    # Matriz: [trayectorias, activos, tiempo]
    retornos_simulados = np.zeros((n_trayectorias, n_activos, horizonte))
    
    for idx, ticker in enumerate(tickers):
        print(f"Simulando {ticker}... ({idx+1}/{n_activos})")
        params = parametros_garch[ticker]
        
        for traj in range(n_trayectorias):
            # Simular volatilidad GARCH
            volatilidades = simular_volatilidad_garch(
                omega=params['omega'],
                alpha=params['alpha'],
                beta=params['beta'],
                vol_inicial=params['volatilidad_actual'],
                T=horizonte
            )
            
            # Simular retornos
            retornos = params['mu'] + volatilidades * np.random.randn(horizonte)
            retornos_simulados[traj, idx, :] = retornos
    
    print(f"\n✓ Simulación completada: {retornos_simulados.shape}")
    
    return retornos_simulados, tickers

def simular_volatilidad_garch(omega, alpha, beta, vol_inicial, T):
    """Simula volatilidad condicional GARCH(1,1)"""
    volatilidades = np.zeros(T)
    varianza = vol_inicial ** 2
    
    for t in range(T):
        if t > 0:
            shock = volatilidades[t-1] * np.random.randn()
            varianza = omega + alpha * (shock ** 2) + beta * varianza
        
        volatilidades[t] = np.sqrt(max(varianza, 0.01))
    
    return volatilidades

# ==========================================
# PASO 4: SAMPLEAR 5 CASOS DIFERENTES
# ==========================================

def samplear_casos(retornos_simulados, n_casos=5):
    """
    Samplea 5 casos diferentes de las 8000 trayectorias
    Cada caso tendrá ~1600 trayectorias (20% del total)
    """
    n_total = retornos_simulados.shape[0]
    n_por_caso = n_total // n_casos
    
    print(f"\n{'='*80}")
    print("SAMPLEANDO CASOS PARA OPTIMIZACIÓN")
    print(f"{'='*80}")
    print(f"Total de trayectorias: {n_total}")
    print(f"Casos a generar: {n_casos}")
    print(f"Trayectorias por caso: {n_por_caso}")
    
    casos = {}
    indices_usados = set()
    
    for i in range(n_casos):
        # Sampleo aleatorio sin reemplazo
        indices_disponibles = list(set(range(n_total)) - indices_usados)
        indices_caso = np.random.choice(indices_disponibles, size=n_por_caso, replace=False)
        indices_usados.update(indices_caso)
        
        casos[f'Caso_{i+1}'] = retornos_simulados[indices_caso]
        
        # Estadísticas del caso
        retornos_finales = np.array([
            np.sum(retornos_simulados[idx], axis=1).sum() / 100
            for idx in indices_caso
        ])
        
        print(f"\nCaso {i+1}:")
        print(f"  Retorno medio: {np.mean(retornos_finales)*100:.2f}%")
        print(f"  Desv. Std: {np.std(retornos_finales)*100:.2f}%")
    
    return casos

# ==========================================
# PASO 5: OPTIMIZAR CADA CASO
# ==========================================

def optimizar_caso(retornos_caso, tickers, gamma=2.0, limite=0.30):
    """Optimiza un caso usando Lagrangiano Aumentado"""
    
    n_activos = len(tickers)
    n_sims = retornos_caso.shape[0]
    
    # Función objetivo
    def objetivo(w):
        retornos_portafolio = np.zeros(n_sims)
        
        for sim in range(n_sims):
            retorno_sim = 0.0
            for idx in range(n_activos):
                retorno_activo = np.sum(retornos_caso[sim, idx, :]) / 100
                retorno_sim += w[idx] * (np.exp(retorno_activo) - 1)
            retornos_portafolio[sim] = retorno_sim
        
        media = np.mean(retornos_portafolio)
        varianza = np.var(retornos_portafolio)
        
        return -media + gamma * varianza
    
    # Restricciones y límites
    restricciones = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    limites = [(0, limite) for _ in range(n_activos)]
    
    # Punto inicial inteligente
    w0 = np.array([0.08, 0.08, 0.12, 0.08, 0.08, 0.12, 0.15, 0.08, 0.13, 0.08])
    
    # Optimizar
    resultado = minimize(
        objetivo,
        w0,
        method='SLSQP',
        bounds=limites,
        constraints=restricciones,
        options={'maxiter': 100, 'disp': False, 'ftol': 1e-8}
    )
    
    if resultado.success:
        pesos = resultado.x
        
        # Calcular métricas del portafolio óptimo
        retornos_optimos = np.zeros(n_sims)
        for sim in range(n_sims):
            retorno_sim = 0.0
            for idx in range(n_activos):
                retorno_activo = np.sum(retornos_caso[sim, idx, :]) / 100
                retorno_sim += pesos[idx] * (np.exp(retorno_activo) - 1)
            retornos_optimos[sim] = retorno_sim
        
        return {
            'pesos': pesos,
            'retornos_simulados': retornos_optimos,
            'media': np.mean(retornos_optimos),
            'desv': np.std(retornos_optimos),
            'var_95': np.percentile(retornos_optimos, 5),
            'exito': True
        }
    else:
        return {'exito': False}

# ==========================================
# PASO 6: CALCULAR RENDIMIENTOS REALES
# ==========================================

def calcular_rendimientos_reales(precios_prueba):
    """Calcula el rendimiento real de los últimos 3 meses"""
    
    print(f"\n{'='*80}")
    print("CALCULANDO RENDIMIENTOS REALES")
    print(f"{'='*80}")
    
    # Retornos reales de cada acción
    retornos_reales = {}
    for ticker in precios_prueba.columns:
        precio_inicial = precios_prueba[ticker].iloc[0]
        precio_final = precios_prueba[ticker].iloc[-1]
        retorno = (precio_final - precio_inicial) / precio_inicial
        retornos_reales[ticker] = retorno
        print(f"{ticker}: {retorno*100:.2f}%")
    
    return retornos_reales

def calcular_rendimiento_portafolio_real(pesos, retornos_reales, tickers):
    """Calcula el rendimiento de un portafolio con datos reales"""
    retorno_total = sum(pesos[i] * retornos_reales[ticker] 
                       for i, ticker in enumerate(tickers))
    return retorno_total

# ==========================================
# PASO 7: VISUALIZACIÓN COMPLETA
# ==========================================

def visualizar_resultados_completos(casos_optimizados, retornos_reales, 
                                   retornos_simulados, tickers):
    """Genera visualizaciones completas de presentación"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Comparación de pesos de los 5 casos
    ax1 = plt.subplot(3, 4, 1)
    x = np.arange(len(tickers))
    width = 0.15
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (nombre, resultado) in enumerate(casos_optimizados.items()):
        ax1.bar(x + i*width, resultado['pesos']*100, width, 
               label=nombre, color=colores[i], alpha=0.8)
    
    ax1.set_xlabel('Activos')
    ax1.set_ylabel('Peso (%)')
    ax1.set_title('Pesos Óptimos - 5 Casos', fontweight='bold', fontsize=12)
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(tickers, rotation=45)
    ax1.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Límite 30%')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribución de retornos - Caso 1
    ax2 = plt.subplot(3, 4, 2)
    caso_1 = list(casos_optimizados.values())[0]
    media_c1 = caso_1['media']
    ax2.hist(caso_1['retornos_simulados']*100, bins=50, 
            color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(media_c1*100, color='red', linestyle='--', 
               linewidth=2, label=f"Media: {media_c1*100:.2f}%")
    ax2.set_xlabel('Retorno (%)')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución - Caso 1', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribución de retornos - Caso 2
    ax3 = plt.subplot(3, 4, 3)
    caso_2 = list(casos_optimizados.values())[1]
    media_c2 = caso_2['media']
    ax3.hist(caso_2['retornos_simulados']*100, bins=50, 
            color='orange', alpha=0.7, edgecolor='black')
    ax3.axvline(media_c2*100, color='red', linestyle='--', 
               linewidth=2, label=f"Media: {media_c2*100:.2f}%")
    ax3.set_xlabel('Retorno (%)')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title('Distribución - Caso 2', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Comparación métricas de los 5 casos
    ax4 = plt.subplot(3, 4, 4)
    casos_nombres = list(casos_optimizados.keys())
    medias = [r['media']*100 for r in casos_optimizados.values()]
    desvs = [r['desv']*100 for r in casos_optimizados.values()]
    
    x_pos = np.arange(len(casos_nombres))
    ax4.bar(x_pos - 0.2, medias, 0.4, label='Retorno Esperado', color='green', alpha=0.7)
    ax4.bar(x_pos + 0.2, desvs, 0.4, label='Volatilidad', color='red', alpha=0.7)
    ax4.set_ylabel('Porcentaje (%)')
    ax4.set_title('Métricas por Caso', fontweight='bold', fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(casos_nombres, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Simulaciones vs Real - Caso 1
    ax5 = plt.subplot(3, 4, 5)
    retorno_real_1 = calcular_rendimiento_portafolio_real(
        list(casos_optimizados.values())[0]['pesos'], 
        retornos_reales, 
        tickers
    )
    ax5.hist(caso_1['retornos_simulados']*100, bins=50, 
            color='lightblue', alpha=0.6, edgecolor='black', label='Simulaciones')
    media_caso_1 = caso_1['media']
    ax5.axvline(retorno_real_1*100, color='red', linestyle='-', 
               linewidth=3, label=f'Real: {retorno_real_1*100:.2f}%')
    ax5.axvline(media_caso_1*100, color='orange', linestyle='--', 
               linewidth=2, label=f'Esperado: {media_caso_1*100:.2f}%')
    ax5.set_xlabel('Retorno (%)')
    ax5.set_ylabel('Frecuencia')
    ax5.set_title('Caso 1: Simulado vs Real', fontweight='bold', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Simulaciones vs Real - Caso 3
    ax6 = plt.subplot(3, 4, 6)
    caso_3 = list(casos_optimizados.values())[2]
    retorno_real_3 = calcular_rendimiento_portafolio_real(
        caso_3['pesos'], 
        retornos_reales, 
        tickers
    )
    media_caso_3 = caso_3['media']
    ax6.hist(caso_3['retornos_simulados']*100, bins=50, 
            color='lightgreen', alpha=0.6, edgecolor='black', label='Simulaciones')
    ax6.axvline(retorno_real_3*100, color='red', linestyle='-', 
               linewidth=3, label=f'Real: {retorno_real_3*100:.2f}%')
    ax6.axvline(media_caso_3*100, color='orange', linestyle='--', 
               linewidth=2, label=f'Esperado: {media_caso_3*100:.2f}%')
    ax6.set_xlabel('Retorno (%)')
    ax6.set_ylabel('Frecuencia')
    ax6.set_title('Caso 3: Simulado vs Real', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Comparación todos los casos vs Real
    ax7 = plt.subplot(3, 4, 7)
    for i, (nombre, resultado) in enumerate(casos_optimizados.items()):
        retorno_real = calcular_rendimiento_portafolio_real(
            resultado['pesos'], retornos_reales, tickers
        )
        ax7.scatter(resultado['desv']*100, resultado['media']*100, 
                   s=200, alpha=0.7, color=colores[i], label=nombre)
        ax7.scatter(resultado['desv']*100, retorno_real*100, 
                   marker='x', s=300, color=colores[i], linewidths=3)
    
    ax7.set_xlabel('Volatilidad (%)')
    ax7.set_ylabel('Retorno (%)')
    ax7.set_title('Frontera Eficiente\n(○=Esperado, ×=Real)', 
                 fontweight='bold', fontsize=12)
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 8. Ratio Sharpe comparación
    ax8 = plt.subplot(3, 4, 8)
    sharpes_esperados = [r['media']/r['desv'] for r in casos_optimizados.values()]
    retornos_reales_casos = []
    for nombre, resultado in casos_optimizados.items():
        retorno_real = calcular_rendimiento_portafolio_real(
            resultado['pesos'], retornos_reales, tickers
        )
        retornos_reales_casos.append(retorno_real)
    
    sharpes_reales = [retornos_reales_casos[i] / list(casos_optimizados.values())[i]['desv'] 
                     for i in range(len(casos_nombres))]
    
    x_pos = np.arange(len(casos_nombres))
    ax8.bar(x_pos - 0.2, sharpes_esperados, 0.4, label='Sharpe Esperado', 
           color='blue', alpha=0.7)
    ax8.bar(x_pos + 0.2, sharpes_reales, 0.4, label='Sharpe Real', 
           color='green', alpha=0.7)
    ax8.set_ylabel('Ratio Sharpe')
    ax8.set_title('Ratio Sharpe: Esperado vs Real', fontweight='bold', fontsize=12)
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(casos_nombres, rotation=45)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9-10. Distribuciones casos 4 y 5
    ax9 = plt.subplot(3, 4, 9)
    caso_4 = list(casos_optimizados.values())[3]
    media_c4 = caso_4['media']
    ax9.hist(caso_4['retornos_simulados']*100, bins=50, 
            color='purple', alpha=0.7, edgecolor='black')
    ax9.axvline(media_c4*100, color='red', linestyle='--', 
               linewidth=2, label=f"Media: {media_c4*100:.2f}%")
    ax9.set_xlabel('Retorno (%)')
    ax9.set_ylabel('Frecuencia')
    ax9.set_title('Distribución - Caso 4', fontweight='bold', fontsize=12)
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    ax10 = plt.subplot(3, 4, 10)
    caso_5 = list(casos_optimizados.values())[4]
    media_c5 = caso_5['media']
    ax10.hist(caso_5['retornos_simulados']*100, bins=50, 
             color='brown', alpha=0.7, edgecolor='black')
    ax10.axvline(media_c5*100, color='red', linestyle='--', 
                linewidth=2, label=f"Media: {media_c5*100:.2f}%")
    ax10.set_xlabel('Retorno (%)')
    ax10.set_ylabel('Frecuencia')
    ax10.set_title('Distribución - Caso 5', fontweight='bold', fontsize=12)
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Tabla resumen
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    texto_resumen = "RESUMEN COMPARATIVO\n" + "="*35 + "\n\n"
    for i, (nombre, resultado) in enumerate(casos_optimizados.items()):
        retorno_real = retornos_reales_casos[i]
        texto_resumen += f"{nombre}:\n"
        texto_resumen += f"  E[R]:  {resultado['media']*100:6.2f}%\n"
        texto_resumen += f"  Real:  {retorno_real*100:6.2f}%\n"
        texto_resumen += f"  σ:     {resultado['desv']*100:6.2f}%\n"
        texto_resumen += f"  VaR95: {resultado['var_95']*100:6.2f}%\n\n"
    
    ax11.text(0.1, 0.95, texto_resumen, fontsize=9, family='monospace',
             verticalalignment='top')
    
    # 12. Retornos individuales reales
    ax12 = plt.subplot(3, 4, 12)
    retornos_valores = [retornos_reales[t]*100 for t in tickers]
    colores_bars = ['green' if r > 0 else 'red' for r in retornos_valores]
    bars = ax12.barh(tickers, retornos_valores, color=colores_bars, alpha=0.7)
    ax12.set_xlabel('Retorno Real (%)')
    ax12.set_title('Retornos Reales por Activo\n(Último Año)', 
                  fontweight='bold', fontsize=12)
    ax12.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax12.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('analisis_completo_8000_trayectorias.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualización guardada: 'analisis_completo_8000_trayectorias.png'")
    plt.show()

# ==========================================
# SCRIPT PRINCIPAL
# ==========================================

def main():
    """Pipeline completo"""
    
    print("\n" + "="*80)
    print(" "*15 + "ANÁLISIS COMPLETO CON 8000 TRAYECTORIAS")
    print("="*80)
    print("Pipeline: GARCH → 8000 Simulaciones → 5 Casos → Optimización → Comparación Real")
    print("="*80)
    
    # PASO 1: Obtener datos
    precios_train, precios_test, precios_full = obtener_datos_historicos(TICKERS, años=2)
    
    # PASO 2: Ajustar GARCH
    parametros_garch = ajustar_modelos_garch(precios_train)
    
    # PASO 3: Simular 8000 trayectorias
    retornos_simulados, tickers = simular_8000_trayectorias(
        parametros_garch, 
        horizonte=HORIZONTE_SEMANAS
    )
    
    # Guardar simulaciones
    np.save('simulaciones_8000_trayectorias.npy', retornos_simulados)
    print(f"✓ Simulaciones guardadas en 'simulaciones_8000_trayectorias.npy'")
    
    # PASO 4: Samplear 5 casos
    casos = samplear_casos(retornos_simulados, n_casos=N_CASOS_SAMPLEO)
    
    # PASO 5: Optimizar cada caso
    print(f"\n{'='*80}")
    print("OPTIMIZANDO 5 CASOS CON LAGRANGIANO AUMENTADO")
    print(f"{'='*80}")
    
    casos_optimizados = {}
    for nombre, retornos_caso in casos.items():
        print(f"\nOptimizando {nombre}...")
        resultado = optimizar_caso(retornos_caso, tickers, gamma=GAMMA, limite=LIMITE_ACTIVO)
        
        if resultado['exito']:
            casos_optimizados[nombre] = resultado
            print(f"✓ {nombre} optimizado")
            print(f"  E[R]: {resultado['media']*100:.2f}%")
            print(f"  σ: {resultado['desv']*100:.2f}%")
            print(f"  Sharpe: {resultado['media']/resultado['desv']:.2f}")
            
            # Mostrar pesos
            print(f"  Pesos:")
            for i, ticker in enumerate(tickers):
                print(f"    {ticker}: {resultado['pesos'][i]*100:.1f}%")
        else:
            print(f"✗ Error en {nombre}")
    
    # PASO 6: Calcular rendimientos reales
    retornos_reales = calcular_rendimientos_reales(precios_test)
    
    # PASO 7: Comparar con datos reales
    print(f"\n{'='*80}")
    print("COMPARACIÓN CON RENDIMIENTOS REALES")
    print(f"{'='*80}")
    
    for nombre, resultado in casos_optimizados.items():
        retorno_real = calcular_rendimiento_portafolio_real(
            resultado['pesos'], 
            retornos_reales, 
            tickers
        )
        print(f"\n{nombre}:")
        print(f"  Retorno esperado (simulado): {resultado['media']*100:6.2f}%")
        print(f"  Retorno real (histórico):    {retorno_real*100:6.2f}%")
        print(f"  Diferencia:                  {(retorno_real - resultado['media'])*100:6.2f}%")
    
    # Guardar resultados
    resultados_df = pd.DataFrame({
        'Caso': list(casos_optimizados.keys()),
        'Retorno_Esperado_%': [r['media']*100 for r in casos_optimizados.values()],
        'Volatilidad_%': [r['desv']*100 for r in casos_optimizados.values()],
        'VaR_95_%': [r['var_95']*100 for r in casos_optimizados.values()],
        'Sharpe_Ratio': [r['media']/r['desv'] for r in casos_optimizados.values()],
    })
    
    # Agregar pesos
    for i, ticker in enumerate(tickers):
        resultados_df[f'Peso_{ticker}_%'] = [r['pesos'][i]*100 for r in casos_optimizados.values()]
    
    resultados_df.to_csv('resultados_5_casos_optimizados.csv', index=False)
    print(f"\n✓ Resultados guardados en 'resultados_5_casos_optimizados.csv'")
    
    # PASO 8: Visualizar todo
    print(f"\n{'='*80}")
    print("GENERANDO VISUALIZACIONES DE PRESENTACIÓN")
    print(f"{'='*80}")
    
    visualizar_resultados_completos(
        casos_optimizados, 
        retornos_reales, 
        retornos_simulados,
        tickers
    )
    
    print(f"\n{'='*80}")
    print("✓ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print(f"{'='*80}\n")
    
    print("\nArchivos generados:")
    print("  1. simulaciones_8000_trayectorias.npy - Simulaciones completas")
    print("  2. resultados_5_casos_optimizados.csv - Resultados optimización")
    print("  3. analisis_completo_8000_trayectorias.png - Visualización completa")


if __name__ == "__main__":
    main()
