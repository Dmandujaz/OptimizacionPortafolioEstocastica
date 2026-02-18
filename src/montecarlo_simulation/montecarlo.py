import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats


@dataclass
class GBMParameters:
    """
    Parámetros del Movimiento Browniano Geométrico para una acción.
    
    Attributes:
        ticker: Símbolo de la acción
        mu: Drift (tendencia) - retorno esperado anualizado
        sigma: Volatilidad - desviación estándar anualizada
        S0: Precio inicial
    """
    ticker: str
    mu: float
    sigma: float
    S0: float
    
    def __repr__(self):
        return f"GBM({self.ticker}): μ={self.mu:.4f}, σ={self.sigma:.4f}, S₀={self.S0:.2f}"


class GeometricBrownianMotion:
    """
    Clase para ajustar y simular Movimiento Browniano Geométrico.
    
    El Movimiento Browniano Geométrico (GBM) modela el precio de una acción como:
    dS_t = μ * S_t * dt + σ * S_t * dW_t
    
    donde:
    - S_t: Precio en el tiempo t
    - μ: Drift (tasa de retorno esperada)
    - σ: Volatilidad (desviación estándar de los retornos)
    - W_t: Proceso de Wiener (movimiento browniano estándar)
    
    Método de estimación:
    ====================
    1. **Cálculo de retornos logarítmicos**: 
       r_t = ln(S_t / S_{t-1})
       
    2. **Estimación de parámetros**:
       - μ̂ = E[r_t] + (σ²/2)  (ajuste de Itô)
       - σ̂ = std(r_t)
       
    3. **Anualización**:
       - Si los datos son diarios: μ_anual = μ̂ * 252, σ_anual = σ̂ * √252
       - Si los datos son mensuales: μ_anual = μ̂ * 12, σ_anual = σ̂ * √12
       - Si los datos son por hora: μ_anual = μ̂ * 252 * 6.5, σ_anual = σ̂ * √(252 * 6.5)
    """
    
    def __init__(self, granularidad: str = 'dia'):
        """
        Inicializa el ajustador de GBM.
        
        Args:
            granularidad: 'dia', 'mes', 'hora' - afecta la anualización
        """
        self.granularidad = granularidad
        self.parametros: Dict[str, GBMParameters] = {}
        
        # Factores de anualización
        self.factores_anualizacion = {
            'dia': 252,        # Días de trading al año
            'mes': 12,         # Meses al año
            'semana': 52,      # Semanas al año
            'hora': 252 * 6.5, # Horas de trading al año (6.5 horas/día)
            'minuto': 252 * 6.5 * 60,
            '5minutos': 252 * 6.5 * 12,
            '15minutos': 252 * 6.5 * 4,
            '30minutos': 252 * 6.5 * 2,
            '90minutos': 252 * 6.5 * (60/90)
        }
    
    def ajustar_desde_csv(self, archivo_csv: str) -> Dict[str, GBMParameters]:
        """
        Ajusta modelos GBM para todas las acciones en un archivo CSV.
        
        Args:
            archivo_csv: Ruta al archivo CSV generado por RetrieveData
            
        Returns:
            Diccionario con parámetros GBM por acción
        """
        df = pd.read_csv(archivo_csv, index_col=0, parse_dates=True)
        return self.ajustar_desde_dataframe(df)
    
    def ajustar_desde_dataframe(self, df: pd.DataFrame) -> Dict[str, GBMParameters]:
        """
        Ajusta modelos GBM para todas las acciones en un DataFrame.
        
        Args:
            df: DataFrame con columnas en formato 'TICKER_Close'
            
        Returns:
            Diccionario con parámetros GBM por acción
        """
        # Identificar columnas de precios de cierre
        columnas_close = [col for col in df.columns if col.endswith('_Close')]
        
        if not columnas_close:
            raise ValueError("No se encontraron columnas de precios de cierre (*_Close)")
        
        print("\n" + "="*80)
        print("AJUSTE DE MODELOS DE MOVIMIENTO BROWNIANO GEOMÉTRICO (GBM)")
        print("="*80)
        print(f"\nMétodo: Estimación de Máxima Verosimilitud (MLE)")
        print(f"Granularidad: {self.granularidad}")
        print(f"Factor de anualización: {self.factores_anualizacion.get(self.granularidad, 252)}")
        print(f"Período de datos: {df.index[0]} a {df.index[-1]}")
        print(f"Número de observaciones: {len(df)}")
        
        for col in columnas_close:
            ticker = col.replace('_Close', '')
            precios = df[col].dropna()
            
            if len(precios) < 2:
                print(f"\n⚠️  {ticker}: Datos insuficientes")
                continue
            
            params = self._ajustar_accion(ticker, precios)
            self.parametros[ticker] = params
            
            # Mostrar estadísticas
            self._mostrar_estadisticas(ticker, precios, params)
        
        print("\n" + "="*80)
        return self.parametros
    
    def _ajustar_accion(self, ticker: str, precios: pd.Series) -> GBMParameters:
        """
        Ajusta un modelo GBM para una acción individual.
        
        Args:
            ticker: Símbolo de la acción
            precios: Serie de precios
            
        Returns:
            Parámetros del modelo GBM ajustado
        """
        # 1. Calcular retornos logarítmicos
        retornos_log = np.log(precios / precios.shift(1)).dropna()
        
        # 2. Estimar parámetros
        mu_periodo = retornos_log.mean()
        sigma_periodo = retornos_log.std()
        
        # 3. Anualizar
        factor = self.factores_anualizacion.get(self.granularidad, 252)
        
        # Ajuste de Itô para el drift
        mu_anual = mu_periodo * factor + 0.5 * (sigma_periodo ** 2) * factor
        sigma_anual = sigma_periodo * np.sqrt(factor)
        
        # Precio inicial (último precio observado)
        S0 = precios.iloc[-1]
        
        return GBMParameters(
            ticker=ticker,
            mu=mu_anual,
            sigma=sigma_anual,
            S0=S0
        )
    
    def _mostrar_estadisticas(self, ticker: str, precios: pd.Series, params: GBMParameters):
        """Muestra estadísticas detalladas del ajuste."""
        retornos_log = np.log(precios / precios.shift(1)).dropna()
        
        print(f"\n{'─'*80}")
        print(f"ACCIÓN: {ticker}")
        print(f"{'─'*80}")
        print(f"Precio inicial (S₀):        ${params.S0:.2f}")
        print(f"Precio mínimo:              ${precios.min():.2f}")
        print(f"Precio máximo:              ${precios.max():.2f}")
        print(f"Rango de precios:           ${precios.max() - precios.min():.2f}")
        
        print(f"\n📈 PARÁMETROS GBM (Anualizados):")
        print(f"Drift (μ):                  {params.mu:.4f} ({params.mu*100:.2f}% anual)")
        print(f"Volatilidad (σ):            {params.sigma:.4f} ({params.sigma*100:.2f}% anual)")
        
        print(f"\n📊 ESTADÍSTICAS DE RETORNOS:")
        print(f"Retorno promedio (período): {retornos_log.mean():.6f}")
        print(f"Desv. estándar (período):   {retornos_log.std():.6f}")
        print(f"Asimetría (skewness):       {retornos_log.skew():.4f}")
        print(f"Curtosis:                   {retornos_log.kurtosis():.4f}")
        
        # Test de normalidad (Jarque-Bera)
        jb_stat, jb_pvalue = stats.jarque_bera(retornos_log)
        print(f"\n🔍 TEST DE NORMALIDAD (Jarque-Bera):")
        print(f"Estadístico JB:             {jb_stat:.4f}")
        print(f"P-valor:                    {jb_pvalue:.4f}")
        if jb_pvalue > 0.05:
            print(f"Conclusión:                 ✓ Los retornos son aproximadamente normales")
        else:
            print(f"Conclusión:                 ⚠ Los retornos se desvían de la normalidad")
        
        # Retorno esperado anual y desviación
        print(f"\n💰 PROYECCIONES (1 año):")
        retorno_esperado = params.mu
        precio_esperado = params.S0 * np.exp(retorno_esperado)
        print(f"Retorno esperado:           {retorno_esperado*100:.2f}%")
        print(f"Precio esperado:            ${precio_esperado:.2f}")
        
    def obtener_vector_parametros(self) -> pd.DataFrame:
        """
        Obtiene un DataFrame con todos los parámetros ajustados.
        
        Returns:
            DataFrame con columnas: ticker, mu, sigma, S0
        """
        if not self.parametros:
            raise ValueError("Primero debe ajustar los modelos usando ajustar_desde_csv()")
        
        datos = []
        for ticker, params in self.parametros.items():
            datos.append({
                'ticker': ticker,
                'mu': params.mu,
                'sigma': params.sigma,
                'S0': params.S0
            })
        
        return pd.DataFrame(datos)


class MonteCarloSimulator:
    """
    Simulador de Monte Carlo para portafolios de acciones usando GBM.
    
    Simula trayectorias futuras de precios de acciones y calcula
    el valor esperado y la varianza del portafolio.
    """
    
    def __init__(self, parametros_gbm: Dict[str, GBMParameters]):
        """
        Inicializa el simulador.
        
        Args:
            parametros_gbm: Diccionario con parámetros GBM por acción
        """
        self.parametros_gbm = parametros_gbm
        self.tickers = list(parametros_gbm.keys())
    
    def simular_portafolio(
        self,
        pesos_portafolio: Dict[str, float],
        presupuesto_total: float,
        horizonte_años: float = 1.0,
        n_simulaciones: int = 10000,
        pasos_tiempo: int = 252,
        semilla: Optional[int] = None
    ) -> Tuple[float, float, np.ndarray]:
        """
        Realiza simulaciones de Monte Carlo para un portafolio.
        
        Args:
            pesos_portafolio: Diccionario {ticker: peso} donde pesos suman 1.0
            presupuesto_total: Presupuesto total a invertir
            horizonte_años: Horizonte de tiempo en años
            n_simulaciones: Número de simulaciones Monte Carlo
            pasos_tiempo: Número de pasos de tiempo (252 = días de trading/año)
            semilla: Semilla para reproducibilidad
            
        Returns:
            Tupla (esperanza, varianza, valores_finales)
            - esperanza: Valor esperado del portafolio al final
            - varianza: Varianza del valor del portafolio
            - valores_finales: Array con todos los valores finales simulados
        """
        # Validar pesos
        self._validar_pesos(pesos_portafolio)
        
        if semilla is not None:
            np.random.seed(semilla)
        
        print("\n" + "="*80)
        print("SIMULACIÓN DE MONTE CARLO - PORTAFOLIO")
        print("="*80)
        print(f"\nCONFIGURACIÓN DEL PORTAFOLIO:")
        print(f"Presupuesto total:          ${presupuesto_total:,.2f}")
        print(f"Horizonte de tiempo:        {horizonte_años} año(s)")
        print(f"Número de simulaciones:     {n_simulaciones:,}")
        print(f"Pasos de tiempo:            {pasos_tiempo}")
        
        print(f"\n📊 COMPOSICIÓN DEL PORTAFOLIO:")
        inversion_inicial = 0
        for ticker, peso in pesos_portafolio.items():
            inversion = presupuesto_total * peso
            params = self.parametros_gbm[ticker]
            n_acciones = inversion / params.S0
            print(f"  {ticker:6s}: {peso*100:5.2f}% → ${inversion:10,.2f} "
                  f"({n_acciones:.2f} acciones @ ${params.S0:.2f})")
            inversion_inicial += inversion
        
        # Calcular número de acciones a comprar por cada ticker
        acciones_por_ticker = {}
        for ticker, peso in pesos_portafolio.items():
            inversion = presupuesto_total * peso
            precio_inicial = self.parametros_gbm[ticker].S0
            acciones_por_ticker[ticker] = inversion / precio_inicial
        
        # Simular precios finales para cada acción
        valores_finales_portafolio = np.zeros(n_simulaciones)
        
        dt = horizonte_años / pasos_tiempo  # Tamaño del paso temporal
        
        print(f"\nEjecutando {n_simulaciones:,} simulaciones...")
        
        for ticker, n_acciones in acciones_por_ticker.items():
            params = self.parametros_gbm[ticker]
            
            # Simular trayectorias para esta acción
            precios_finales = self._simular_gbm(
                S0=params.S0,
                mu=params.mu,
                sigma=params.sigma,
                T=horizonte_años,
                pasos=pasos_tiempo,
                n_sim=n_simulaciones
            )
            
            # Contribución de esta acción al valor del portafolio
            valores_finales_portafolio += n_acciones * precios_finales
        
        # Calcular estadísticas
        esperanza = np.mean(valores_finales_portafolio)
        varianza = np.var(valores_finales_portafolio)
        desv_std = np.sqrt(varianza)
        
        # Retornos
        retornos = (valores_finales_portafolio - presupuesto_total) / presupuesto_total
        retorno_esperado = np.mean(retornos)
        
        print(f"\n✓ Simulación completada!")
        print(f"\n" + "="*80)
        print("RESULTADOS DE LA SIMULACIÓN")
        print("="*80)
        print(f"\n💰 VALOR DEL PORTAFOLIO (al final del horizonte):")
        print(f"Valor esperado (E[V]):      ${esperanza:,.2f}")
        print(f"Desviación estándar:        ${desv_std:,.2f}")
        print(f"Varianza:                   ${varianza:,.2f}")
        
        print(f"\nRETORNOS:")
        print(f"Retorno esperado:           {retorno_esperado*100:.2f}%")
        print(f"Ganancia/Pérdida esperada:  ${esperanza - presupuesto_total:,.2f}")
        
        print(f"\nDISTRIBUCIÓN DE RESULTADOS:")
        percentiles = [5, 25, 50, 75, 95]
        for p in percentiles:
            valor = np.percentile(valores_finales_portafolio, p)
            print(f"Percentil {p:2d}:               ${valor:,.2f}")
        
        print(f"\nPROBABILIDADES:")
        prob_ganancia = np.mean(valores_finales_portafolio > presupuesto_total) * 100
        prob_perdida = 100 - prob_ganancia
        print(f"P(Ganancia):                {prob_ganancia:.2f}%")
        print(f"P(Pérdida):                 {prob_perdida:.2f}%")
        
        # Value at Risk (VaR) y Conditional VaR (CVaR)
        var_95 = presupuesto_total - np.percentile(valores_finales_portafolio, 5)
        cvar_95 = presupuesto_total - np.mean(
            valores_finales_portafolio[valores_finales_portafolio <= np.percentile(valores_finales_portafolio, 5)]
        )
        print(f"\nMÉTRICAS DE RIESGO:")
        print(f"VaR (95%):                  ${var_95:,.2f}")
        print(f"CVaR (95%):                 ${cvar_95:,.2f}")
        print(f"(Máxima pérdida esperada en el 5% peor de los casos)")
        
        print("="*80 + "\n")
        
        return esperanza, varianza, valores_finales_portafolio
    
    def _simular_gbm(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        pasos: int,
        n_sim: int
    ) -> np.ndarray:
        """
        Simula trayectorias de GBM y retorna precios finales.
        
        Usa la solución exacta del GBM:
        S_T = S_0 * exp((μ - σ²/2)*T + σ*√T*Z)
        donde Z ~ N(0,1)
        """
        dt = T / pasos
        
        # Generar variables aleatorias normales
        Z = np.random.standard_normal((n_sim, pasos))
        
        # Calcular incrementos
        incrementos = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        
        # Sumar incrementos para obtener log(S_T/S_0)
        log_retornos = np.sum(incrementos, axis=1)
        
        # Calcular precios finales
        precios_finales = S0 * np.exp(log_retornos)
        
        return precios_finales
    
    def _validar_pesos(self, pesos: Dict[str, float]):
        """Valida que los pesos del portafolio sean correctos."""
        # Verificar que todos los tickers existen
        for ticker in pesos.keys():
            if ticker not in self.parametros_gbm:
                raise ValueError(f"Ticker '{ticker}' no encontrado en parámetros GBM")
        
        # Verificar que los pesos sumen aproximadamente 1
        suma_pesos = sum(pesos.values())
        if not np.isclose(suma_pesos, 1.0, atol=1e-6):
            raise ValueError(
                f"Los pesos deben sumar 1.0 (suma actual: {suma_pesos:.6f})"
            )
        
        # Verificar que todos los pesos sean no negativos
        for ticker, peso in pesos.items():
            if peso < 0:
                raise ValueError(f"El peso para '{ticker}' no puede ser negativo: {peso}")
    
    def visualizar_simulaciones(
        self,
        valores_finales: np.ndarray,
        presupuesto_inicial: float,
        nombre_archivo: Optional[str] = None
    ):
        """
        Visualiza los resultados de las simulaciones.
        
        Args:
            valores_finales: Array con valores finales del portafolio
            presupuesto_inicial: Valor inicial del portafolio
            nombre_archivo: Nombre del archivo para guardar la figura
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Resultados de Simulación Monte Carlo', fontsize=16, fontweight='bold')
        
        # 1. Histograma de valores finales
        ax1 = axes[0, 0]
        ax1.hist(valores_finales, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(presupuesto_inicial, color='red', linestyle='--', 
                    linewidth=2, label=f'Inversión inicial: ${presupuesto_inicial:,.0f}')
        ax1.axvline(np.mean(valores_finales), color='green', linestyle='--', 
                    linewidth=2, label=f'Valor esperado: ${np.mean(valores_finales):,.0f}')
        ax1.set_xlabel('Valor Final del Portafolio ($)')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Valores Finales')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Histograma de retornos
        ax2 = axes[0, 1]
        retornos = (valores_finales - presupuesto_inicial) / presupuesto_inicial * 100
        ax2.hist(retornos, bins=50, alpha=0.7, color='coral', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Retorno = 0%')
        ax2.axvline(np.mean(retornos), color='green', linestyle='--', 
                    linewidth=2, label=f'Retorno esperado: {np.mean(retornos):.1f}%')
        ax2.set_xlabel('Retorno (%)')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Retornos')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Box plot
        ax3 = axes[1, 0]
        bp = ax3.boxplot([valores_finales], vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax3.axhline(presupuesto_inicial, color='red', linestyle='--', 
                    linewidth=2, label='Inversión inicial')
        ax3.set_ylabel('Valor del Portafolio ($)')
        ax3.set_title('Box Plot de Valores Finales')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Percentiles
        ax4 = axes[1, 1]
        percentiles = range(0, 101, 5)
        valores_percentiles = [np.percentile(valores_finales, p) for p in percentiles]
        ax4.plot(percentiles, valores_percentiles, marker='o', linewidth=2, markersize=4)
        ax4.axhline(presupuesto_inicial, color='red', linestyle='--', 
                    linewidth=2, label='Inversión inicial')
        ax4.fill_between(percentiles, presupuesto_inicial, valores_percentiles, 
                         where=np.array(valores_percentiles) >= presupuesto_inicial,
                         alpha=0.3, color='green', label='Ganancia')
        ax4.fill_between(percentiles, presupuesto_inicial, valores_percentiles, 
                         where=np.array(valores_percentiles) < presupuesto_inicial,
                         alpha=0.3, color='red', label='Pérdida')
        ax4.set_xlabel('Percentil')
        ax4.set_ylabel('Valor del Portafolio ($)')
        ax4.set_title('Curva de Percentiles')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if nombre_archivo:
            plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
            print(f"✓ Visualización guardada en: {nombre_archivo}")
        
        plt.show()


def ejemplo_completo():
    """
    Ejemplo completo del flujo de trabajo.
    """
    print("\n" + "="*80)
    print("EJEMPLO COMPLETO: AJUSTE GBM + SIMULACIÓN MONTE CARLO")
    print("="*80)
    
    # Paso 1: Ajustar modelos GBM desde un archivo CSV
    print("\nPaso 1: Cargar datos y ajustar modelos GBM")
    gbm = GeometricBrownianMotion(granularidad='dia')
    
    # Asumiendo que existe un archivo de ejemplo
    # parametros = gbm.ajustar_desde_csv('acciones_diarias.csv')
    
    # Para este ejemplo, crearemos parámetros ficticios
    parametros_ejemplo = {
        'AAPL': GBMParameters('AAPL', mu=0.15, sigma=0.25, S0=180.0),
        'GOOGL': GBMParameters('GOOGL', mu=0.12, sigma=0.22, S0=140.0),
        'MSFT': GBMParameters('MSFT', mu=0.18, sigma=0.28, S0=370.0),
    }
    
    print("\n✓ Modelos ajustados (ejemplo):")
    for ticker, params in parametros_ejemplo.items():
        print(f"  {params}")
    
    # Paso 2: Configurar simulación Monte Carlo
    print("\nPaso 2: Configurar simulación Monte Carlo")
    simulador = MonteCarloSimulator(parametros_ejemplo)
    
    # Definir portafolio
    pesos_portafolio = {
        'AAPL': 0.4,   # 40% en Apple
        'GOOGL': 0.3,  # 30% en Google
        'MSFT': 0.3    # 30% en Microsoft
    }
    
    presupuesto = 100000  # $100,000
    
    # Paso 3: Ejecutar simulación
    print("\nPaso 3: Ejecutar simulación Monte Carlo")
    esperanza, varianza, valores_finales = simulador.simular_portafolio(
        pesos_portafolio=pesos_portafolio,
        presupuesto_total=presupuesto,
        horizonte_años=1.0,
        n_simulaciones=10000,
        semilla=42  # Para reproducibilidad
    )
    
    # Paso 4: Visualizar resultados
    print("\nPaso 4: Visualizar resultados")
    simulador.visualizar_simulaciones(
        valores_finales=valores_finales,
        presupuesto_inicial=presupuesto,
        nombre_archivo='simulacion_montecarlo.png'
    )


if __name__ == "__main__":
    ejemplo_completo()
