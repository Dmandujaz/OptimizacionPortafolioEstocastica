import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List, Optional
import os


class StockDataRetriever:
    """
    Clase para obtener datos históricos de acciones usando yfinance.
    """
    
    VALID_INTERVALS = {
        'hora': '1h',
        'dia': '1d',
        'mes': '1mo',
        'semana': '1wk',
        'minuto': '1m',
        '5minutos': '5m',
        '15minutos': '15m',
        '30minutos': '30m',
        '90minutos': '90m'
    }
    
    def __init__(self):
        """Inicializa el recuperador de datos de acciones."""
        pass
    
    def obtener_datos(
        self,
        acciones: List[str],
        fecha_inicio: str,
        fecha_fin: str,
        granularidad: str = 'dia',
        guardar_csv: bool = True,
        nombre_archivo: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Obtiene datos históricos de una lista de acciones.
        
        Args:
            acciones: Lista de símbolos de acciones (ej: ['AAPL', 'GOOGL', 'MSFT'])
            fecha_inicio: Fecha de inicio en formato 'YYYY-MM-DD'
            fecha_fin: Fecha de fin en formato 'YYYY-MM-DD'
            granularidad: Granularidad de los datos ('dia', 'hora', 'mes', 'semana')
            guardar_csv: Si True, guarda los datos en un archivo CSV
            nombre_archivo: Nombre personalizado para el archivo CSV
            
        Returns:
            DataFrame con los datos de las acciones
        """
        # Validar granularidad
        if granularidad not in self.VALID_INTERVALS:
            raise ValueError(
                f"Granularidad '{granularidad}' no válida. "
                f"Opciones válidas: {list(self.VALID_INTERVALS.keys())}"
            )
        
        interval = self.VALID_INTERVALS[granularidad]
        
        # Validar fechas
        try:
            datetime.strptime(fecha_inicio, '%Y-%m-%d')
            datetime.strptime(fecha_fin, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Las fechas deben estar en formato 'YYYY-MM-DD'")
        
        # Validar que hay acciones
        if not acciones or len(acciones) == 0:
            raise ValueError("Debe proporcionar al menos una acción")
        
        print(f"Descargando datos para: {', '.join(acciones)}")
        print(f"Periodo: {fecha_inicio} a {fecha_fin}")
        print(f"Granularidad: {granularidad} ({interval})")
        
        # Descargar datos
        try:
            datos = yf.download(
                tickers=' '.join(acciones),
                start=fecha_inicio,
                end=fecha_fin,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                progress=True
            )
            
            if datos.empty:
                raise ValueError("No se obtuvieron datos. Verifique los símbolos y fechas.")
            
            # Reorganizar datos si hay múltiples acciones
            if len(acciones) > 1:
                # Aplanar el MultiIndex para mejor legibilidad
                datos_procesados = self._procesar_multiples_acciones(datos, acciones)
            else:
                datos_procesados = datos
                datos_procesados.columns = [f"{acciones[0]}_{col}" for col in datos_procesados.columns]
            
            print(f"\n✓ Datos obtenidos exitosamente: {len(datos_procesados)} registros")
            
            # Guardar CSV si se solicita
            if guardar_csv:
                if nombre_archivo is None:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    nombre_archivo = f"stock_data_{timestamp}.csv"
                
                # Asegurar que termine en .csv
                if not nombre_archivo.endswith('.csv'):
                    nombre_archivo += '.csv'
                
                datos_procesados.to_csv(nombre_archivo)
                print(f"✓ Datos guardados en: {nombre_archivo}")
            
            return datos_procesados
            
        except Exception as e:
            raise Exception(f"Error al descargar datos: {str(e)}")
    
    def _procesar_multiples_acciones(
        self,
        datos: pd.DataFrame,
        acciones: List[str]
    ) -> pd.DataFrame:
        """
        Procesa los datos cuando hay múltiples acciones.
        
        Args:
            datos: DataFrame con MultiIndex
            acciones: Lista de símbolos de acciones
            
        Returns:
            DataFrame procesado con columnas aplanadas
        """
        # Crear un nuevo DataFrame con columnas aplanadas
        columnas_nuevas = {}
        
        for accion in acciones:
            if accion in datos.columns.get_level_values(0):
                for col in datos[accion].columns:
                    columnas_nuevas[f"{accion}_{col}"] = datos[accion][col]
        
        return pd.DataFrame(columnas_nuevas, index=datos.index)


def ejemplo_uso():
    """
    Función de ejemplo que muestra cómo usar la clase StockDataRetriever.
    """
    # Crear instancia del recuperador
    retriever = StockDataRetriever()
    
    # Ejemplo 1: Datos diarios (predeterminado)
    print("\n=== Ejemplo 1: Datos diarios ===")
    acciones = ['AAPL', 'GOOGL', 'MSFT']
    df_diario = retriever.obtener_datos(
        acciones=acciones,
        fecha_inicio='2024-01-01',
        fecha_fin='2024-12-31',
        granularidad='dia',
        nombre_archivo='acciones_diarias.csv'
    )
    print(f"Dimensiones: {df_diario.shape}")
    print(df_diario.head())
    
    # Ejemplo 2: Datos mensuales
    print("\n=== Ejemplo 2: Datos mensuales ===")
    df_mensual = retriever.obtener_datos(
        acciones=['TSLA', 'NVDA'],
        fecha_inicio='2023-01-01',
        fecha_fin='2024-12-31',
        granularidad='mes',
        nombre_archivo='acciones_mensuales.csv'
    )
    print(f"Dimensiones: {df_mensual.shape}")
    print(df_mensual.head())
    
    # Ejemplo 3: Datos por hora (últimos 7 días)
    print("\n=== Ejemplo 3: Datos por hora ===")
    df_hora = retriever.obtener_datos(
        acciones=['AAPL'],
        fecha_inicio='2024-11-01',
        fecha_fin='2024-11-08',
        granularidad='hora',
        nombre_archivo='acciones_horarias.csv'
    )
    print(f"Dimensiones: {df_hora.shape}")
    print(df_hora.head())


if __name__ == "__main__":
    # Ejecutar ejemplos
    ejemplo_uso()
    
    # O usar interactivamente:
    print("\n=== Uso Personalizado ===")
    retriever = StockDataRetriever()
    
    # Personaliza estos valores
    mis_acciones = ['AAPL', 'MSFT']  # Lista de acciones
    inicio = '2024-01-01'             # Fecha de inicio
    fin = '2024-11-11'                # Fecha de fin
    granularidad = 'dia'              # 'dia', 'hora', 'mes', etc.
    
    df = retriever.obtener_datos(
        acciones=mis_acciones,
        fecha_inicio=inicio,
        fecha_fin=fin,
        granularidad=granularidad,
        nombre_archivo='mis_datos_stock.csv'
    )
    
    print(f"\n✓ Datos obtenidos: {df.shape[0]} filas, {df.shape[1]} columnas")
    print("\nPrimeras filas:")
    print(df.head())
    print("\nÚltimas filas:")
    print(df.tail())
