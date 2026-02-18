# Optimización de Portafolios con Restricciones

**Curso:** Optimización Numérica I — ITAM
**Autores:** Darien Mandujano, Rodrigo Pagola, Elmer Ortega
**Referencia principal:** Nocedal & Wright, *Numerical Optimization* (2nd ed.)

---

## Descripción

Este proyecto implementa un pipeline completo para la **optimización de portafolios de renta variable** bajo restricciones regulatorias. El problema central es un programa no-lineal con restricciones mixtas:

```
min   f(w) = -E[R(w)] + γ · Var[R(w)]        (γ = 20)
 w

s.t.  Σ wᵢ = 1                                (igualdad: pesos suman 1)
      wᵢ ≥ 0         i = 1, …, 10            (desigualdad: no short-selling)
      wᵢ ≤ 0.30      i = 1, …, 10            (desigualdad: límite por activo)
```

donde `E[R(w)]` y `Var[R(w)]` se evalúan numéricamente vía simulación Monte Carlo, haciendo que la función objetivo no tenga forma analítica cerrada.

### Cobertura del temario

| Tema (Nocedal & Wright) | Implementación |
|---|---|
| Tema 1 — KKT y condiciones de optimalidad | Criterio de parada `‖∇L‖ < ε` en SLSQP |
| Tema 2 — Programación cuadrática (QP) | Subproblema QP interno de SLSQP (Alg. 18.3) |
| Tema 3 — SQP y búsqueda de línea | SLSQP con función de mérito ℓ₁ |
| Tema 4 — Lagrangiano aumentado | `L_A(w,λ;ρ) = f(w) + λh(w) + (ρ/2)h(w)²` |

---

## Pipeline

```
Datos Yahoo Finance (semanal, 2024–2025)
            ↓
    Ajuste GARCH(1,1) por activo
            ↓
  8,000 trayectorias × 13 semanas × 10 activos
            ↓
  Sampleo estratificado: 5 casos × 1,600 trayectorias
            ↓
    Optimización NLP con SLSQP (scipy)
            ↓
  Comparación out-of-sample vs retornos reales
            ↓
        Visualizaciones
```

### Activos (10)

`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `META`, `NVDA`, `JPM`, `XOM`, `JNJ`, `KO`

---

## Estructura del repositorio

```
OptimizacionPortafolioEstocastica/
│
├── README.md
│
└── src/
    ├── simulacion_completa_8000.py      # Pipeline principal (ejecutar primero)
    ├── visualizacion_por_sampleo.py     # Postprocesamiento y figuras
    │
    ├── stock_data/
    │   └── RetrieveData.py              # Descarga de datos vía yfinance
    │
    └── montecarlo_simulation/
        └── montecarlo.py                # GBM + Monte Carlo (módulo auxiliar)
```

> **Nota:** `stock_data/` y `montecarlo_simulation/` son módulos auxiliares del prototipo
> inicial con GBM de volatilidad constante. El pipeline de producción del proyecto
> (`simulacion_completa_8000.py`) usa GARCH(1,1) directamente con la librería `arch`.

---

## Instalación

```bash
pip install yfinance pandas numpy matplotlib scipy arch seaborn
```

| Librería | Uso |
|---|---|
| `arch` | Ajuste GARCH(1,1) |
| `scipy.optimize` | SLSQP (optimizador NLP) |
| `yfinance` | Descarga de precios históricos |
| `numpy` / `pandas` | Álgebra y manipulación de datos |
| `matplotlib` / `seaborn` | Visualizaciones |

---

## Uso

### 1. Ejecutar el pipeline completo

```bash
cd src
python simulacion_completa_8000.py
```

Genera:
- `simulaciones_8000_trayectorias.npy` — tensor de retornos simulados `[8000, 10, 13]`
- `resultados_5_casos_optimizados.csv` — pesos óptimos y métricas por caso
- `analisis_completo_8000_trayectorias.png` — dashboard principal

### 2. Generar visualizaciones detalladas

```bash
cd src
python visualizacion_por_sampleo.py
```

Requiere que existan los archivos generados en el paso anterior. Produce 5 figuras:

| Figura | Contenido |
|---|---|
| `analisis_por_sampleo_detallado.png` | Pesos, métricas de riesgo y comparación simulado vs real por caso |
| `analisis_convergencia_samples.png` | Consistencia entre casos, heatmap de pesos, frontera eficiente |
| `composicion_portafolios_samples.png` | Gráficos de torta e índice de Herfindahl |
| `distribuciones_esperado_vs_real.png` | Distribuciones de retorno por caso (requiere `.npy`) |
| `analisis_completo_integrado.png` | Dashboard integrado completo |

---

## Metodología

### Modelo de volatilidad: GARCH(1,1)

Para cada activo `i`, se ajusta:

```
r_{i,t} = μᵢ + σ_{i,t} εₜ,     εₜ ~ N(0,1)

σ²_{i,t} = ωᵢ + αᵢ ε²_{t-1} + βᵢ σ²_{i,t-1}
```

donde `ωᵢ > 0`, `αᵢ ≥ 0`, `βᵢ ≥ 0` se estiman por máxima verosimilitud.
Esto captura el **clustering de volatilidad** (períodos calmos seguidos de períodos turbulentos)
que el GBM de volatilidad constante ignora.

### Simulación Monte Carlo

Se generan 8,000 trayectorias completas de 13 semanas para los 10 activos simultáneamente.
El retorno del portafolio en cada simulación `s` es:

```
R(w, s) = Σᵢ wᵢ · (exp(Σₜ r_{i,t,s}) - 1)
```

Las 8,000 trayectorias se dividen en 5 muestras disjuntas de 1,600 cada una
(*sampleo estratificado*), permitiendo evaluar la estabilidad de la solución óptima.

### Optimización: SLSQP

Se usa `scipy.optimize.minimize(method='SLSQP')`, que implementa el Algoritmo 18.3
de Nocedal & Wright. En cada iteración `k`:

1. Resolver el subproblema QP local:
   ```
   min   ∇f(wₖ)ᵀ p + ½ pᵀ Bₖ p
    p
   s.a.  eᵀ(wₖ + p) = 1,   0 ≤ wₖ + p ≤ 0.30 e
   ```
2. Búsqueda de línea con función de mérito ℓ₁: `φ₁(w; σ) = f(w) + σ|h(w)|`
3. Actualización quasi-Newton de la Hessiana: `Bₖ₊₁` con BFGS de Powell

Convergencia típica: **5–10 iteraciones** por caso.

### Métricas de riesgo

| Métrica | Definición |
|---|---|
| `E[R]` | Media de los retornos simulados del portafolio |
| `σ` | Desviación estándar de los retornos simulados |
| VaR 95% | Percentil 5 de la distribución de retornos |
| Sharpe | `E[R] / σ` (tasa libre de riesgo = 0) |
| Índice Herfindahl | `Σ wᵢ²` — mide concentración (0 = diversificado, 1 = concentrado) |

---

## Resultados

### Pesos óptimos (promedio entre los 5 casos)

Los 5 casos convergen a soluciones similares (consistencia con la convexidad aproximada
para `γ = 20`). Ningún activo alcanza el límite del 30% (`μ⁺ = 0` en KKT).

| Métrica | Valor |
|---|---|
| Retorno esperado | 8.3–8.6% (horizonte 3 meses) |
| Volatilidad | ~4.2% |
| Sharpe ratio | ~2.0 |
| Índice Herfindahl | ~0.11 |

### Validación out-of-sample

| Caso | Esperado | Real | Error |
|---|---|---|---|
| Caso 1 | 8.38% | 7.76% | −0.62% |
| Caso 2 | 8.53% | 7.97% | −0.56% |
| Caso 3 | 8.29% | 8.07% | −0.21% |
| Caso 4 | 8.62% | 8.36% | −0.26% |
| Caso 5 | 8.45% | 7.58% | −0.87% |
| **Media** | **8.45%** | **7.95%** | **−0.50%** |

Error relativo medio: **6%** — aceptable para un horizonte de 3 meses.

---

## Limitaciones

- **GARCH asume innovaciones normales**: no captura completamente colas pesadas ni saltos.
- **f(w) sin forma analítica**: el gradiente se aproxima numéricamente (diferencias finitas en SLSQP), lo que puede ser sensible al ruido estocástico de la simulación.
- **Punto inicial fijo**: los 5 casos parten del mismo `w₀`, por lo que la diversidad entre muestras refleja solo la variabilidad estadística del Monte Carlo, no múltiples cuencas de atracción.
- **Sin correlaciones entre activos**: la simulación GARCH es independiente por activo; no modela copulas ni volatilidad estocástica multivariada.

---

## Referencias

- Nocedal, J. & Wright, S. (2006). *Numerical Optimization* (2nd ed.). Springer. — Algoritmos 16.3, 18.3
- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77–91.
- Engle, R. (1982). Autoregressive Conditional Heteroscedasticity. *Econometrica*, 50(4).
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroscedasticity. *Journal of Econometrics*, 31(3).
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
