# Informe Final: Optimización Estocástica de Portafolios mediante SQP con Modelado GARCH

**Curso:** Optimización Numérica I — ITAM
**Autores:** Darien Mandujano, Rodrigo Pagola, Elmer Ortega
**Referencia teórica:** Nocedal & Wright, *Numerical Optimization* (2nd ed.)

---

## 1. Introducción

Este proyecto implementa un sistema completo de optimización estocástica para asignación óptima de activos bajo incertidumbre. El pipeline integra:

1. Modelado econométrico GARCH(1,1)
2. Simulación Monte Carlo (8,000 trayectorias)
3. Formulación como Programa No Lineal (NLP)
4. Resolución mediante Sequential Quadratic Programming (SLSQP)
5. Validación mediante condiciones de Karush–Kuhn–Tucker (KKT)
6. Análisis paramétrico respecto al coeficiente de aversión al riesgo γ

El objetivo es combinar teoría rigurosa de optimización numérica con modelado financiero realista.

---

## 2. Formulación Matemática del Problema

### 2.1 Función Objetivo

Se minimiza:

```
f(w) = -E[R(w)] + γ · Var[R(w)]
```

donde:

- `w ∈ ℝ¹⁰`: vector de pesos del portafolio
- `γ = 20`: coeficiente de aversión al riesgo
- `E[R(w)]` y `Var[R(w)]` se estiman vía simulación Monte Carlo sobre 1,600 trayectorias GARCH por llamada al optimizador

Bajo el marco clásico media-varianza (Markowitz), si la matriz de covarianza es semidefinida positiva, el problema es convexo para todo `γ ≥ 0`.

### 2.2 Restricciones

**Restricción de presupuesto (igualdad):**
```
Σᵢ wᵢ = 1
```

**Cotas regulatorias (desigualdad):**
```
0 ≤ wᵢ ≤ 0.30,    i = 1, …, 10
```

El problema completo en forma estándar:

```
min   f(w)
 w
s.t.  cₑ(w) = 0          (igualdad: presupuesto)
      cᵢ(w) ≥ 0          (desigualdad: cotas)
```

---

## 3. Condiciones KKT y Multiplicadores de Lagrange

Las condiciones KKT garantizan optimalidad — necesaria siempre, suficiente bajo convexidad.

### 3.1 Condición de Estacionariedad

```
∇f(w*) = λ* · 1 + Σⱼ μⱼ* · ∇gⱼ(w*)
```

El gradiente del objetivo (riesgo-retorno) se equilibra exactamente con los gradientes de las restricciones activas ponderados por los multiplicadores. El criterio de parada del algoritmo SLSQP es:

```
‖∇L‖ < 1×10⁻⁸
```

lo que implica que el sistema KKT se satisface aproximadamente al nivel de tolerancia numérica.

### 3.2 Interpretación Económica de los Multiplicadores

**Restricción de presupuesto** `Σwᵢ = 1`:

El multiplicador `λ_pres` mide la sensibilidad del óptimo ante relajaciones del presupuesto:

```
df*/dε = -λ_pres
```

Interpretación: valor marginal de una unidad adicional de capital.

**Restricciones de techo** `wᵢ ≤ 0.30`:

| Caso | Significado |
|---|---|
| `μᵢ = 0` | Restricción inactiva: el activo no alcanza el límite de forma natural |
| `μᵢ > 0` | Restricción activa: el activo tendería a recibir más peso si no existiera el límite |

**Resultado con γ = 20:** ningún activo alcanza el 30%, por lo que `μᵢ = 0` para todos. La diversificación es endógena — no forzada por las restricciones regulatorias.

### 3.3 Holgura Complementaria

```
μᵢ · gᵢ(w*) = 0    para todo i
```

Si una cota está inactiva (`wᵢ < 0.30`), su multiplicador es cero. Si está activa (`wᵢ = 0.30`), el multiplicador puede ser positivo y mide el costo de oportunidad de esa restricción.

---

## 4. Impacto del Parámetro de Aversión al Riesgo γ

### 4.1 Régimen γ Alto (γ = 20, configuración actual)

- Predomina la penalización por varianza sobre la maximización de retorno
- El óptimo se ubica en el interior del poliedro factible
- Restricciones superiores inactivas (`μᵢ = 0`)
- Alta diversificación: Índice Herfindahl ≈ 0.11
- Sharpe ≈ 2.0

### 4.2 Régimen γ Bajo

Al reducir γ:

- La función objetivo se aproxima a maximizar `E[R(w)]` sin penalización
- El óptimo migra hacia vértices del poliedro factible
- Las cotas superiores se activan (`μᵢ > 0`) para activos de mayor retorno esperado
- Mayor concentración, mayor VaR

### 4.3 Caso Límite γ → 0

- El problema degenera a un programa lineal: `max Σᵢ wᵢ · E[Rᵢ]`
- La solución ocurre en vértices del poliedro (posiblemente no única)
- Todo el capital se concentra en el activo de mayor retorno esperado, hasta la cota del 30%

### 4.4 Resumen del Efecto de γ

| γ | Estructura del óptimo | Restricciones activas | Diversificación |
|---|---|---|---|
| → 0 | Vértice | `wᵢ = 0.30` para top activos | Mínima |
| Moderado | Interior/frontera | Algunas activas | Media |
| 20 (actual) | Interior | Ninguna | Alta (H ≈ 0.11) |
| → ∞ | Pesos iguales `1/n` | Ninguna | Máxima |

---

## 5. Elección de SLSQP sobre Métodos de Punto Interior

Se seleccionó SLSQP (Sequential Least Squares Programming) por las siguientes razones:

| Criterio | SLSQP | Punto Interior (Cap. 19 N&W) |
|---|---|---|
| Escala del problema | n=10 (pequeño) | Ventaja en n >> 100 |
| Función objetivo | Costosa (Monte Carlo) | Requiere muchas evaluaciones |
| Conjunto activo | Identificación eficiente | Mantiene interior estrictamente |
| Hessiana | Aproximación BFGS | Factorización completa |
| Convergencia | Tipo Newton en KKT | Barrera logarítmica |

Para `n = 10` con restricciones lineales y función objetivo no-lineal costosa, SLSQP es el método apropiado. Los métodos de punto interior son más ventajosos en problemas de gran escala con estructura dispersa.

El algoritmo implementado sigue el Algoritmo 18.3 de Nocedal & Wright:

1. En cada iteración `k`, resolver el subproblema QP:
   ```
   min   ∇f(wₖ)ᵀ p + ½ pᵀ Bₖ p
    p
   s.a.  eᵀ(wₖ + p) = 1
         0 ≤ wₖ + p ≤ 0.30 e
   ```
2. Búsqueda de línea con función de mérito ℓ₁: `φ₁(w; σ) = f(w) + σ|h(w)|`
3. Actualización quasi-Newton: `Bₖ₊₁` mediante BFGS de Powell

Convergencia típica: **5–10 iteraciones** por caso.

---

## 6. Modelo Estocástico: GARCH vs GBM

### 6.1 Limitación del GBM (Movimiento Browniano Geométrico)

```
dSₜ = μ Sₜ dt + σ Sₜ dWₜ
```

- Volatilidad `σ` constante en todo el horizonte
- No captura clustering de volatilidad
- Homocedasticidad: la varianza de los retornos es constante
- Consecuencia: subestima el riesgo en períodos de alta volatilidad

### 6.2 Ventaja del GARCH(1,1)

```
σ²ₜ = ω + α ε²ₜ₋₁ + β σ²ₜ₋₁
```

- `ω > 0`: nivel base de volatilidad
- `α ≥ 0`: sensibilidad a shocks recientes (efecto ARCH)
- `β ≥ 0`: persistencia de la volatilidad pasada
- `α + β < 1`: condición de estacionariedad covariante

Propiedades relevantes:
- Varianza condicional dependiente del historial
- Captura heterocedasticidad condicional
- Reproduce clustering de volatilidad (períodos turbulentos se agrupan)
- Mejora la estimación de `Var[R(w)]` y por tanto la calidad del óptimo

### 6.3 Parámetros estimados por MLE (librería `arch`)

Para cada activo se ajustan `{μ, ω, α, β}` sobre retornos logarítmicos semanales. La volatilidad condicional final `σ_T` del período de entrenamiento se usa como semilla de la simulación forward.

---

## 7. Extensión: Introducción de Correlaciones entre Activos

### 7.1 Supuesto actual (independencia)

El modelo simula cada activo de forma independiente. La varianza del portafolio se estima como:

```
Var(Rₚ) ≈ Σᵢ wᵢ² σᵢ²    (sin términos cruzados)
```

### 7.2 Modelo con correlaciones

Con correlaciones:

```
Var(Rₚ) = Σᵢ wᵢ² σᵢ² + Σᵢ≠ⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ
```

**Consecuencias si `ρᵢⱼ > 0` (activos correlacionados positivamente):**

- La varianza real del portafolio es mayor que la estimada
- La diversificación es menor de lo que el modelo supone
- El Sharpe real disminuye respecto al simulado
- El VaR real empeora

**Impacto sobre la optimización:**

- La función objetivo `f(w)` subestima `Var[R(w)]`
- Los pesos óptimos pueden estar mal calibrados (exceso de concentración en activos correlacionados)
- Las condiciones KKT se satisfacen para el problema mal especificado

### 7.3 Implementación técnica posible

1. Estimar matriz de correlación `Σ` sobre retornos históricos
2. Descomposición de Cholesky: `Σ = L Lᵀ`
3. Generar shocks incorrelados `z ~ N(0, I)` y transformar: `ε = L z`
4. Extensión avanzada: DCC-GARCH (Dynamic Conditional Correlation)

---

## 8. Resultados Principales

### 8.1 Con γ = 20 (configuración implementada)

| Métrica | Valor |
|---|---|
| Retorno esperado | 8.3–8.6% (3 meses) |
| Volatilidad | ~4.2% |
| Sharpe ratio | ~2.0 |
| Índice Herfindahl | ~0.11 |
| Restricciones activas | Ninguna (`μᵢ = 0` para todo i) |
| Convergencia | 5–10 iteraciones |

**Validación out-of-sample:**

| Caso | Esperado | Real | Error |
|---|---|---|---|
| Caso 1 | 8.38% | 7.76% | −0.62% |
| Caso 2 | 8.53% | 7.97% | −0.56% |
| Caso 3 | 8.29% | 8.07% | −0.21% |
| Caso 4 | 8.62% | 8.36% | −0.26% |
| Caso 5 | 8.45% | 7.58% | −0.87% |
| **Media** | **8.45%** | **7.95%** | **−0.50%** |

Error relativo medio: **6%** — aceptable para horizonte de 3 meses.

### 8.2 Efecto esperado al reducir γ

| Consecuencia | Mecanismo |
|---|---|
| Mayor concentración | Dominan activos de alto retorno esperado |
| Cotas activas | `μᵢ > 0` para activos con `wᵢ = 0.30` |
| Mayor volatilidad | `Var[R(w)]` penalizada con menor peso |
| Mayor VaR | Distribución de retornos más dispersa |
| Menor Herfindahl inverso | Menor número efectivo de activos |

---

## 9. Conclusiones

1. **La formulación como NLP convexo** (bajo γ grande) permite uso eficiente de SQP con convergencia garantizada al óptimo global.

2. **Las condiciones KKT certifican optimalidad**: el hecho de que `μᵢ = 0` para todos los activos confirma que la diversificación es una consecuencia natural del problema, no una imposición de las restricciones regulatorias.

3. **El parámetro γ controla la estructura del conjunto activo**: γ alto → interior → restricciones inactivas; γ bajo → frontera/vértices → restricciones activas.

4. **GARCH mejora estructuralmente frente a GBM**: la heterocedasticidad condicional produce una estimación más realista de `Var[R(w)]`, que es el término penalizado en la función objetivo.

5. **Ignorar correlaciones sobreestima la diversificación**: la extensión a shocks correlacionados (Cholesky / DCC-GARCH) es la mejora de mayor impacto sobre la calidad del óptimo.

6. **La arquitectura del pipeline** integra rigurosamente teoría numérica (KKT, SQP, Lagrangiano aumentado) con modelado econométrico (GARCH) y validación empírica (out-of-sample), cubriendo los cuatro temas del temario.

---

## Referencias

- Nocedal, J. & Wright, S. (2006). *Numerical Optimization* (2nd ed.). Springer. — Algoritmos 16.3, 18.3
- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77–91.
- Engle, R. (1982). Autoregressive Conditional Heteroscedasticity. *Econometrica*, 50(4).
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroscedasticity. *Journal of Econometrics*, 31(3).
- Engle, R. & Sheppard, K. (2002). Theoretical and Empirical Properties of DCC. NBER Working Paper.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
