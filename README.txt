# Resolución de ecuaciones diferenciales mediante redes neuronales informadas por la física

Repositorio asociado al Trabajo Fin de Grado:

**Resolución de ecuaciones diferenciales mediante redes neuronales informadas por la física**  
**Autor:** Miguel Cáceres Nogales  
**Grado en Física**  
**Universidad de Córdoba**  
**Curso académico:** 2025/2026

---

## Descripción general

Este repositorio contiene el código, los datos entrenados y las figuras generadas durante el desarrollo del Trabajo Fin de Grado, centrado en el estudio de las **Physics-Informed Neural Networks** o **PINNs** para la resolución de ecuaciones diferenciales.

El trabajo analiza el uso de redes neuronales informadas por la física como herramienta computacional para resolver problemas directos e inversos en física matemática. Para ello, se comparan métodos clásicos de integración numérica con redes neuronales supervisadas y modelos PINN.

El proyecto se divide en dos bloques principales:

1. Resolución del oscilador armónico amortiguado.
2. Aplicación de PINNs a un problema inverso de mecánica de fluidos mediante Hidden Fluid Mechanics.

---

## Objetivos

Los objetivos principales del proyecto son:

- Resolver ecuaciones diferenciales mediante métodos numéricos clásicos.
- Implementar redes neuronales multicapa desde cero utilizando NumPy.
- Implementar modelos neuronales en PyTorch.
- Construir una Physics-Informed Neural Network para el oscilador armónico amortiguado.
- Comparar precisión, coste computacional y estabilidad entre los distintos métodos.
- Aplicar el formalismo Hidden Fluid Mechanics a un problema inverso basado en las ecuaciones de Navier-Stokes.
- Estudiar la robustez de las PINNs frente a ruido y escasez de datos.
- Analizar el papel de las PINNs como herramienta complementaria en problemas físicos con información parcial.

---

## Contenido del repositorio

La estructura principal del repositorio es la siguiente:

```text
.
└── TFG_
    ├── CODIGO
    │   ├── amortiguado.py
    │   ├── hfm_pinn.py
    │   ├── oscilador_amortiguado_runge_kutta_4.py
    │   ├── PERCEPTRONS_DIAGRAMA.py
    │   ├── PINN_DIAGRAMA.py
    │   ├── red_neuronal_mlp_numpy.py
    │   ├── red_neuronal_mlp_pytorch.py
    │   └── Tangh.py
    │
    ├── DATA
    │   ├── Modelos entrenados de la red MLP en NumPy
    │   ├── Modelos entrenados de la red MLP en PyTorch
    │   ├── Modelos entrenados de la PINN del oscilador
    │   └── Datos auxiliares en formato .npz, .pth y .npy
    │
    ├── GRAFICAS
    │   ├── GRAFICAS_TEORIA
    │   ├── RED_MLP_NUMPY
    │   ├── RED_MLP_PYTORCH
    │   ├── RED_PINN_OSCILADOR
    │   └── RED_HFM_PINN
    │
    └── red_pinn_oscilador.py
```

---

## Descripción de los módulos principales

### Oscilador armónico amortiguado

El primer bloque del proyecto estudia el oscilador armónico amortiguado, descrito mediante una ecuación diferencial ordinaria de segundo orden.

Este sistema se utiliza como problema de validación porque permite comparar diferentes métodos frente a una solución conocida.

Los archivos principales asociados a este bloque son:

```text
TFG_/CODIGO/amortiguado.py
TFG_/CODIGO/oscilador_amortiguado_runge_kutta_4.py
TFG_/CODIGO/red_neuronal_mlp_numpy.py
TFG_/CODIGO/red_neuronal_mlp_pytorch.py
TFG_/red_pinn_oscilador.py
```

En este bloque se comparan:

- Solución analítica.
- Método de Runge-Kutta de cuarto orden.
- Red neuronal multicapa implementada en NumPy.
- Red neuronal multicapa implementada en PyTorch.
- Physics-Informed Neural Network.

---

### Red neuronal MLP en NumPy

El archivo:

```text
TFG_/CODIGO/red_neuronal_mlp_numpy.py
```

contiene una implementación propia de una red neuronal multicapa utilizando NumPy.

Este modelo permite estudiar de forma explícita los elementos fundamentales de una red neuronal:

- Propagación hacia delante.
- Función de activación.
- Cálculo de la función de pérdida.
- Retropropagación.
- Actualización de pesos y sesgos.
- Refinamiento progresivo del entrenamiento.

Los modelos entrenados y pruebas asociadas se encuentran en:

```text
TFG_/DATA/PRIMERA_RED
TFG_/DATA/SEGUNDA_RED
TFG_/DATA/PRUEBAS
```

Las figuras generadas se encuentran en:

```text
TFG_/GRAFICAS/RED_MLP_NUMPY
```

---

### Red neuronal MLP en PyTorch

El archivo:

```text
TFG_/CODIGO/red_neuronal_mlp_pytorch.py
```

contiene la implementación de una red neuronal multicapa utilizando PyTorch.

Este bloque permite comparar una implementación manual en NumPy con una implementación basada en un framework de aprendizaje profundo moderno.

Los modelos guardados se encuentran en:

```text
TFG_/DATA/red_oscilador.pth
TFG_/DATA/red_oscilador_pytorch_50000.pth
```

Las figuras asociadas se encuentran en:

```text
TFG_/GRAFICAS/RED_MLP_PYTORCH
```

---

### PINN aplicada al oscilador armónico amortiguado

El archivo principal de este bloque es:

```text
TFG_/red_pinn_oscilador.py
```

En este caso, la red neuronal no se entrena únicamente con datos supervisados, sino incorporando directamente la ecuación diferencial en la función de pérdida.

La función de pérdida combina:

- Error en las condiciones iniciales.
- Residuo físico de la ecuación diferencial.
- Restricciones impuestas en puntos de colocación del dominio temporal.

Los archivos entrenados asociados a este modelo se encuentran en:

```text
TFG_/DATA/pinn_oscilador_amortiguado_PINNs.npz
TFG_/DATA/pinn_oscilador_amortiguado_PINNs.pth
```

Las gráficas generadas se encuentran en:

```text
TFG_/GRAFICAS/RED_PINN_OSCILADOR
```

---

### Hidden Fluid Mechanics

El segundo bloque del trabajo aplica PINNs a un problema inverso de mecánica de fluidos mediante el enfoque conocido como **Hidden Fluid Mechanics**.

El archivo principal es:

```text
TFG_/CODIGO/hfm_pinn.py
```

En este bloque se busca reconstruir información física no observada directamente a partir de datos parciales del flujo. En particular, se estudia la inferencia del campo de presión y de parámetros físicos asociados a las ecuaciones de Navier-Stokes.

Los resultados generados se organizan según el número de observaciones y el nivel de ruido introducido en los datos:

```text
TFG_/GRAFICAS/RED_HFM_PINN/HFM
```

Ejemplos de carpetas de resultados:

```text
resultados_N1000_ruido10
resultados_N3000_ruido20
resultados_N7000_ruido30
```

Cada carpeta contiene:

```text
datos_entrenamiento.npz
fig1_hfm_campos_predichos.png
fig2_hfm_comparacion_presion.png
fig3_hfm_evolucion_temporal_p.png
loss_history.txt
metricas.txt
modelo_pinn_hfm.pt
parametros.npy
resumen_resultados.txt
```

---

## Resultados principales

### Oscilador armónico amortiguado

El método de Runge-Kutta de cuarto orden proporciona la solución más precisa y eficiente para el problema directo del oscilador armónico amortiguado.

La PINN consigue aproximar correctamente la dinámica del sistema utilizando únicamente las condiciones iniciales y el residuo físico de la ecuación diferencial.

Resultado destacado:

```text
Error relativo L2 de la PINN: 7.98e-03
```

Este resultado muestra que el modelo es capaz de reconstruir la solución sin utilizar una base completa de datos supervisados.

---

### Hidden Fluid Mechanics

En el problema inverso de Navier-Stokes, la PINN permite inferir magnitudes físicas no observadas directamente a partir de datos parciales de velocidad.

Para el caso sin ruido y con un número elevado de observaciones, se obtienen valores aproximados:

```text
λ1 ≈ 0.992
λ2 ≈ 0.0102
Re ≈ 98.2
```

Además, el error relativo de la presión inferida, eliminando el offset constante característico del problema, es aproximadamente:

```text
Error relativo L2 de presión ≈ 4.7e-02
```

Estos resultados reflejan que las PINNs son especialmente útiles en problemas inversos donde la información experimental o numérica disponible es incompleta.

---

## Gráficas generadas

Las figuras del proyecto se encuentran organizadas en la carpeta:

```text
TFG_/GRAFICAS
```

La estructura distingue entre:

```text
GRAFICAS_TEORIA
RED_MLP_NUMPY
RED_MLP_PYTORCH
RED_PINN_OSCILADOR
RED_HFM_PINN
```

Esta organización permite separar las figuras conceptuales utilizadas para explicar la metodología de las figuras obtenidas directamente a partir de los experimentos numéricos.

---

## Instalación

Para ejecutar el código se recomienda crear un entorno virtual de Python.

```bash
python -m venv .venv
source .venv/bin/activate
```

A continuación, instalar las dependencias principales:

```bash
pip install numpy scipy matplotlib pandas scikit-learn torch deepxde
```

En caso de disponer de un archivo `requirements.txt`, puede instalarse todo mediante:

```bash
pip install -r requirements.txt
```

---

## Ejecución de los scripts

Desde la raíz del repositorio, los scripts principales pueden ejecutarse de la siguiente forma:

### Solución analítica del oscilador amortiguado

```bash
python TFG_/CODIGO/amortiguado.py
```

### Método de Runge-Kutta de cuarto orden

```bash
python TFG_/CODIGO/oscilador_amortiguado_runge_kutta_4.py
```

### Red neuronal MLP implementada en NumPy

```bash
python TFG_/CODIGO/red_neuronal_mlp_numpy.py
```

### Red neuronal MLP implementada en PyTorch

```bash
python TFG_/CODIGO/red_neuronal_mlp_pytorch.py
```

### PINN aplicada al oscilador armónico amortiguado

```bash
python TFG_/red_pinn_oscilador.py
```

### PINN aplicada a Hidden Fluid Mechanics

```bash
python TFG_/CODIGO/hfm_pinn.py
```

---

## Dependencias principales

El proyecto utiliza principalmente:

- Python
- NumPy
- SciPy
- Matplotlib
- pandas
- PyTorch
- DeepXDE
- scikit-learn

Estas librerías permiten implementar tanto los métodos numéricos clásicos como los modelos neuronales empleados en el trabajo.

---

## Conclusiones

Las conclusiones principales del proyecto son:

- Los métodos clásicos siguen siendo más precisos y eficientes en problemas directos bien definidos.
- Runge-Kutta de cuarto orden proporciona excelentes resultados para el oscilador armónico amortiguado.
- Las redes neuronales supervisadas pueden aproximar soluciones, pero dependen de la calidad y cantidad de datos disponibles.
- Las PINNs permiten incorporar directamente las ecuaciones diferenciales en el proceso de entrenamiento.
- En problemas directos simples, las PINNs no sustituyen necesariamente a los métodos clásicos.
- En problemas inversos con información parcial, las PINNs presentan un potencial considerable.
- La principal ventaja de las PINNs es su capacidad para combinar datos, restricciones físicas y aprendizaje automático dentro de un mismo marco computacional.

---

## Limitaciones

Durante el desarrollo del trabajo se han identificado varias limitaciones:

- Alto coste computacional durante el entrenamiento.
- Sensibilidad a la elección de hiperparámetros.
- Dependencia de la arquitectura de la red.
- Dificultad para ponderar correctamente los distintos términos de la función de pérdida.
- Mayor complejidad de implementación frente a métodos numéricos clásicos.
- Necesidad de realizar múltiples pruebas para obtener entrenamientos estables.

---

## Posibles líneas futuras

Algunas posibles extensiones del proyecto son:

- Aplicar PINNs a otros sistemas descritos mediante ecuaciones diferenciales parciales.
- Incorporar estrategias adaptativas de selección de puntos de colocación.
- Comparar diferentes técnicas de ponderación de la función de pérdida.
- Estudiar arquitecturas neuronales más avanzadas.
- Optimizar el entrenamiento en GPU.
- Aplicar el enfoque a problemas físicos con geometrías más complejas.
- Explorar métodos híbridos que combinen esquemas numéricos clásicos y redes neuronales informadas por la física.

---

## Autor

**Miguel Cáceres Nogales**  
Grado en Física  
Universidad de Córdoba  

---

## Cita recomendada

Si se utiliza este repositorio o parte del código desarrollado, se recomienda citar el trabajo como:

```bibtex
@misc{caceres2026pinn,
  author = {Cáceres Nogales, Miguel},
  title = {Resolución de ecuaciones diferenciales mediante redes neuronales informadas por la física},
  year = {2026},
  note = {Trabajo Fin de Grado, Grado en Física, Universidad de Córdoba}
}
```

---

## Licencia

Este repositorio se publica con fines académicos y de divulgación científica.

El código y los resultados pueden consultarse como material asociado al Trabajo Fin de Grado. Para reutilización, modificación o distribución, se recomienda citar adecuadamente el trabajo original.

