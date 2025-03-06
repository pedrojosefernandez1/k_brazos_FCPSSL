# Bandido de K-Brazos
## Información
- **Alumnos:** Sendra Lázaro, Ricardo Javier; Pujante Saez, Jaime; Fernández Campillo, Pedro José;
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2024/2025
- **Grupo:** FCPSSL

## Descripción
Este proyecto analiza el problema del **Bandido de K-Brazos**, un escenario de **aprendizaje por refuerzo** en el que un agente debe seleccionar entre **K opciones** para maximizar su recompensa acumulada. Se estudian y comparan diferentes algoritmos en entornos **estacionarios**, evaluando su desempeño en términos de **recompensas promedio** y **regret acumulado**.

Para ello, se han implementado varias estrategias de toma de decisiones:
- **𝜀-Greedy**: Una de las estrategias más simples, que equilibra exploración y explotación.
- **Upper Confidence Bound (UCB)**: Un enfoque basado en el optimismo ante la incertidumbre.
- **Ascenso de Gradiente**: Métodos como **Softmax** y **Gradiente de Preferencias** que ajustan dinámicamente la probabilidad de selección de cada brazo.

## 📂 Estructura del Repositorio
El repositorio está organizado en los siguientes notebooks:

- **`introduccion.ipynb`**: Explicación teórica del problema y enlaces a los estudios específicos.
- **`UCB.ipynb`**: Implementación y análisis del método **Upper Confidence Bound**.
- **`ascensoGradiente.ipynb`**: Estudio de los métodos de **Softmax** y **Gradiente de Preferencias**.
- **`epsilonGreedy.ipynb`**: Evaluación de la estrategia **𝜀-Greedy**.

Además, el código fuente se encuentra dentro del directorio `src/`, con la siguiente estructura:

- **`src/algorithms/`**: Implementación de los métodos de decisión:
  - `UCB1`, `UCB2`
  - `EpsilonGreedy`
  - `Softmax`
  - `GradientePreferencias`

- **`src/arms/`**: Implementación de los modelos de recompensa:
  - `Bernoulli`
  - `Normal`
  - `Binomial`

- **`src/experiments/`**: Código para la ejecución de simulaciones y evaluación de los algoritmos.

- **`src/plotting/`**: Funciones para la visualización de resultados, incluyendo gráficos de recompensas, regret acumulado y selección de brazos.

El repositorio está organizado en los siguientes notebooks:

- **`introduccion.ipynb`**: Explicación teórica del problema y enlaces a los estudios específicos.
- **`UCB.ipynb`**: Implementación y análisis del método **Upper Confidence Bound**.
- **`ascensoGradiente.ipynb`**: Estudio de los métodos de **Softmax** y **Gradiente de Preferencias**.
- **`epsilonGreedy.ipynb`**: Evaluación de la estrategia **𝜀-Greedy**.

## ▶️ Ejecución
Para ejecutar los notebooks en Google Colab:
Dirigirse primero de todo al notebook **`main.ipynb`** donde se podra acceder a cualquier notebook que se encuentra en el repositorio de una manera intercativa. Podria seleccionar **`introduccion.ipynb`** para ver una breve introduccion al problema y enlace a los demas estudios.

3. **Abre y ejecuta los notebooks** en Google Colab o Jupyter Notebook.

## 🛠️ Tecnologías Utilizadas
El proyecto está desarrollado con:
- **Python 3.11**
- **NumPy, Matplotlib, Seaborn** para cálculos y visualización de datos.
- **Google Colab** para la ejecución interactiva.


