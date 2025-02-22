#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/pedrojosefernandez1/k_brazos_FCPSSL/blob/main/bandit_experiment.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Estudio comparativo de diferentes algoritmos en un problema de k-armed bandit
# 
# *Description:* El experimento compara el rendimiento de tres familias de algoritmos en tres problemas diferentes de k-armed bandit, teniendo las familias entre uno y dos algoritmos.
# Se generán para cada problema 3 gráficas, de recompensas promedio por pasos, de comparación de recompensas promedio por brazo y algoritmo y de arrepentimiento acumilado por pasos. Cada uno de los estudios de familias se realizará en un fichero .ipynb diferente para estructurar el estudio por módulos separados. 
# 
#     Author: Luis Daniel Hernández Molinero
#     Email: ldaniel@um.es
#     Date: 2025/01/29
# 
# This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
# with the additional restriction that it may not be used for commercial purposes.
# 
# For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
# 
# 

# ## Preparación del entorno
# 

# In[1]:


#@title Copiar el repositorio.

#!git clone https://github.com/pedrojosefernandez1/k_brazos_FCPSSL.git
#!cd k_brazos_FCPSSL/


# Vamos a preparar el entorno en el que realizaremos los experimentos. Para ello, instalamos las librerias necesarias y establecemos la semilla a utilizar, incluyendo las librerías internas que hemos desarrollado para el estudio.

# In[1]:


get_ipython().system(' pip install numpy typing seaborn nbconvert')


#@title Importamos todas las clases y funciones

import sys

# Añadir los directorio fuentes al path de Python
#sys.path.append('/content/k_brazos_FCPSSL/src')
sys.path.append('src')

# Verificar que se han añadido correctamente
print(sys.path)

import numpy as np
from typing import List

from algorithms import Algorithm
from arms import ArmNormal, Bandit, ArmBernoulli, ArmBinomial

# Fijamos la semilla para reproducibilidad
seed = 42
np.random.seed(seed)  


# ## Experimento
# 
# Cada algoritmo se ejecuta en tres problemas de k-armed bandit, cada uno con un conjunto de brazos con una distribución diferente. Cada problema se resolverá durante un número de pasos de tiempo y ejecuciones determinado.
# Se 
# Para el experimento se van a hacer:
# - 3 bandidos de distribuciones distintos
# - 500 ejecuciones por bandido
# - 1000 pasos por ejecucion
# - Cada algoritmo de cada familia y (con sus configuraciones a analizar) por cada paso
# 
# Para ello, definimos la función _run_experiment_ a continuación:

# In[8]:


def run_experiment(bandit: Bandit, algorithms: List[Algorithm], steps: int, runs: int):

    optimal_arm = bandit.optimal_arm  # Necesario para calcular el porcentaje de selecciones óptimas.

    rewards = np.zeros((len(algorithms), steps)) # Matriz para almacenar las recompensas promedio.

    optimal_selections = np.zeros((len(algorithms), steps))  # Matriz para almacenar el porcentaje de selecciones óptimas.

    np.random.seed(seed)  # Asegurar reproducibilidad de resultados.
    stats_arms_algorithms = {}
    for algo in algorithms:
        stats_arms_algorithms[algo] = {'counts':np.zeros(algo.k, dtype=float),
                                       'rewards':np.zeros(algo.k, dtype=float),
                                       'optimal': optimal_arm}    

    for run in range(runs):
        current_bandit = Bandit(arms=bandit.arms)

        for algo in algorithms:
            algo.reset() # Reiniciar los valores de los algoritmos.

        total_rewards_per_algo = np.zeros(len(algorithms)) # Acumulador de recompensas por algoritmo. Necesario para calcular el promedio.

        for step in range(steps):
            for idx, algo in enumerate(algorithms):
                chosen_arm = algo.select_arm() # Seleccionar un brazo según la política del algoritmo.
                reward = current_bandit.pull_arm(chosen_arm) # Obtener la recompensa del brazo seleccionado.

                algo.update(chosen_arm, reward) # Actualizar el valor estimado del brazo seleccionado.

                rewards[idx, step] += reward # Acumular la recompensa obtenida en la matriz rewards para el algoritmo idx en el paso step.
                total_rewards_per_algo[idx] += reward # Acumular la recompensa obtenida en total_rewards_per_algo para el algoritmo idx.

                #TODO: modificar optimal_selections cuando el brazo elegido se corresponda con el brazo óptimo optimal_arm
                optimal_selections[idx, step] += int(chosen_arm == optimal_arm) # Actualizar el porcentaje de selecciones óptimas.
        for algo in algorithms:
            stats_arms_algorithms[algo]['counts'] += np.array(algo.counts, dtype=float)
            stats_arms_algorithms[algo]['rewards'] += algo.values
            
    rewards /= runs

    # TODO: calcular el porcentaje de selecciones óptimas y almacenar en optimal_selections
    optimal_selections /= runs

    for algo in algorithms:
        stats_arms_algorithms[algo]['counts'] /= runs
        stats_arms_algorithms[algo]['rewards'] /= runs

    regret = np.cumsum(bandit.get_expected_value(optimal_arm) - rewards, axis=1)
    #for i in range(1000):
    #    if bandit.get_expected_value(optimal_arm) < rewards[0][i] :
    #        print("patata")
    #        print("bandit " + str(bandit) + " in step = " + str(i) + " optimal_arm reward " +  str(bandit.get_expected_value(optimal_arm)) + ", reward " + str(rewards[0][i]))

    return rewards, optimal_selections, stats_arms_algorithms, regret


# ## Preparación del experimento
# Para ultimar los detalles de los estudios, fijaremos las variables que nos determinarán los problemas a probar (los 3 bandidos distintos) y las ejecuciones y pasos del experimento. Concretamente se generaran 3 bandidos de 10 brazos:
# - Uno con una distribucion Normal
# - Uno con una distribucion Binomial
# - Uno con una distribucion Bernoulli
# 
# Tras generarlo, mostrará cuál es su brazo óptimo y cual es la recompensa máxima estimada.

# In[ ]:


k = 10  # Número de brazos
steps = 1000  # Número de pasos que se ejecutarán cada algoritmo
runs = 500  # Número de ejecuciones

# Creación de los bandit, cada uno con una distribución de Arm distinta
# Bandit Normal
banditNormal = Bandit(arms=ArmNormal.generate_arms(k)) # Generar un bandido con k brazos de distribución normal
optimal_arm_normal = banditNormal.optimal_arm
print(f"Optimal normal arm: {optimal_arm_normal + 1} with expected reward={banditNormal.get_expected_value(optimal_arm_normal)}")
# Bandit Binomial
banditBinomial = Bandit(arms=ArmBinomial.generate_arms(k)) # Generar un bandido con k brazos de distribución binomial
optimal_arm_binomial = banditBinomial.optimal_arm
print(f"Optimal binumial arm: {optimal_arm_binomial + 1} with expected reward={banditBinomial.get_expected_value(optimal_arm_binomial)}")
# Bandit Bernoulli
banditBernoulli = Bandit(arms=ArmBernoulli.generate_arms(k)) # Generar un bandido con k brazos de distribución bernoulli
optimal_arm_bernoulli = banditBernoulli.optimal_arm
print(f"Optimal bernoulli arm: {optimal_arm_bernoulli + 1} with expected reward={banditBernoulli.get_expected_value(optimal_arm_bernoulli)}")


# ## Almacenamos todo en un script .py
# Para poder utilizar los estudios en otros notebooks, lo mejor es almacenarlos en un script .py.

# In[7]:


get_ipython().system(' jupyter nbconvert --to script studiesIntroduction.ipynb')


# ## Ejecución de los experimentos
# 
# Pasamos a realizar los estudios, realizando las ejecuciones de los problemas resolviéndolos con cada familia de algoritmos por separado. Para ello, finalizaremos cada experimento en un notebook distinto con el objetivo de simplificar y aislar la visualización de los resultados en un intento de evitar confusiones en la lectura de los resultados. Concretamente podremos encontrarlos en:
# 
# - [ε-greedy](./epsilon-greedy.ipynb)
# - [UCB](./UCB.ipynb)
# - [Ascenso de gradiente](./ascenso-de-gradiente.ipynb)
# 
