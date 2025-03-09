from typing import List

import numpy as np

from algorithms import Algorithm
from arms import  Bandit

def run_default_experiment(bandit: Bandit, algorithms: List[Algorithm]):

    steps = 1000 # Número de pasos
    runs = 500 # Número de ejecuciones

    optimal_arm = bandit.optimal_arm  # Necesario para calcular el porcentaje de selecciones óptimas.

    rewards = np.zeros((len(algorithms), steps)) # Matriz para almacenar las recompensas promedio.

    optimal_selections = np.zeros((len(algorithms), steps))  # Matriz para almacenar el porcentaje de selecciones óptimas.

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

    optimal_selections /= runs

    for algo in algorithms:
        stats_arms_algorithms[algo]['counts'] /= runs
        stats_arms_algorithms[algo]['rewards'] /= runs

    regret = np.cumsum(bandit.get_expected_value(optimal_arm) - rewards, axis=1)

    return rewards, optimal_selections, stats_arms_algorithms, regret

def run_experiment(bandit: Bandit, algorithms: List[Algorithm], steps: int, runs: int):

    optimal_arm = bandit.optimal_arm  # Necesario para calcular el porcentaje de selecciones óptimas.

    rewards = np.zeros((len(algorithms), steps)) # Matriz para almacenar las recompensas promedio.

    optimal_selections = np.zeros((len(algorithms), steps))  # Matriz para almacenar el porcentaje de selecciones óptimas.

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
                optimal_selections[idx, step] += int(chosen_arm == optimal_arm) # Actualizar el porcentaje de selecciones óptimas.
        for algo in algorithms:
            stats_arms_algorithms[algo]['counts'] += np.array(algo.counts, dtype=float)
            stats_arms_algorithms[algo]['rewards'] += algo.values
            
    rewards /= runs

    optimal_selections /= runs

    for algo in algorithms:
        stats_arms_algorithms[algo]['counts'] /= runs
        stats_arms_algorithms[algo]['rewards'] /= runs

    regret = np.cumsum(bandit.get_expected_value(optimal_arm) - rewards, axis=1)

    return rewards, optimal_selections, stats_arms_algorithms, regret