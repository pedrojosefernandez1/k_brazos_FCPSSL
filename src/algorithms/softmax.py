"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo softmax para el problema de los k-brazos.


Author: Jaime Pujante Sáez
Email: jaime.pujantes@um.es

Author: Ricardo Javier Sendra Lázaro
Email: ricardojavier.sendral@um.es

Author: Pedro José Fernandez Campillo
Email: pedrojose.fernandez1@um.es

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class Softmax(Algorithm):
    def __init__(self, k, tau=1.0):
        """
        Inicializa el algoritmo ucb2.

        :param k: Número de brazos.
        :param tau: Parámetro constante tau
        :raises ValueError: Si epsilon no está en [0, ...].
        """
        assert 0 < tau, "El parámetro tau debe ser estrictamente mayor que 0."

        super().__init__(k)
        self.tau = tau
    
    def select_arm(self):
        """Selecciona un brazo usando la estrategia Softmax."""
        exp_values = np.exp(self.values / (self.tau + 1e-5))
        probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(self.k, p=probabilities)

    def update(self, chosen_arm, reward):
        """Actualiza las estimaciones del brazo seleccionado."""
        super().update(chosen_arm, reward)

    def reset(self):
        super().reset()