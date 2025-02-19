"""
Module: algorithms/ucb2.py
Description: Implementación del algoritmo UCB2 para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.ucb1 import UCB1

class UCB2(UCB1):

    def __init__(self, k: int, alpha: float = 0.5):
        """
        Inicializa el algoritmo ucb2.

        :param k: Número de brazos.
        :param epsilon: Probabilidad de exploración (seleccionar un brazo al azar).
        :raises ValueError: Si epsilon no está en [0, 1].
        """
        assert 0 < alpha < 1, "El parámetro epsilon debe estar entre 0 y 1."

        super().__init__(k)
        
        self.alpha = alpha
        self.r = np.zeros(self.k, dtype=int)  # Cuándo volver a probar un brazo


    def select_arm(self):
        """Selecciona un brazo utilizando la estrategia UCB2."""

        for arm in range(self.k):
            if self.counts[arm] == 0 or self.counts[arm] < self.r[arm]:
                return arm  # Explorar cada brazo al menos una vez
    
        
        ucb_values = self.values + np.sqrt((1 + self.alpha) * np.log(np.exp(1) * self.counts) / (2 * self.counts))
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        """Actualiza las estimaciones del brazo seleccionado."""
        super().update(chosen_arm, reward)
        self.r[chosen_arm] = self.counts[chosen_arm] * (1 + self.alpha)

    def reset(self):
        super().reset()
        self.r = np.zeros(self.k, dtype=int)










    





