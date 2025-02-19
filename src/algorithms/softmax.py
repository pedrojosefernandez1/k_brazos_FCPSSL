"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo softmax para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class Softmax(Algorithm):
    def __init__(self, num_arms, temperature=1.0):
        self.num_arms = num_arms
        self.temperature = temperature
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def select_arm(self):
        """Selecciona un brazo usando la estrategia Softmax."""
        exp_values = np.exp(self.values / (self.temperature + 1e-5))
        probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(self.num_arms, p=probabilities)




