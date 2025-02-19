"""
Module: algorithms/gradiente_preferencias.py
Description: Implementación del algoritmo gradiente de preferencias para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class GradientePreferencias(Algorithm):
    def __init__(self, num_arms, alpha=0.1):
        self.num_arms = num_arms
        self.alpha = alpha
        self.preferences = np.zeros(num_arms)
        self.average_reward = 0
        self.total_pulls = 0
    
    def select_arm(self):
        """Selecciona un brazo usando el algoritmo de Gradiente de Preferencias."""
        exp_preferences = np.exp(self.preferences)
        probabilities = exp_preferences / np.sum(exp_preferences)
        return np.random.choice(self.num_arms, p=probabilities)
    
    def update(self, chosen_arm, reward):
        """Actualiza las preferencias del brazo seleccionado."""
        self.total_pulls += 1
        self.average_reward += (reward - self.average_reward) / self.total_pulls
        probabilities = np.exp(self.preferences) / np.sum(np.exp(self.preferences))
        self.preferences -= self.alpha * (reward - self.average_reward) * probabilities
        self.preferences[chosen_arm] += self.alpha * (reward - self.average_reward) * (1 - probabilities[chosen_arm])




