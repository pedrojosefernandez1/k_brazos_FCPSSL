"""
Module: algorithms/gradiente_preferencias.py
Description: Implementación del algoritmo gradiente de preferencias para el problema de los k-brazos.


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

class GradientePreferencias(Algorithm):
    def __init__(self, k, alpha=0.1):
        super().__init__(k)
        self.alpha = alpha
        self.preferences = np.zeros(self.k)
        self.average_reward = 0
        self.total_counts = 0
    
    def select_arm(self):
        """Selecciona un brazo usando el algoritmo de Gradiente de Preferencias."""
        exp_preferences = np.exp(self.preferences)
        probabilities = exp_preferences / np.sum(exp_preferences)
        return np.random.choice(self.k, p=probabilities)
    
    def update(self, chosen_arm, reward):
        """Actualiza las preferencias del brazo seleccionado."""
        super().update(chosen_arm, reward)
        self.total_counts += 1
        self.average_reward += (reward - self.average_reward) / self.total_counts
        probabilities = np.exp(self.preferences) / np.sum(np.exp(self.preferences))
    
        for arm in range(self.k):
            if arm == chosen_arm:
                self.preferences[chosen_arm] += self.alpha * (reward - self.average_reward) * (1 - probabilities[chosen_arm])
            else:
                self.preferences[arm] -= self.alpha * (reward - self.average_reward) * probabilities[arm]

    def reset(self):
        """
        Reinicia el estado del algoritmo (opcional).
        """
        super().reset()
        self.preferences = np.zeros(self.k)
        self.average_reward = 0
        self.total_counts = 0