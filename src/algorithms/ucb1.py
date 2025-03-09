"""
Module: algorithms/ucb1.py
Description: Implementación del algoritmo UCB1 para el problema de los k-brazos.


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

class UCB1(Algorithm):

    def __init__(self, k: int):
        """
        Inicializa el algoritmo ucb1.

        :param k: Número de brazos.
        """
        super().__init__(k)
        
        self.total_counts = 0 # Contador total de selecciones 

    def select_arm(self) -> int:
        """
        Selecciona un brazo utilizando la estrategia UCB1.
        :return: Índice del brazo seleccionado
        """

        if self.total_counts < self.k:
            return self.total_counts  # Asegurar que cada brazo se prueba al menos una vez
        
        ucb_values = self.values + np.sqrt((2 * np.log(self.total_counts)) / self.counts)
        return np.argmax(ucb_values)
    
    def update(self, chosen_arm, reward):
        """Actualiza las estimaciones del brazo seleccionado."""
        super().update(chosen_arm, reward)
        self.total_counts += 1

    def reset(self):
        super().reset()
        self.total_counts = 0

