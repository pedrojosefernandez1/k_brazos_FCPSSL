"""
Module: arms/armbinomial.py
Description: Contains the implementation of the ArmBinomial class for the Binomial distribution arm.

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
from typing import List
from arms import Arm


class ArmBinomial(Arm):
    def __init__(self, n: int, p: List[float]):
        """
        Inicializa el brazo con distribución Binomial.

        :param n: Numero de intentos.
        :param p: Probabilidad de éxito de cada intento.
        
        """
        
        assert 0 < n, "El numero de intentos debe de ser mayor de 0"
        assert n==len(p), "El conjunto de probabilidades debe de ser igual al numero de intentos"
        assert all(0 <= x <= 1 for x in p), "Las probabilidades deben de estar entre 0 y 1"

        self.n = n 
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución Binomial.

        :return: Recompensa obtenida del brazo.
        """
        
        reward = sum(np.random.rand() < prob for prob in self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Binomial.

        :return: Valor esperado de la distribución.
        """

        return sum(self.p)

    def __str__(self):
        """
        Representación en cadena del brazo Binomial.

        :return: Descripción detallada del brazo Binomial.
        """
        return f"ArmBinomial(n={self.n}p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n_min: int = 1, n_max: int = 10, p_min: float = 0.0, p_max: float = 1.0):
        """
        Genera k brazos con medias únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param n_min: Valor mínimo del numero de intentos.
        :param n_max: Valor máximo del numero de intentos.

        :param p_min: Valor mínimo de la probabilidad.
        :param p_max: Valor máximo de la probabilidad.
        :return: Lista de brazos generados.
        """
        assert 0 < n_min <= n_max, "El numero de intentos debe de ser mayor de 1"
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert 0 <= p_min < p_max <= 1, "El valor de p_min debe ser menor que p_max y deben de estar entre 0 y 1"

        # Generar k- valores únicos de mu con decimales
        arms = []
        while len(arms) < k:
            n = np.random.randint(n_min, n_max + 1)
            p = np.random.uniform(p_min, p_max, n).tolist()
            
            arms.append(ArmBinomial(n,p))

    
        return arms


