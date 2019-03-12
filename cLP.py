import numpy as np
from scipy.optimize import minimize


def dot(l1, l2):
    """Dot product between two lists"""
    return sum([e1 * e2 for e1, e2 in zip(l1, l2)])

class LP:
    def __init__(self):
        self.q_names = ['< 5,000 cfs', '5-6,000 cfs', '6-8,000 cfs', '8-10,000 cfs', '10,000+ cfs']
        self.q_probs = [0.8, 0.11, 0.06, 0.02, 0.01]
        self.q_upstream_probs = [0.9, 0.05, 0.03, 0.01, 0.01]
        self.damage0 = [0, 2.1e6, 3e6, 4.2e6, 6e6]

        self.perm_names = ['raise', 'warning', 'sacrifice']
        self.perm_costs = [10, 1, 40]
        self.perm_lims = [1e6, 200e3, 200e3]
        self.perm_reds = [[0, 100, 70, 60, 10], [0, 2, 3, 4, 7], [0, 100, 60, 50, 20]]

        self.em_names = ['evac', 'sand', 'monitor']
        self.em_costs = [200e3, 20e3, 1]
        self.em_lims = [1, 2, 20e3]
        self.em_reds = [[0, 200e3, 300e3, 500e3, 1e6], [0, 1e6, 800e3, 0, 0], [0, 2, 1, 0, 0]]

        self.perm_ranges = []
        for option in range(len(self.perm_names)):
            self.perm_ranges.append(np.linspace(0, self.perm_lims[option], 5))

        self.em_ranges = []
        for option in range(len(self.em_names)):
            self.em_ranges.append(np.linspace(0, self.em_lims[option], 5))

        self.perm_vectors = np.array(np.meshgrid(*self.perm_ranges)).T.reshape(-1, len(self.perm_names))
        self.em_vectors = np.array(np.meshgrid(*self.em_ranges)).T.reshape(-1, len(self.em_names))

    def damage(self, q, perm_vector, em_vector):
        """
        For flow index q and vectors for permanent option values and emergency option values,
        return the associated damage
        """
        d0 = self.damage0[q]
        perm_red = dot(perm_vector, [self.perm_reds[option][q] for option in range(len(self.perm_names))])
        em_red = dot(em_vector, [self.em_reds[option][q] for option in range(len(self.em_names))])

        return max(d0 - perm_red - em_red, 0)

    def evc(self, perm_vector, em_vector):
        perm_cost = dot(perm_vector, self.perm_costs)
        em_cost = dot(em_vector, self.em_costs)
        dam_cost = dot(self.q_probs, [self.damage(q, perm_vector, em_vector) for q in range(len(self.q_names))])

        return perm_cost + em_cost + dam_cost

    def run_LP(self):
        best_evc = np.inf
        best_perm = []
        best_em = []
        for perm_vector in self.perm_vectors:
            print(perm_vector)
            for em_vector in self.em_vectors:
                val = self.evc(perm_vector, em_vector)
                if val < best_evc:
                    best_evc = val
                    best_perm = perm_vector
                    best_em = em_vector

        return best_evc, best_perm, best_em


lp = LP()
solution = lp.run_LP()
print(solution)
