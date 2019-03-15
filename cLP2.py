import numpy as np
import pandas as pd
import pulp


def dot(l1, l2):
    """Dot product between two lists"""
    return sum([e1 * e2 for e1, e2 in zip(l1, l2)])


class LP:
    def __init__(self, *args):
        self.q_names = ['< 5,000 cfs', '5-6,000 cfs', '6-8,000 cfs', '8-10,000 cfs', '10,000+ cfs']
        self.q_probs = [0.8, 0.11, 0.06, 0.02, 0.01]
        if 'upstream' in args:
            self.q_probs = [0.9, 0.05, 0.03, 0.01, 0.01]
        self.damage0 = [0, 2.1e6, 3e6, 4.2e6, 6e6]

        self.perm_names = ['Raise Structures', 'Warning System', 'Sacrificial First Stories']
        self.perm_costs = [10, 1, 40]
        self.perm_lims = [1e6, 200e3, 200e3]
        self.perm_reds = [[0, 100, 70, 60, 10], [0, 2, 3, 4, 7], [0, 100, 60, 50, 20]]
        self.perm_reds = [[0, 200, 90, 70, 10], [0, 3, 4, 6, 10], [0, 100, 60, 50, 20]] # ***

        self.em_names = ['Evacuate', 'Sandbagging', 'Heightened Levee Monitoring']
        self.em_costs = [200e3, 20e3, 1]
        self.em_costs = [100e3, 30e3, 1] # ***
        self.em_lims = [1, 2, 20e3]
        self.em_reds = [[0, 200e3, 300e3, 500e3, 1e6], [0, 1e6, 800e3, 0, 0], [0, 2, 1, 0, 0]]

        # make set of linear coefficients for optimization
        # index 0-2: permanent options
        self.coefficients = []
        for i in range(len(self.perm_names)):
            c = self.perm_costs[i] - dot(self.q_probs, self.perm_reds[i])
            self.coefficients.append(c)
        # index 3-17: em options, listed primarily by q, secondarily by option
        for q in range(len(self.q_names)):
            for j in range(len(self.em_names)):
                c = self.q_probs[q] * (self.em_costs[j] - self.em_reds[j][q])
                self.coefficients.append(c)


    def run_LP(self):
        # initialize decision variables
        perm_vars = [pulp.LpVariable(self.perm_names[i], lowBound=0, upBound=self.perm_lims[i], cat='Integer') for i in range(len(self.perm_names))]
        em_vars = [pulp.LpVariable.dict(self.em_names[j], self.q_names, lowBound=0, upBound=self.em_lims[j], cat='Integer') for j in range(len(self.em_names))]

        # initialize model
        model = pulp.LpProblem('Optimizing Floodplain Management', pulp.LpMinimize)

        # initialize objective function
        # costs of implementing permanent options
        model.objective += pulp.lpSum([perm_vars[i] * self.perm_costs[i] for i in range(len(self.perm_names))])
        # evc of implementing emergency options
        for j, em_var in enumerate(em_vars):
            model.objective += pulp.lpSum([self.q_probs[q] * self.em_costs[j] * em_var[self.q_names[q]] for q in range(len(self.q_names))])
        # evc base damages
        model.objective += pulp.lpSum([self.q_probs[q] * self.damage0[q] for q in range(len(self.q_names))])
        # evc permanent damage reductions
        for i, perm_var in enumerate(perm_vars):
            model.objective -= pulp.lpSum([self.q_probs[q] * self.perm_reds[i][q] * perm_var for q in range(len(self.q_names))])
        # evc emergency damage reductions
        for j, em_var in enumerate(em_vars):
            model.objective -= pulp.lpSum([self.q_probs[q] * self.em_reds[i][q] * em_var[self.q_names[q]] for q in range(len(self.q_names))])

        # initialize constraints
        for q in range(len(self.q_names)):
            model += pulp.lpSum([self.q_probs[q] * self.perm_reds[i][q] * perm_vars[i] for i in range(len(self.perm_names))]) + pulp.lpSum([self.q_probs[q] * self.em_reds[j][q] * em_vars[j][self.q_names[q]] for j in range(len(self.em_names))]) <= self.damage0[q]

        model.solve()
        print(pulp.value(model.objective))
        print(model.)

        return model


lp = LP()
lp.run_LP()