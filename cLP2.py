import pulp
import pandas as pd


def dot(l1, l2):
    """Dot product between two lists"""
    return sum([e1 * e2 for e1, e2 in zip(l1, l2)])


class LP:
    def __init__(self, *args):
        self.q_names = ['<5,000 cfs', '5-6,000 cfs', '6-8,000 cfs', '8-10,000 cfs', '10,000+ cfs']
        self.q_probs = [0.8, 0.11, 0.06, 0.02, 0.01]
        if 'upstream' in args:
            self.q_probs = [0.9, 0.05, 0.03, 0.01, 0.01]
        self.damage0 = [0, 2.1e6, 3e6, 4.2e6, 6e6]

        self.perm_names = ['Raise Structures', 'Warning System', 'Sacrificial First Stories']
        self.perm_costs = [10, 1, 40]
        self.perm_lims = [1e6, 200e3, 200e3]
        self.perm_reds = [[0, 100, 70, 60, 10], [0, 2, 3, 4, 7], [0, 100, 60, 50, 20]]
        # self.perm_reds = [[0, 200, 90, 70, 10], [0, 3, 4, 6, 10], [0, 100, 60, 50, 20]] # test case (Jay's paper example)

        self.em_names = ['Evacuate', 'Sandbagging', 'Heightened Levee Monitoring']
        self.em_costs = [200e3, 20e3, 1]
        # self.em_costs = [100e3, 30e3, 1] # test case (Jay's paper example)
        self.em_lims = [1, 2, 20e3]
        self.em_reds = [[0, 200e3, 300e3, 500e3, 1e6], [0, 1e6, 800e3, 0, 0], [0, 2, 1, 0, 0]]

        # make set of linear coefficients for optimization
        # index 0-2: permanent options

        print('base damage: %.2f' % dot(self.q_probs, self.damage0))

        self.cs = []
        print('\nWorthwhile investments:\n')
        for i in range(len(self.perm_names)):
            c = self.perm_costs[i] - dot(self.q_probs, self.perm_reds[i])
            if c < 0:
                print(self.perm_names[i])
            self.cs.append(c)
        # index 3-17: em options, listed primarily by q, secondarily by option
        for q in range(len(self.q_names)):
            for j in range(len(self.em_names)):
                c = self.q_probs[q] * (self.em_costs[j] - self.em_reds[j][q])
                if c < 0:
                    print(self.em_names[j] + ', ' + self.q_names[q])
                self.cs.append(c)

        self.run_LP()


    def run_LP(self):
        # initialize decision variables
        # list of permanent variables
        perm_vars = [pulp.LpVariable(self.perm_names[i], lowBound=0, upBound=self.perm_lims[i], cat='Integer') for i in range(len(self.perm_names))]
        # list of dicts of emergency variables, dict for each emergency option has flow names as keys
        em_vars = [pulp.LpVariable.dict(self.em_names[j], self.q_names, lowBound=0, upBound=self.em_lims[j], cat='Integer') for j in range(len(self.em_names))]

        self.varlist = perm_vars
        for q in self.q_names:
            for j in range(len(self.em_names)):
                self.varlist.append(em_vars[j][q])

        # initialize model, set to minimize objective fn
        model = pulp.LpProblem('Optimizing Floodplain Management', pulp.LpMinimize)

        # define objective function
        model.objective += pulp.lpSum([self.cs[i] * self.varlist[i] for i in range(len(self.varlist))])
        model.objective += dot(self.q_probs, self.damage0)

        # initialize constraints
        for q in range(1, len(self.q_names)):
            model += pulp.lpSum([self.perm_reds[i][q] * perm_vars[i] for i in range(len(self.perm_names))]) + pulp.lpSum([self.em_reds[j][q] * em_vars[j][self.q_names[q]] for j in range(len(self.em_names))]) <= self.damage0[q]

        model.solve()
        print('\nModel status: %s\n' % pulp.LpStatus[model.status])
        print('Model Objective:')
        print(model.objective)
        print('\nModel constraints:')
        print(model.constraints)
        print('\nObjective function value: %.2f' % pulp.value(model.objective))
        print('\nOptimal values:\n')
        for var in model.variables():
            print var
            print var.varValue

        self.perm_df = pd.DataFrame({'Implemented Quantity': [perm_vars[i].value() for i in range(len(self.perm_names))]}, index=self.perm_names)
        self.perm_df.index.name = 'Permanent Option'
        self.em_df = pd.DataFrame(dict(zip(self.q_names, [[em_vars[j][q].value() for j in range(len(self.em_names))] for q in self.q_names])), index=self.em_names)
        self.em_df = self.em_df[self.q_names]
        self.em_df.index.name = 'Emergency Option'

        print(self.perm_df)
        print(self.em_df)
        return model



lp = LP()
lp.run_LP()
