"""
    Author: Stephen Pauwels
"""

import multiprocessing as mp
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# from CPT_inverted_index import CPT_inverted_index
from Methods.EDBN.model.CPT import CPT
# from EDBN.NNTable import NNT
import Utils.Result as Result


def calculate(trace, model, time_attribute):
    case = trace[0]
    data = trace[1]

    result = Result.Trace_result(case, time=data[time_attribute].iloc[0])
    for row in data.itertuples():
        e_result = model.test_row(row)
        result.add_event(e_result)
    return result



class ExtendedDynamicBayesianNetwork():
    """
    Class for representing an extended Dynamic Bayesian Network (EDBN)
    """

    def __init__(self, num_attrs, k, trace_attr):
        self.variables = {}
        self.current_variables = []
        self.num_attrs = num_attrs
        self.log = None
        self.k = k
        self.trace_attr = trace_attr
        self.durations = None

    def print_parents(self):
        for var_name, var in self.variables.items():
            for parent in var.conditional_table.parents:
                print(parent.attr_name, "->", var_name)
            for parent in var.functional_parents:
                print(parent.attr_name, "=>", var_name)

    def add_discrete_variable(self, name, new_values, empty_val):
        self.variables[name] = Discrete_Variable(name, new_values, self.num_attrs, empty_val)
        m = re.search(r'Prev\d+$', name)
        if m is None:
            self.current_variables.append(name)

    def add_numerical_variable(self, name):
        self.variables[name] = Numerical_Variable(name, self.num_attrs)
        m = re.search(r'Prev\d+$', name)
        if m is None:
            self.current_variables.append(name)

    def add_discretized_variable(self, name):
        self.variables[name] = Discretized_Variable(name, self.num_attrs)
        m = re.search(r'Prev\d+$', name)
        if m is None:
            self.current_variables.append(name)

    def remove_variable(self, name):
        del self.variables[name]

    def add_edges(self, edges):
        for edge in edges:
            self.variables[edge[1]].add_parent(self.variables[edge[0]])

    def iterate_variables(self):
        for key in self.variables:
            yield (key, self.variables[key])

    def iterate_current_variables(self):
        for key in self.current_variables:
            yield (key, self.variables[key])

    def get_variable(self, attr_name):
        return self.variables[attr_name]

    def train(self, data, single=False):
        print("GENERATE: Start Training")
        if isinstance(data, pd.DataFrame):
            self.log = data
        else:
            self.log = data.get_data()

        for (_, value) in self.iterate_current_variables():
            value.train(self.log)

        # TODO: instead of training all variables, just copy current values to previous variables
        for (_, value) in self.iterate_variables():
            value.train(self.log)

        print("GENERATE: Training Done")

    def update(self, row):
        for (_, value) in self.iterate_variables():
            value.update(row)

    def update_log(self, log):
        for (_, value) in self.iterate_variables():
            value.update_log(log)

    def calculate_scores_per_trace(self, data, accum_attr=None):
        """
        Return the result for all traces in the data
        """
        # def initializer(init_model, time_attr):
        #     global model
        #     model = init_model
        #     global time_attribute
        #     time_attribute = time_attr

        data.create_k_context()

        print("EVALUATION: Calculate Scores")
        if accum_attr is None:
            accum_attr = self.trace_attr
        scores = []

        for group in data.contextdata.groupby([accum_attr]):
            scores.append(calculate(group, self, data.time))
        # Disabled due to pickle error
        # with mp.Pool(mp.cpu_count(), initializer, (self, data.time)) as p:
        #     scores = p.map(calculate, data.contextdata.groupby([accum_attr]))
        print("EVALUATION: Scores Calculated")
        # scores.sort(key=lambda l: l.time)
        return scores

    def calculate_scores_per_attribute(self, data, accum_attr = None):
        """
        Return the results for all traces per attribute
        """
        result = self.calculate_scores_per_trace(data, accum_attr)
        scores = {}

        print("EVALUATION: Combine by attribute")
        for attribute in [x for x in self.current_variables]:
            for trace_result in result:
                if attribute not in scores:
                    scores[attribute] = []
                scores[attribute].append(trace_result.get_attribute_score(attribute))

        return scores

    def test_data(self, data):
        """
        Compute the score for all events in the k-context of the data
        """
        print("EVALUATION: Calculate Scores")
        data.create_k_context()
        log = data.contextdata

        with mp.Pool(mp.cpu_count()) as p:
            if self.trace_attr is not None:
                result = p.map(self.test_trace, log.groupby([self.trace_attr]))
            else:
                result = self.test_parallel(data.get_data())
        print("EVALUATION: Scores Calculated")
        return result


    def test_parallel(self, data):
        njobs = mp.cpu_count()
        size = data.shape[0]
        if size < njobs:
            njobs = 1

        results = []
        chunks = np.array_split(data, njobs)

        for r in Parallel(n_jobs=njobs)(delayed(self.test)(d) for d in chunks):
            results.extend(r)
        return results

    def test_trace(self, trace):
        result = Result.Trace_result(trace[0])
        for row in trace[1].itertuples():
            result.add_event(self.test_row(row))
        return result

    def test(self, data):
        """
        Compute the scores for all events in the data
        :param data:
        :return:
        """
        ranking = []
        for row in data.itertuples():
            result = Result.Trace_result(row.Index)
            result.add_event(self.test_row(row))
            ranking.append(result)
        return ranking

    def test_row(self, row):
        """
        Return the score for the k-context of a single event
        """
        result = Result.Event_result(row.Index)#, getattr(row, "anom_types"))
        for (key, value) in self.iterate_current_variables():
            result.set_attribute_score(value.attr_name, value.test(row))
        return result



class Variable:
    """
    Basic class representing a variable of an EDBN
    """
    def __init__(self, attr_name, new_values, num_attrs, empty_val):
        self.attr_name = attr_name
        self.new_values = new_values
        self.new_relations = 0
        self.num_attrs = num_attrs
        self.empty_val = empty_val

    def __repr__(self):
        return self.attr_name

    def add_parent(self, var):
        pass

    def add_mapping(self, var):
        pass

    def train(self, log):
        pass

    def test(self, row):
        pass

    def update(self, row):
        pass

    def update_log(self, log):
        pass


class Discrete_Variable(Variable):
    """
    Discrete variable of an EDBN
    """
    def __init__(self, attr_name, new_values, num_attrs, empty_val, conditional_table="nnt"):
        self.attr_name = attr_name
        self.new_values = new_values
        self.empty_val = empty_val
        self.total_rows = 0

        self.values = set()
        self.value_counts = {}

        # self.conditional_table = NNT(self.attr_name)
        self.conditional_table = CPT(self.attr_name)
        # self.conditional_table = CPT_inverted_index(self.attr_name)
        self.functional_parents = []
        self.fdt = []
        self.fdt_violation = []

    def __repr__(self):
        return self.attr_name

    def add_parent(self, var):
        self.conditional_table.add_parent(var)

    def remove_parent(self, var):
        self.conditional_table.remove_parent(var)

    def add_mapping(self, var):
        self.functional_parents.append(var)
        self.fdt.append({})

    def get_conditional_parents(self):
        return self.conditional_table.parents

    ###
    # Training
    ###
    def train(self, log):
        self.train_variable(log)
        self.train_fdt(log)
        self.conditional_table.train(log)
        # self.cpt.train(log)

    def train_variable(self, log):
        print("Train Variable", self.attr_name)
        self.value_counts = {val: set(log.index[log[self.attr_name] == val].tolist()) for val in log[self.attr_name].unique()}
        # TODO: rework so self.values becomes obsolete
        self.values = {val: len(self.value_counts[val]) for val in self.value_counts}
        self.total_rows = log.shape[0]
        self.conditional_table.num_values = log[self.attr_name].max()

    def train_fdt(self, log):
        if len(self.functional_parents) == 0:
            return

        for i in range(len(self.functional_parents)):
            violations = 0
            log_size = log.shape[0]
            parent = self.functional_parents[i]
            grouped = log.groupby([parent.attr_name, self.attr_name]).size().reset_index(name='counts')
            tmp_mapping = {}
            for t in grouped.itertuples():
                row = list(t)
                parent_val = row[1]
                val = row[2]
                if parent_val not in tmp_mapping:
                    tmp_mapping[parent_val] = (row[-1], val)
                elif row[-1] > tmp_mapping[parent_val][0]:
                    violations += tmp_mapping[parent_val][0] # Add previous number to violations
                    tmp_mapping[parent_val] = (row[-1], val)
                else:
                    violations += row[-1] # Add number to violations

            for p in tmp_mapping:
                self.fdt[i][p] = tmp_mapping[p][1]

            self.fdt_violation.append(violations / log_size)

    def update(self, row):
        atr_value = getattr(row, self.attr_name)
        if atr_value in self.values:
            self.values[atr_value] += 1
            self.total_rows += 1
        else:
            self.values[atr_value] = 1
            self.total_rows += 1
        self.conditional_table.update(row)

    def update_log(self, log):
        for row in log.contextdata.iterrows():
            self.update(row[1])

    ###
    # Testing
    ###
    def test(self, row):
        total_score = 1
        for score in self.test_fdt(row).values():
            total_score *= score
        total_score *= self.conditional_table.test(row)
        total_score *= self.test_value(row)
        if total_score == 0:
            total_score = 0.0000000001
        return np.log(total_score)

    def test_fdt(self, row):
        scores = {}
        if len(self.functional_parents) > 0:
            for i in range(len(self.functional_parents)):
                parent = self.functional_parents[i]
                if getattr(row, parent.attr_name) not in self.fdt[i]:
                    scores[parent.attr_name] = 1 - self.fdt_violation[i]
                    self.fdt[i][getattr(row, parent.attr_name)] = getattr(row, self.attr_name)
                    # TODO: check if still needed
                    # self.values.add(getattr(row, self.attr_name))
                elif self.fdt[i][getattr(row, parent.attr_name)] == getattr(row, self.attr_name) or getattr(row, parent.attr_name) == 0:
                    scores[parent.attr_name] = 1 - self.fdt_violation[i]
                else:
                    scores[parent.attr_name] = self.fdt_violation[i]
        return scores

    def test_value(self, row):
        if getattr(row, self.attr_name) not in self.values:
            return self.new_values
        else:
            return 1 - self.new_values



class Numerical_Variable(Variable):
    """
    Numerical variable of an EDBN
    """
    def __init__(self, attr_name, num_attrs):
        self.attr_name = attr_name
        self.num_attrs = num_attrs

        self.kernels_nom = {}
        self.kernels_denom = {}
        self.kernel_nominator = None
        self.kernel_denominator = None

        self.discrete_parents = []
        self.continuous_parents = []

    def __repr__(self):
        return self.attr_name

    def add_parent(self, var):
        if isinstance(var, Numerical_Variable):
            self.continuous_parents.append(var)
        else:
            self.discrete_parents.append(var)

    def add_mapping(self, var):
        raise NotImplementedError()


    ###
    # Training
    ###
    def train(self, log):
        self.train_relations(log)

    def train_relations(self, log):
        from sklearn.model_selection import GridSearchCV
        from sklearn.neighbors import KernelDensity

        if len(self.discrete_parents) > 0: # Split log in partitions according to the discrete parents
            partitions = log.groupby([p.attr_name for p in self.discrete_parents])
        else: # If no discrete parents -> split log in just one partition
            partitions = [("", log)]

        cont_par = [p.attr_name for p in self.continuous_parents]

        for partition in partitions:
            if len(partition[1]) <= 20:
                continue

            if isinstance(partition[0], int):
                disc_parent = str(partition[0])
            else:
                disc_parent = "-".join([str(i) for i in partition[0]])

            nom_vals = partition[1][cont_par + [self.attr_name]].values
            # Calculate best bandwith for KDE
            params = {'bandwidth': np.logspace(-2, 1, 20)}
            grid = GridSearchCV(KernelDensity(), params, cv=2, n_jobs=mp.cpu_count())
            grid.fit(nom_vals)

            self.kernels_nom[disc_parent] = KernelDensity(kernel='gaussian', bandwidth=grid.best_estimator_.bandwidth).fit(nom_vals)

            if len(cont_par) > 0:
                denom_vals = partition[1][cont_par].values
                # Calculate best bandwith for KDE
                params = {'bandwidth': np.logspace(-2, 1, 20)}
                grid = GridSearchCV(KernelDensity(), params, cv=2, n_jobs=mp.cpu_count())
                grid.fit(denom_vals)

                self.kernels_denom[disc_parent] = KernelDensity(kernel='gaussian', bandwidth=grid.best_estimator_.bandwidth).fit(denom_vals)

        vals = log[cont_par + [self.attr_name]].values
        # Calculate best bandwith for KDE
        params = {'bandwidth': np.logspace(-2, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params, cv=2, n_jobs=mp.cpu_count())
        grid.fit(vals)

        self.kernel_nominator = KernelDensity(kernel='gaussian', bandwidth=grid.best_estimator_.bandwidth).fit(vals)

        if len(cont_par) > 0:
            vals = log[cont_par].values
            # Calculate best bandwith for KDE
            params = {'bandwidth': np.logspace(-2, 1, 20)}
            grid = GridSearchCV(KernelDensity(), params, cv=2, n_jobs=mp.cpu_count(), return_train_score=False)
            grid.fit(vals)

            self.kernel_denominator = KernelDensity(kernel='gaussian', bandwidth=grid.best_estimator_.bandwidth).fit(vals)
        else:
            self.kernel_denominator = None

    ###
    # Testing
    ###
    def test(self, row):
        return self.test_continuous(row)


    def test_continuous(self, row):
        disc_par = []
        for p in self.discrete_parents:
            disc_par.append(str(getattr(row, p.attr_name)))
        disc_par_val = "-".join(disc_par)

        val = getattr(row, self.attr_name)

        parent_vals = np.zeros(len(self.continuous_parents) + 1)
        i = 0
        for p in self.continuous_parents:
            parent_vals[i] = getattr(row, p.attr_name)
            i += 1
        parent_vals[-1] = getattr(row, self.attr_name)
        parent_vals = parent_vals.reshape(1, -1)

        if disc_par_val in self.kernels_nom:
            nominator = self.kernels_nom[disc_par_val].score_samples(parent_vals)[0]
            if disc_par_val in self.kernels_denom and len(parent_vals) > 1:
                denominator = self.kernels_denom[disc_par_val].score_samples(parent_vals[:-1])[0]
            else:
                denominator = 0
        else:
            nominator = self.kernel_nominator.score_samples(parent_vals)[0]
            if self.kernel_denominator is None:
                denominator = 0
            else:
                denominator = self.kernel_denominator.score_samples(parent_vals[:-1])[0]

        return nominator - denominator



class Discretized_Variable(Variable):
    """
    Discretized numerical variable for the EDBN
    """
    def __init__(self, attr_name, num_attrs):
        self.attr_name = attr_name
        self.new_relations = 0
        self.num_attrs = num_attrs

    def __repr__(self):
        return self.attr_name

    def add_parent(self, var):
        pass

    def add_mapping(self, var):
        pass

    def train(self, log):
        self.value_counts = {}
        vc = log[self.attr_name].value_counts(normalize=True)
        print(vc)
        for row in vc.index:
            self.value_counts[row] = vc[row]

    def test(self, row):
        if getattr(row, self.attr_name) in self.value_counts:
            return self.value_counts[getattr(row, self.attr_name)]
        return 0
