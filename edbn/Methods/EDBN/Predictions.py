"""
    Author: Stephen Pauwels
"""

import functools
import itertools
import re
import numpy as np

LAMBDA = 1


def test(model, log):
    return predict_next_event(model, log)


def test_and_update(logs, model):
    results = []
    i = 0
    for t in logs:
        print(i, "/", len(logs))
        i += 1
        results.extend(predict_next_event_update(model, logs[t]["data"]))
    return results


def test_and_update_retain(test_logs, model, train_log):
    from Methods.EDBN.model.LearnBayesianStructure import Structure_learner

    # Create the list of allowed edges
    restrictions = []
    attributes = list(train_log.attributes())
    for attr1 in attributes:
        if attr1 != train_log.activity:
            continue
        for attr2 in attributes:
            if attr2 not in train_log.ignoreHistoryAttributes:
                for i in range(train_log.k):
                    restrictions.append((attr2 + "_Prev%i" % i, attr1))

    learner = Structure_learner()

    results = []
    i = 0
    for t in test_logs:
        print(i, "/", len(test_logs))
        i += 1
        test_log = test_logs[t]["data"]
        results.extend(predict_next_event_update(model, test_log))

        train_log = train_log.extend_data(test_log)
        # print("Length train:", train_log.contextdata.shape)

        learner.start_model(train_log, model, restrictions)
        relations = learner.learn()

        updated = False
        activity_var = model.get_variable(train_log.activity)
        for relation in relations:
            if relation[0] not in [p.attr_name for p in activity_var.get_conditional_parents()]:
                activity_var.add_parent(model.get_variable(relation[0]))
                print("   ", relation[0], "->", relation[1])
                updated = True
        for parent in activity_var.get_conditional_parents():
            if parent.attr_name not in [r[0] for r in relations]:
                print("Removing", parent, "->", train_log.activity)
                activity_var.remove_parent(parent)
                updated = True
        if updated:
            model.train(train_log)
    return results


def cond_prob(a,b):
    """
    Return the conditional probability of a given b. With a and b sets containing row ids.
    """
    return len(set.intersection(a,b)) / len(b)


def get_probabilities(variable, val_tuple, parents):
    """
    Calculate probabilities for all possible next values for the given variable.
    """
    if variable.conditional_table.check_parent_combination(val_tuple):
        cpt_probs = variable.conditional_table.get_values(val_tuple)
        probs = {}
        for value in cpt_probs:
            # p_value = len(variable.value_counts[value]) / variable.total_rows
            # probs[value] = cpt_probs[value] * \
            #                (LAMBDA + (1 - LAMBDA) * (variable.conditional_table.cpt_probs[val_tuple] / p_value))
            probs[value] = cpt_probs[value]

        return probs, False
        # return variable.conditional_table.get_values(val_tuple), False
    else:
        unseen_value = False
        value_combinations = []
        known_attributes_indexes = None
        unseen_attribute_i = []
        for i in range(len(val_tuple)):
            if val_tuple[i] not in parents[i].value_counts:
                unseen_value = True
                value_combinations.append(parents[i].value_counts.keys())
                unseen_attribute_i.append(i)
            else:
                value_combinations.append([val_tuple[i]])
                if known_attributes_indexes is None:
                    known_attributes_indexes = parents[i].value_counts[val_tuple[i]]
                else:
                    known_attributes_indexes = set.intersection(known_attributes_indexes, parents[i].value_counts[val_tuple[i]])

        if unseen_value:
            return prob_unseen_value(variable, parents, known_attributes_indexes, unseen_attribute_i,value_combinations)
        else:
            return prob_unseen_combination(variable, val_tuple, parents)


def prob_unseen_combination(variable, val_tuple, parents):
    """
    Estimate the probabilities for the next value when current combination of values did not occur in the training data
    """
    predictions = {}
    for i in range(len(val_tuple)):
        values = [[v] for v in val_tuple]
        attr = parents[i]
        values[i] = attr.value_counts.keys()

        fixed_attrs = [parents[j].value_counts[val_tuple[j]] for j in range(len(val_tuple)) if j != i]
        if len(fixed_attrs) == 0:
            continue
        fixed_indexes = set.intersection(*fixed_attrs)


        product = list(itertools.product(*values))
        for combination in [c for c in product if variable.conditional_table.check_parent_combination(c)]:
        # for combination in product:
            variable_indexes = attr.value_counts[combination[i]]

            if len(fixed_indexes) == 0:
                parent_prob = 0
            else:
                parent_prob = cond_prob(variable_indexes, fixed_indexes)

            for value, prob in variable.conditional_table.get_values(combination).items():
                if value not in predictions:
                    predictions[value] = 0
                predictions[value] += prob * parent_prob
    for pred_val in predictions:
        predictions[pred_val] = predictions[pred_val] / len(parents)
    if len(predictions) > 0:
        return predictions, True
    else:
        return {0: 0}, True


def prob_unseen_value(variable, parents, known_attributes_indexes, unseen_attribute_i, value_combinations):
    """
    Estimate the probabilities for the next value when values that did not occur in the training data occur in the test data
    """
    predictions = {}
    for combination in variable.conditional_table.get_parent_combinations():
        valid_combination = True
        if type(combination) == type(0):
            combination = (combination,)
        for i in range(len(combination)):
            if combination[i] not in value_combinations[i]:
                valid_combination = False
                break
        if not valid_combination:
            continue

        unseen_attributes = [parents[i].value_counts[combination[i]] for i in unseen_attribute_i]
        unseen_indexes = set.intersection(*unseen_attributes)

        if known_attributes_indexes is None:
            parent_prob = len(unseen_indexes) / variable.total_rows
        elif len(known_attributes_indexes) == 0:
            continue
        else:
            parent_prob = cond_prob(unseen_indexes, known_attributes_indexes)

        if parent_prob > 0:
            for value, prob in variable.conditional_table.get_values(combination).items():
                if value not in predictions:
                    predictions[value] = 0
                predictions[value] += prob * parent_prob
    if len(predictions) > 0:
        return predictions, True
    else:
        return {0: 0}, True


def predict_next_event(edbn_model, log):
    """"
    Predict next activity for all rows in the logfile
    """
    # with mp.Pool(mp.cpu_count()) as p:
    #     result = p.map(functools.partial(predict_next_event_row, model=edbn_model, activity=log.activity), log.contextdata.iterrows())
    result = map(functools.partial(predict_next_event_row, model=edbn_model, activity=log.activity), log.contextdata.iterrows())

    result = [r for r in result if r != -1]

    return result


def predict_next_event_row(row, model, activity):
    """"
    Predict next activity for a single row
    """
    parents = model.variables[activity].conditional_table.parents

    value = []
    for parent in parents:
        value.append(getattr(row[1], parent.attr_name))
    tuple_val = tuple(value)

    activity_var = model.variables[activity]
    probs, unknown = get_probabilities(activity_var, tuple_val, parents)

    predicted_val = max(probs, key=lambda l: probs[l])

    return (getattr(row[1], activity), predicted_val, probs[predicted_val], probs.get(getattr(row[1], activity),0))

    # if getattr(row[1], activity) == predicted_val:
    #     return 1
    # else:
    #     return 0

def predict_next_event_multi_row(row, models, activity, bypass_unknown=False):
    """"
    Predict next activity for a single row
    """
    total_prob = {}
    for model in models:
        parents = model.variables[activity].conditional_table.parents

        value = []
        for parent in parents:
            value.append(getattr(row[1], parent.attr_name))
        tuple_val = tuple(value)

        activity_var = model.variables[activity]
        probs, unknown = get_probabilities(activity_var, tuple_val, parents)
        if not unknown or bypass_unknown:
            for prob in probs:
                if prob not in total_prob:
                    total_prob[prob] = []
                total_prob[prob].append(probs[prob])
    if len(total_prob) > 0:
        predicted_val = max(total_prob, key=lambda l: np.average(total_prob[l]))

        if getattr(row[1], activity) == predicted_val:
            return 1
        else:
            return 0
    else:
        return 0

def predict_next_event_update(edbn_model, log):
    parents = edbn_model.variables[log.activity].conditional_table.parents
    result = []

    for row in log.contextdata.iterrows():
        value = []
        for parent in parents:
            value.append(getattr(row[1], parent.attr_name))
        tuple_val = tuple(value)

        activity_var = edbn_model.variables[log.activity]
        probs, unknown = get_probabilities(activity_var, tuple_val, parents)

        predicted_val = max(probs, key=lambda l: probs[l])

        result.append((getattr(row[1], log.activity), predicted_val, probs[predicted_val]))

    edbn_model.update_log(log)

    return result

def predict_next_event_multi(edbn_models, log, bypass_unknown=False):
    """"
    Predict next activity for all rows in the logfile
    """
    # with mp.Pool(mp.cpu_count()) as p:
    #     result = p.map(functools.partial(predict_next_event_row, model=edbn_model, activity=log.activity), log.contextdata.iterrows())
    result = map(functools.partial(predict_next_event_multi_row, models=edbn_models, activity=log.activity,
                                   bypass_unknown=bypass_unknown), log.contextdata.iterrows())

    result = [r for r in result if r != -1]

    return np.average(result)

def predict_suffix(model, log):
    """
    Predict all suffixes for the entire logfile
    """
    all_parents, attributes = get_prediction_attributes(model, log.activity)

    predict_case_func = functools.partial(predict_suffix_case, all_parents=all_parents, attributes=attributes, model=model,
                                          end_event=log.convert_string2int(log.activity, "END"),
                                          activity_attr=log.activity, k=log.k)
    # with mp.Pool(mp.cpu_count()) as p:
    #     results = p.map(predict_case_func, log.get_cases())

    results = map(predict_case_func, log.get_cases())

    prefix_results = {}
    for result in results:
        for prefix_size in result:
            if prefix_size not in prefix_results:
                prefix_results[prefix_size] = []
            prefix_results[prefix_size].extend(result[prefix_size])

    all_sims = []
    for prefix in sorted(prefix_results.keys()):
        all_sims.extend(prefix_results[prefix])

    total_sim = np.average(all_sims)

    return total_sim


def predict_suffix_case(case, all_parents, attributes, model, end_event, activity_attr, k):
    """
    Predict all suffixes for a single case
    """
    prefix_results = {}
    case = case[1]
    case_events = case[activity_attr].values
    for prefix_size in range(1, case.shape[0]):  # Iterate over the different prefixes of the case
        # Create last known row (including known history, depending on k of the model)
        current_row = {}
        for iter_k in range(k, -1, -1):
            index = prefix_size - 1 - iter_k
            if index >= 0:
                row = []
                for attr in attributes:
                    row.append(getattr(case.iloc[index], attr))
                current_row[iter_k] = row
            else:
                current_row[iter_k] = [0] * len(attributes)

        predicted_rows, unknown_value = predict_case_suffix_loop_threshold(all_parents, attributes, current_row, model,activity_attr, end_event)

        # Get predicted trace
        predicted_events = [i[0] for i in predicted_rows if i[0] is not None]
        if prefix_size not in prefix_results:
            prefix_results[prefix_size] = []
        # Store similarity for predicted trace according to size of prefix
        prefix_results[prefix_size].append(1 - (
                    damerau_levenshtein_distance(predicted_events, case_events[prefix_size:]) / max(
                len(predicted_events), len(case_events[prefix_size:]))))
    return prefix_results


def predict_case_suffix_loop_threshold(all_parents, attributes, current_row, model, activity_attr, end_event):
    """
    Predict the suffix for a case, given the latest known row(s)
    Selecting values with highest probability + Only allowing a limited amount of repetition of a single event

    :param all_parents: detailed list of attributes
    :param attributes: ordered list of attributes
    :param current_row: current row, containg history
    :param model: eDBN model
    :param activity_attr: name of control flow attribute
    :param end_event: event indicating end of a trace
    :return: updated current_row
    """
    predicted_rows = []
    repeated_event = [None]
    unknown_value = False

    while current_row[0][0] != end_event and len(predicted_rows) < 100:  # The event attribute should always be the first attribute in the list
        current_row[2] = current_row[1]
        current_row[1] = current_row[0]
        current_row[0] = [None] * len(all_parents)
        # Predict value for every attribute
        for attr in attributes:
            value = []
            for parent_detail in all_parents[attr]:
                value.append(current_row[parent_detail["k"]][attributes.index(parent_detail["name"])])
            tuple_val = tuple(value)

            probs, unknown = get_probabilities(model.variables[attr], tuple_val, [v["variable"] for v in all_parents[attr]])

            if unknown:
                unknown_value = True

            if 0 not in probs:
                max_val = None
                max_prob = 0
                for val, prob in probs.items():
                    duplicate_threshold = model.duplicate_events.get(val, 1)

                    if (prob > max_prob and attr != activity_attr) or \
                            (prob > max_prob and attr == activity_attr and repeated_event[0] != val) or \
                            (prob > max_prob and attr == activity_attr and len(repeated_event) <= duplicate_threshold):
                        max_prob = prob
                        max_val = val

                current_row[0][attributes.index(attr)] = max_val
            else:
                current_row[0][attributes.index(activity_attr)] = end_event

        if current_row[0][0] == repeated_event[0]:
            repeated_event.append(current_row[0][0])
        else:
            repeated_event = [current_row[0][0]]

        predicted_rows.append(current_row[0][:])
    return predicted_rows, unknown_value


def get_prediction_attributes(model, activity_attribute):
    """
    Return lists containing attributes needed to predict in order to be able to predict the control flow

    :param model: eDBN model used
    :param activity_attribute:
    :return:
    """
    prev_pattern = re.compile(r"_Prev[0-9]*")
    all_parents = {}
    to_check = [activity_attribute]
    all_parents[activity_attribute] = model.variables[activity_attribute].conditional_table.parents[:]
    while len(to_check) > 0:
        attr = to_check[0]
        all_parents[attr] = model.variables[attr].conditional_table.parents[:]
        for parent in all_parents[attr]:
            current_attribute_version = prev_pattern.sub("", parent.attr_name)
            if current_attribute_version not in all_parents:
                to_check.append(current_attribute_version)
        to_check.remove(attr)

    for par_attr in all_parents:
        detailed_attributes = []
        for parent in all_parents[par_attr]:
            attr_details = {"name": prev_pattern.sub("", parent.attr_name), "variable": parent}
            if "Prev" in parent.attr_name:
                attr_details["k"] = int(re.sub(r".*_Prev", "", parent.attr_name)) + 1
            else:
                attr_details["k"] = 0
            detailed_attributes.append(attr_details)
        all_parents[par_attr] = detailed_attributes
    attributes = list(all_parents.keys())
    return all_parents, attributes


"""
Compute the Damerau-Levenshtein distance between two given
lists (s1 and s2)
From: https://www.guyrutenberg.com/2008/12/15/damerau-levenshtein-distance-in-python/
"""
def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]


def learn_duplicated_events(logfile):
    duplicated_events = {}
    for case_name, case in logfile.get_cases():
        trace = case[logfile.activity].values
        current_count = 1
        prev_event = trace[0]
        for event_idx in range(1, len(trace)):
            if trace[event_idx] == prev_event:
                current_count += 1
            else:
                if current_count > 1:
                    if prev_event not in duplicated_events:
                        duplicated_events[prev_event] = []
                    duplicated_events[prev_event].append(current_count)
                prev_event = trace[event_idx]
                current_count = 1
    avg_duplicated_events = {}
    for event in duplicated_events:
        avg_duplicated_events[event] = int(np.average(duplicated_events[event]) + 1)
    return avg_duplicated_events


def brier_multi(targets, probs):
    return np.mean(np.sum((probs - targets)**2))

