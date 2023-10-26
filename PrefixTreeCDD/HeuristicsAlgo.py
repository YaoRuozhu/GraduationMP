# Module for Heuristics Miner algorithm
from itertools import combinations
import PrefixTreeCDD.settings as settings
#from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

# Module for Model Evaluation metrics
#from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
#from pm4py.evaluation.precision import evaluator as precision_evaluator
# from pm4py.evaluation.generalization import evaluator as generalization_evaluator
# from pm4py.evaluation.simplicity import evaluator as simplicity_evaluator
import pandas

def directlyFollows(FBold, TPO):
    """
    Generates the directly-follows relation dictionary of events from the log by filtering those relations that have a
    frequency < than the threshold measure.

    Parameters
    --------------
    FBold - stores the pairs of activities from the event log
    TPO - dependency threshold for the relation |a > b|

    Returns
    --------------
    direcFoll - dictionary with the list of relations and their frequencies.
    """

    direcFoll2 = dict()

    direcFoll = {k: v for k, v in FBold.items() if v >= TPO}

    for k, v in FBold.items():
        if v >= TPO: # Formula (1) in paper
            direcFoll2[k] = v

    return direcFoll

def dependencyMeasure(direcFoll, Tdep):
    """
    Generates the conntection between a and b only if b is dependent on a

    Parameters
    --------------
    direcFoll - directly follows relations with frequencies
    Tdep - dependency threshold for dependency a => b

    Returns
    --------------
    dependencyRel - dictionary with dependencies and their frequencies.
    """

    dependencyRel = dict()

    for k, v in direcFoll.items():
        analogRel = (k[1], k[0])
        direcFoll.setdefault(analogRel, 0)
        dependencyValue = (v - direcFoll[analogRel])/(v + direcFoll[analogRel] + 1)
        if (dependencyValue) >= Tdep: # Formula (2) in paper
            dependencyRel[k] = v

        # Delete record from dictionary if it didn't exist before
        if direcFoll[analogRel] == 0:
            del direcFoll[analogRel]

    return dependencyRel

def sucessorFilter(dependencyRel, Tbest):
    """
    Generates the conntection between a and b only if b is dependent on a and there's no better b'. Same for a'.

    Parameters
    --------------
    dependencyRel - dictionary with dependency relations and their frequencies
    Tbest - dependency threshold for best connections

    Returns
    --------------
    bestRel - dictionary with the best dependencies after filter
    """

    postBest = dict()
    preBest = dict()

    for k, v in dependencyRel.items(): # Get the post best and pre best for each of the activities in the dependencies
        postBest.setdefault(k[0], 0)
        preBest.setdefault(k[1], 0)
        if v > postBest[k[0]]:
            postBest[k[0]] = v
        if v > preBest[k[1]]:
            preBest[k[1]] = v

        # Delete the records from the dictionary if they didn't exist before
        if postBest[k[0]] == 0:
            del postBest[k[0]]
        if preBest[k[1]] == 0:
            del preBest[k[1]]

    # for k, v in dependencyRel.items():
    bestRel = {k: v for k, v in dependencyRel.items() if (abs(postBest[k[0]] - v) < Tbest) and (abs(preBest[k[1]] - v) < Tbest)}  # Formula (4) and (5) in paper
        # if (postBest[k[0]] - v < Tbest) and (preBest[k[1]] - v < Tbest):
        #     bestRel[k] = v

    return bestRel

def splitsAndJoins(splitsDic, bestRel, TAND):
    """
    Evaluates the join conditions for relation a => (b^c) to identify AND/XOR splits.

    Parameters
    --------------
    bestRel - dictionary with top dependency relations and their frequencies
    TAND - threshold for AND/XOR splits

    Returns
    --------------
    splitsDic - dictionary with the relations and the corresponding split definition for each
    """

    for depRel,depFreq in bestRel.items():
        comparDic = {rel: freq for rel, freq in bestRel.items() if depRel[0] == rel[0]} # extract all the relations for a
        if len(comparDic) > 1: # if there exists two or more relations (a, something)
            for relation1, relation2 in combinations(comparDic, 2):
                # mainRels = list(comparDic.keys()) # Keys from the relations of 'a'
                splitRel1 = (relation1[1], relation2[1]) # tuple (b,c)
                splitRel2 = (relation2[1], relation1[1]) # tuple (c,b)

                bestRel.setdefault(splitRel1, 0) # if not existent assign value of 0 to key (b,c)
                bestRel.setdefault(splitRel2, 0) # if not existent assign value of 0 to key (c,b)
                splitMeasure = (bestRel[splitRel1] + bestRel[splitRel2]) / (comparDic[relation1] + comparDic[relation2] + 1)

                # Delete the records from the dictionary if they didn't exist before
                if bestRel[splitRel1] == 0:
                    del bestRel[splitRel1]
                if bestRel[splitRel2] == 0:
                    del bestRel[splitRel2]

                # Condition to identify if it is an AND or XOR split
                if splitMeasure >= TAND:
                    splitsDic[(relation1[0], (relation1[1], relation2[1]))] = "AND"
                else:
                    splitsDic[(relation1[0], (relation1[1], relation2[1]))] = "XOR"
                # print("hay que hacer el desmadre")

def initialEndMarkings(bestRel):
    """
    Generate artificial START and END events for those which don't have an incoming/outgoing relation.

    Parameters
    --------------
    bestRel - dictionary with top dependency relations and their frequencies

    Returns
    --------------
    bestRel - dictionary with the artificial START and END events
    """
    artificialRel = {k: v for k, v in bestRel.items()}
    startRelList = [k[0] for k in bestRel.keys()] # List of all the events which appear as the 'a' of relation a -> b
    endRelList = [k[1] for k in bestRel.keys()] # List of all the events which appear as the 'b' of relation a -> b

    for depRel,depFreq in bestRel.items():
        if depRel[1] != 'END' and depRel[1] not in startRelList: # If 'b' never appears as 'a' in another relation, then we set it as an "artificial" end event (i.e. there is no further relation after it)
            artificialRel[(depRel[1], 'END')] = depFreq
        if depRel[0] != 'START' and depRel[0] not in endRelList: # If 'a' never appears as 'b' in another relation, then we set it as an "artificial" start event (i.e. there is no previous relation)
            artificialRel[('START', depRel[0])] = depFreq

    return artificialRel

def modelNet(tree):
    """
    Generates a Petri Net and prints it by executing the Hueristics Miner over the DFG generated with the frequency lists

    Parameters
    --------------
    tree - The Prefix Tree which stores all events and frequencies
    """

    direcFoll = directlyFollows(settings.FBold, tree.TPO)

    # settings.startActiv = {k[1]: v for k, v in direcFoll.items() if 'START' in k}
    # settings.endActiv = {k[0]: v for k, v in direcFoll.items() if 'END' in k}

    dependencyRel = dependencyMeasure(direcFoll, tree.Tdep)
    bestRel = sucessorFilter(dependencyRel, tree.Tbest)
    artificialRel = initialEndMarkings(bestRel)
    splitsAndJoins(settings.splitsDic, direcFoll, tree.TAND)

    settings.startActiv = {k[1]: v for k, v in artificialRel.items() if 'START' in k}
    settings.endActiv = {k[0]: v for k, v in artificialRel.items() if 'END' in k}

    # artificialRel = {k: v for k, v in artificialRel.items() if 'END' not in k and 'START' not in k}
    direcFoll = {k: v for k, v in direcFoll.items() if 'END' not in k and 'START' not in k}

    heu_net, im, fm = heuristics_miner.apply_dfg(
        direcFoll, settings.FAold.keys(), settings.FAold, settings.startActiv, settings.endActiv, parameters={
        # artificialRel, settings.FAold.keys(), settings.FAold, settings.startActiv, settings.endActiv, parameters={
            # heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.1,
            # heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: 0.5,
            # heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_DFG_OCCURRENCES: 0.3,
            # heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_ACT_COUNT: tree.lambdaDecay}

            heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5,
            heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: 0.5,
            heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_DFG_OCCURRENCES: 0.5,
            heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_ACT_COUNT: tree.lambdaDecay}
    )

    gviz = pn_visualizer.apply(heu_net, im, fm)
    pn_visualizer.view(gviz)

    return heu_net, im, fm

def modelEvaluation(log, net, im, fm):
    """
    Evaluates Simplicity, Generalization, Precision, and Fitness of the Petri Net with the corresponding Log

    Parameters
    --------------
    log - Log to evaluate over the petri net
    net - Petri net of the process model
    im - Initial marking of the petri net
    fm - Final marking of the petri net

    Returns
    --------------
    simp = Simplicity value
    gen = Generalization value
    prec = Precision value
    fitness = Fitness value

    """

    simp = simplicity_evaluator.apply(net)
    gen = generalization_evaluator.apply(log, net, im, fm)
    prec = precision_evaluator.apply(log, net, im, fm, variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    fit = replay_fitness_evaluator.apply(log, net, im, fm, variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)["averageFitness"]

    return simp, gen, prec, fit

def fitnessEvaluation(log, net, im, fm):
    """
    Evaluates Fitness of the Petri Net with the corresponding Log

    Parameters
    --------------
    log - Log to evaluate over the petri net
    net - Petri net of the process model
    im - Initial marking of the petri net
    fm - Final marking of the petri net

    Returns
    --------------
    fit = Fitness value
    """

    fit = replay_fitness_evaluator.apply(log, net, im, fm, variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)["averageFitness"]

    return fit

def simplicityEvaluation(net):
    """
    Evaluates Simplicity, Generalization, Precision, and Fitness of the Petri Net with the corresponding Log

    Parameters
    --------------
    net - Petri net of the process model

    Returns
    --------------
    simp = Simplicity value
    """

    simp = simplicity_evaluator.apply(net)

    return simp