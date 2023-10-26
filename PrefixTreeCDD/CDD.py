import pandas as pd
from math import ceil
from itertools import tee, islice
from collections import deque
from PrefixTreeCDD.PrefixTreeClass import PrefixTree
from PrefixTreeCDD.HeuristicsAlgo import directlyFollows
# import rpy2.robjects.numpy2ri
# rpy2.robjects.numpy2ri.activate()
from PrefixTreeCDD.DDScripts import prefixTreeDistances, driftDetectionADWIM, driftDetectionPH

# Class for the Drifts
class Drift:
    def __init__(self, refWinSize, testWinSize, refTree, testTree, treeDistance, eventsSeen, criticalNodes):
        self.refWinSize = refWinSize
        self.testWinSize = testWinSize
        self.refTree = refTree
        self.testTree = testTree
        self.treeDistance = treeDistance
        self.eventsSeen = eventsSeen
        self.criticalNodes = criticalNodes

# Class for the Adaptive Window
class Window:
    def __init__(self, initWinSize, WinMax = 10000):
        self.maxWindowSize = initWinSize  # Number of trees that the window can hold
        self.adwinDict = dict()
        self.peltDict = dict()
        self.WinSize = initWinSize
        self.prefixTreeList = deque(maxlen=self.maxWindowSize)
        self.driftsIdentified = []
        self.cddFlag = False  # Flag used to trigger CDD once a new tree has been seen

    class subWindowTree:
        def __init__(self):
            self.winNodeFreq = []
            self.winRelFreq = []

    def newWindowSize(self, previosWindow, treeRefAold, treeRefBold, treeDetAold, treeDetBold):
        evolutionRatio = (len(treeDetAold) + len(treeDetBold)) / (len(treeRefAold) + len(treeRefBold))
        newWindow = ceil(previosWindow * evolutionRatio)

        return newWindow

    def conceptDriftDetection(self, adwin, ph, eventNum):
        drifts = {}
        indexSlider = 1
        while indexSlider < len(self.prefixTreeList):
            W0 = deque(islice(self.prefixTreeList, indexSlider))
            W1 = deque(islice(self.prefixTreeList, indexSlider, None))
            Window0, Window1 = self.buildContinMatrix(W0, W1)

            treeDistance = prefixTreeDistances(Window0, Window1)
            indexSlider = driftDetectionADWIM(adwin, treeDistance.treeDistanceMetric, self, indexSlider)

            if self.cddFlag:  # If a drift was detected
                referenceWinNumberOfEvents = len(W0) * self.prefixTreeList[0].pruningSteps
                testWinNumberOfEvents = len(W1) * self.prefixTreeList[0].pruningSteps
                eventsSeen = W0[-1].eventsSeen
                criticalNodes = [x for x in treeDistance.notInterDict if x[1] >= 300]
                drift = Drift(referenceWinNumberOfEvents, testWinNumberOfEvents, Window0, Window1, treeDistance,
                              eventsSeen, criticalNodes)
                self.driftsIdentified.append(drift)
                print("Drift detected at event index {}".format(eventNum))
                if eventNum not in drifts:
                    drifts[eventNum] = {'curEv':eventNum, 'detEv':drift.eventsSeen,
                                        'refWinSize': drift.refWinSize, 'testWinSize':drift.testWinSize,
                                        'treeDist':drift.treeDistance.treeDistanceMetric}
                print("ADWIN change detected at: " + str(drift.eventsSeen) + " events\n"
                                                                             "Reference window size: " + str(
                    drift.refWinSize) + "events\n"
                                        "Test window size: " + str(drift.testWinSize) + "events\n"
                                                                                        "Tree distance metric: " + str(
                    drift.treeDistance.treeDistanceMetric) + "\n"
                                                             "Critical nodes and relations... \n")

                for tupleNode in drift.criticalNodes:
                    if isinstance(tupleNode[0], tuple):
                        rel1 = tupleNode[0][0].split(",")[-1]
                        rel2 = tupleNode[0][1].split(",")[-1]
                        print("Relation: " + rel1 + " -> " + rel2)
                    else:
                        node1 = tupleNode[0].split(",")[-1]
                        print("Node: " + node1)
                print()

        self.cddFlag = False  # Finished with CDD
        return drifts

    def buildContinMatrix(self, W0, W1):  # Receive the windows containing the Prefix Trees
        W0Tree = PrefixTree(self.prefixTreeList[0].pruningSteps, self.prefixTreeList[0].lambdaDecay, self.prefixTreeList[0].TPO)
        W1Tree = PrefixTree(self.prefixTreeList[0].pruningSteps, self.prefixTreeList[0].lambdaDecay, self.prefixTreeList[0].TPO)

        W0TreeLists = self.subWindowTree()
        W1TreeLists = self.subWindowTree()

        for tree1 in W0:
            W0TreeLists.winNodeFreq.append(tree1.nodeFrequencies.copy())
            W0TreeLists.winRelFreq.append(tree1.relationFrequencies.copy())

        for tree2 in W1:
            W1TreeLists.winNodeFreq.append(tree2.nodeFrequencies.copy())
            W1TreeLists.winRelFreq.append(tree2.relationFrequencies.copy())

        W0Tree.nodeFrequencies = dict(pd.DataFrame(W0TreeLists.winNodeFreq).mean())
        W0Tree.relationFrequencies = dict(pd.DataFrame(W0TreeLists.winRelFreq).mean())

        W1Tree.nodeFrequencies = dict(pd.DataFrame(W1TreeLists.winNodeFreq).mean())
        W1Tree.relationFrequencies = dict(pd.DataFrame(W1TreeLists.winRelFreq).mean())

        Window0 = {**W0Tree.nodeFrequencies, **W0Tree.relationFrequencies}
        Window0 = directlyFollows(Window0, W0Tree.TPO)

        Window1 = {**W1Tree.nodeFrequencies, **W1Tree.relationFrequencies}
        Window1 = directlyFollows(Window1, W1Tree.TPO)

        return Window0, Window1

    def pairwise(self, iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)