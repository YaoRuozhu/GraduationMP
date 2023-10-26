import PrefixTreeCDD.settings as settings
from collections import OrderedDict
import copy as cp
import uuid
import pandas as pd
import numpy as np
def init():
    global FAold, FBold, startActiv, endActiv, splitsDic, currentCaseList, traceTimestamps, completedCases
    FAold = dict()
    FBold = dict()
    startActiv = dict()
    endActiv = dict()
    splitsDic = dict()
    currentCaseList = [] # List of cases seen since the current model
    completedCases = [] # List of cases completed
    traceTimestamps = dict()

def reset():
    global FAold, FBold, startActiv, endActiv, splitsDic, currentCaseList
    FAold = dict() # Comment this line if you wish to have a complete process model.
    FBold = dict() # Comment this line if you wish to have a complete process model.
    startActiv = dict()
    endActiv = dict()
    splitsDic = dict()
    currentCaseList = []  # List of cases seen since the current model


# Class for every node of our tree
class TrieNode:
    # instances = pd.DataFrame(columns=['NodeId', 'Node', 'Activty', 'ParentNodeActivity', 'ParentNodeParentList'])  # class Dataframe to keep track of class instances

    def __init__(self, activity='root', parentNode=None):
        self.nodeId = uuid.uuid1()  # Random identifier for the node
        self.activity = str(activity)  # Activity name for the event stored in the node
        self.parent = parentNode  # Parent node/events
        self.parentList = []  # Create list of all parent nodes for this node
        self.fillParentList()  # Fill the list with the previous parents and the new parent
        self.children = dict()  # Children nodes/events are stored in a dictionary
        self.frequency = int()  # Frequency of the event
        if self.parent:
            self.branchId = ','.join(self.parentList) + "," + str(activity)

    # class method to access the get method without any instance
    @classmethod
    def get(cls, activity, parentNode):  # Get the node which has the same activity name and parents (branch)
        parentListStr = pd.DataFrame({'ParentNodeParentList': ''.join(parentNode.parentList)}, index=[0])
        parentListMerge = cls.instances.merge(parentListStr, left_on='ParentNodeParentList', right_on='ParentNodeParentList')
        foundNode = cls.instances.loc[(cls.instances['Activty'] == activity) & (cls.instances['ParentNodeActivity'] == parentNode.activity) & (cls.instances['NodeId'].reset_index(drop=True).sort_index(inplace=True) == parentListMerge['NodeId'].reset_index(drop=True).sort_index(inplace=True)), ['Node']]
        return foundNode

    def fillParentList(self):
        if self.parent:  # If the node has a parent node
            self.parentList = self.parent.parentList.copy()
            self.parentList.append(self.parent.activity)

    def pruneRoots(self, tree):
        tree.nodeFrequencies.setdefault(self.branchId, 0)
        tree.nodeFrequencies[self.branchId] += self.frequency  # Add frequency to FA (nodes)
        for event, node in self.children.items():
            if node:  # If there's children nodes and it's not the root node
                tree.relationFrequencies.setdefault((self.branchId, node.branchId), 0)
                tree.relationFrequencies[
                    (self.branchId, node.branchId)] += node.frequency  # Add frequency to FB (relations)
                node.pruneRoots(tree)  # Repeat the pruning for the children

    # Iterate over the tree searching for a node with the corresponding event ID
    def lookForNode(self, eventID, findNode):
        for event, node in self.children.items():
            if eventID == event: # If the event of the node is the same as the one we're searching for
                findNode = self
                return findNode
            else:
                return node.lookForNode(eventID, findNode)

        return None

# Class of the cases
class Case:
    def __init__(self, caseIdentifier, caseNode):
        self.caseId = caseIdentifier  # Case name
        self.node = caseNode # Node/Event to which the case is pointing to
        self.active = True # Case active/inactive flag

# Class of the root node
class PrefixTree:
    def __init__(self, pruningSteps, lambdaDecay, noiseFilter):
        self.reset()
        self.lambdaDecay = lambdaDecay
        self.pruningSteps = pruningSteps
        self.treePruneSteps = self.pruningSteps / 4
        self.Cmax = 6 # Max number of cases to track concurrently
        self.TPO = noiseFilter
        self.Tdep = 0.5
        self.Tbest = 3
        self.TAND = 0.5

    def reset(self):
        self.root = TrieNode()
        self.nodeFrequencies = dict()
        self.relationFrequencies = dict()
        self.startActFrequencies = dict()
        self.endActFrequencies = dict()
        self.treeNodes = []  # List which holds all the nodes of the trees
        self.eventsSeen = 0  # Number of events the tree has processed since t=0

        self.treeNodes.append(self.root)

    def resetFrequencies(self):
        self.nodeFrequencies = dict()
        self.relationFrequencies = dict()
        self.startActFrequencies = dict()
        self.endActFrequencies = dict()
        for node in self.treeNodes:
            node.frequency = 0

    # Function to insert a new node/event on the tree
    def insertByEvent(self, caseList, Dcase, current, ev, pruningCounter, traceCounter, endEventsDic, window):
        window.cddFlag = False  # We don't want CDD until we finish processing a complete tree
        # caseID = ev["case:concept:name"]  # Case ID from the trace attributes
        caseID = ev["case"]  # Case ID from the trace attributes
        # eventID = ev["concept:name"] # Event ID from the event attributes
        eventID = ev["event"]  # Event ID from the event attributes
        # eventTimestamp = ev["time:timestamp"]
        eventTimestamp = ev["completeTime"]
        if caseID not in caseList: # If it is a new case that has never been seen before
            current = self.root
            Dcase[caseID] = Case(caseID, current)  # Instance of the Case object for the dictionary
            Dcase.move_to_end(caseID, last=False)  # Move this case to the front (newest)
            if len(Dcase) > self.Cmax:  # If the size of the tracking dictionary is already the max size
                Dcase.popitem(last=True)  # Remove the last element from the tracking Case dictionary
            caseList.append(caseID) # Add case to the historic list
            if eventID not in self.startActFrequencies.keys():
                self.startActFrequencies[eventID] = 1
                if eventID not in current.children.keys():  # If the event is not one of the children events of that node
                    current.children[eventID] = TrieNode(eventID, current)  # New TrieNode instance (new event with parent)
                    current = current.children[eventID]  # Current node is the children event
                else:
                    current = current.children[eventID]  # Current node is the children event
                current.frequency += 1  # Increase frequency
                self.treeNodes.append(current)  # Append the new node to the node list of the tree
                self.startActFrequencies[eventID] = 1
                self.relationFrequencies[('START', current.branchId)] = 1  # Add frequency to FB (relations)
            else:
                self.startActFrequencies[eventID] += 1
                current = current.children[eventID]  # Current node is the children event
                self.relationFrequencies[('START', current.branchId)] += 1
            Dcase[caseID].startActivity = eventID
        elif caseID not in Dcase: # If the case has been seen before, but it's not one of the ones being tracked
            findNode = None  # Node of the event we want to find
            findNode = self.root.lookForNode(eventID, findNode)  # Look for the node of this event on the tree
            if findNode is None:  # If we can't find a node with this event
                current = self.root  # Consider it a new trace and start from the root
            else:
                current = findNode  # We take this node as the "closest" one to continue the trace
            Dcase[caseID] = Case(caseID, current)
            Dcase.move_to_end(caseID, last=False)  # Move this case to the front (newest)
            if len(Dcase) > self.Cmax:  # If the size of the tracking dictionary is already the max size
                Dcase.popitem(last=True)  # Remove the last element from the tracking Case dictionary
            if eventID not in current.children.keys():  # If the event is not one of the children events of that node
                current.children[eventID] = TrieNode(eventID, current)  # New TrieNode instance (new event with parent)
                current = current.children[eventID]  # Current node is the children event
            else:
                current = current.children[eventID]  # Current node is the children event
                current.frequency += 1  # Increase frequency
        else: # If the case has been seen before and is one of the ones being tracked
            Dcase.move_to_end(caseID, last=False) # Move this case to the front (newest)
            current = Dcase[caseID].node
            if eventID not in current.children.keys():  # If the event is not one of the children events of that node
                current.children[eventID] = TrieNode(eventID, current)  # New TrieNode instance (new event with parent)
                current = current.children[eventID]  # Current node is the children event
            else:
                current = current.children[eventID]  # Current node is the children event
                current.frequency += 1  # Increase frequency

        settings.currentCaseList.append(caseID)  # Add case to the list of cases seen for this period
        self.treeNodes.append(current) # Append the new node to the node list of the tree
        Dcase[caseID].node = current  # Case instance is pointing to the latest event
        settings.traceTimestamps[caseID] = eventTimestamp
        pruningCounter += 1
        if endEventsDic[caseID][0] == eventID and endEventsDic[caseID][1] == eventTimestamp:  # If last event of trace, set the case to inactive
            Dcase[caseID].active = False
            if eventID not in self.endActFrequencies.keys():
                self.endActFrequencies[eventID] = 1
                self.relationFrequencies[(current.branchId, 'END')] = 1
            else:
                self.endActFrequencies[eventID] += 1
                self.relationFrequencies.setdefault((current.branchId, 'END'), 0)
                self.relationFrequencies[(current.branchId, 'END')] += 1
            Dcase[caseID].endActivity = eventID
            settings.completedCases.append(caseID) # Add case to the completed cases list
            traceCounter += 1
            current = self.root

        # Frequency List Pruning
        if (pruningCounter % self.treePruneSteps == 0): # If there's a tree pruning step.
            self.pruneTree()  # Save the frequencies of the tree
            settings.FAold, settings.FBold = self.decayTree(settings.FAold,
                                                            settings.FBold)  # Decay the old frequencies and store new ones
            self.resetFrequencies()  # Only reset the frequencies of the tree. Leave the tree structure as is.

        # Tree Pruning
        if (pruningCounter % self.pruningSteps == 0):  # If pruning steps achieved
            self.nodeFrequencies = settings.FAold.copy()  # Update the frequencies of the nodes after the decaying
            self.relationFrequencies = settings.FBold.copy()  # Update the frequencies of the relations after the decaying
            self.eventsSeen = pruningCounter  # Save how many events were seen for this tree

            Dcase = self.cleanCases(Dcase, window) # Leave only active cases
            if not Dcase:  # If all cases are closed at this moment, start from the root
                current = self.root
            else:
                lastEventSeen = self.firstCaseElement(Dcase).node.activity # Get the last event seen from an active case
                current = self.root.children[lastEventSeen]
            if len(window.prefixTreeList) == window.maxWindowSize:
                del window.prefixTreeList[0]
            window.prefixTreeList.append(cp.deepcopy(self))  # Save the tree in our window
            window.cddFlag = True
            settings.reset()  # Reset the frequency lists
            self.reset()  # Reset the tree

        return caseList, Dcase, current, pruningCounter, traceCounter, window

    def pruneTree(self):
        for event, node in self.root.children.items():
            current = node # Current becomes the children
            current.pruneRoots(self) # Prune the roots of the tree

    def decayTree(self, FAold, FBold): # Decay the frequencies of the old nodes and add the new frequencies
        FAold2 = {k: v * (1 - self.lambdaDecay) for k, v in FAold.items()} # Decay for node list
        FBold2 = {k: v * (1 - self.lambdaDecay) for k, v in FBold.items()} # Decay for relations list
        for k, v in self.nodeFrequencies.items():
            FAold2.setdefault(k, 0)
            FAold2[k] += v # Add new frequencies for nodes
        for k, v in self.relationFrequencies.items():
            FBold2.setdefault(k, 0)
            FBold2[k] += v # Add new frequencies for relations
        return FAold2, FBold2 #, startActiv2, endActiv2

    # def cleanCases(self, caseList, Dcase):
    def cleanCases(self, Dcase, window):
        current = self.root
        activeCases = OrderedDict({caseID: caseObject for caseID, caseObject in Dcase.items() if caseObject.active == True})  # If it is an active case add it to the new case list
        for case, casObject in activeCases.items():
            eventID = casObject.node.activity
            if eventID not in current.children.keys(): # If the last activity is not already a children of the root
                current.children[eventID] = TrieNode(eventID, current) # Leave the last activity as nodes in the tree
            else:
                current = current.children[eventID]
                casObject.node = current  # Point the case to the existing node of the tree

        return activeCases

    def firstCaseElement(self, Dcase):
        '''Return the first element from an ordered collection
           or an arbitrary element from an unordered collection.
           Raise StopIteration if the collection is empty.
        '''
        return next(iter(Dcase.values()))