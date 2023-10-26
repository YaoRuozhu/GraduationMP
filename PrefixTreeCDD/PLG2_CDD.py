from telnetlib import Telnet
import xml.etree.ElementTree as ET
import settings
import pandas as pd
from collections import deque
from itertools import islice
from collections import OrderedDict
from PrefixTreeClass import PrefixTree
from CDD import Window
from skmultiflow.drift_detection import ADWIN, PageHinkley

settings.init()
endEventsDic = dict()
modelPeriod = 600 # Number of events for generating the Heuristics Miner model

caseList = [] # Complete list of cases seen
driftList = [] # List of drifts identified
Dcase = OrderedDict() # Dictionary of cases that we're tracking.
logDF = pd.DataFrame(columns = ["case:concept:name", "concept:name", "time:timestamp"])
metricsDF = pd.DataFrame(columns = ["Simplicity", "Generalization", "Precision", "Fitness"])
traceCounter = 0  # Counter to create the Heuristics Miner model
logCounter = 0 # Counter for the complete log of events
eventCounter = 0 # Counter for number of events per model


tree = PrefixTree() # Create the prefix tree with the first main node empty
currentNode = tree.root  # Start from the root node
adwin = ADWIN()
ph = PageHinkley()

# treeDetSubLog = PrefixTree() # Create the prefix tree for the detection sub-log with the first main node empty
# treeRefSubLog = PrefixTree() # Create the prefix tree for the reference sub-log with the first main node empty
# currentDetSubLogNode = treeDetSubLog.root  # Start from the root node
# currentRefSubLogNode = treeRefSubLog.root  # Start from the root node

window = Window() # Window used for this experiment
activeGTest = Gtest() # G-Test used for this experiment

pruningCounter = 0 # Counter to check if pruning needs to be done

with Telnet('localhost', 8888) as tn:
    while True:
        line = tn.read_until(b'</org.deckfour.xes.model.impl.XTraceImpl>')
        # print(line)
        stringLine = line.decode("utf-8")
        root = ET.fromstring(stringLine)
        ns = {'xes': 'http://www.xes-standard.org/'}
        for log in root:
            for trace in log.findall('xes:trace', ns):
                ev = dict()
                case = trace.find("xes:string", ns).attrib['value']
                # eventID = trace.find("xes:event//[@key='concept:name']", ns)
                eventID = trace.find("./xes:event//*[@key='concept:name']", ns).attrib['value']
                eventTimestamp = trace.find("./xes:event//*[@key='time:timestamp']", ns).attrib['value']
                # eventID = event.find("[@key='concept:name']").attrib['value']
                    # print(event.attrib)
                ev["case:concept:name"] = case
                ev["concept:name"] = eventID
                ev["time:timestamp"] = eventTimestamp
                logDF.loc[logCounter] = [case, eventID, eventTimestamp]
                # if eventID == 'Activity B':
                endEventsDic[case] = ['Activity U', eventTimestamp]
                # print(case, eventID, eventTimestamp)

                caseList, Dcase, currentNode, pruningCounter, traceCounter, window = tree.insertByEvent(caseList, Dcase,
                                                                                                        currentNode, ev,
                                                                                                        pruningCounter,
                                                                                                        traceCounter,
                                                                                                        endEventsDic,
                                                                                                        window)

                eventCounter += 1
                logCounter += 1

                if window.cddFlag:  # If a complete new tree has been created
                    # if len(window.prefixTreeList) >= 2:  # If our window is of size 2 or more, we can already perform CDD
                    if len(
                            window.prefixTreeList) == window.WinSize:  # Maximum size of window reached, start concept drift detection within the window
                        window.conceptDriftDetection(adwin, ph)
                        window.WinSize = min(window.WinSize + 1, window.maxWindowSize)

                        if len(
                                window.prefixTreeList) == window.WinSize:  # If there was no drift detected within the window
                            # window.prefixTreeList = deque(islice(window.prefixTreeList, window.WinSize - 1, None))  # Just keep the last tree as a representation of our population
                            window.prefixTreeList = deque(
                                islice(window.prefixTreeList, 1, None))  # Drop the oldest tree

