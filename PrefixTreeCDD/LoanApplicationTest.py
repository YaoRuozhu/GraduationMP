import os
import sys

from pm4py.streaming.importer.xes import importer as stream_xes_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as stream_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

from PrefixTreeClass import PrefixTree
import settings
import time
import pandas as pd
import click as click
from collections import deque
from itertools import islice
from collections import OrderedDict
from CDD import Window
from skmultiflow.drift_detection import ADWIN, PageHinkley

@click.command()
@click.option('-l', '--decay_lambda', default=0.25, show_default=True, is_flag=False, flag_value=0.25, help='Decaying lambda for the older traces when performing a pruning step.')
@click.option('-n', '--noise', default=1, show_default=True, is_flag=False, flag_value=1, help='Noise filter applied to relations and activities which have a frequency below certain threshold.')
@click.option('-w', '--window_size', default=10, show_default=True, is_flag=False, flag_value=10, help='Maximum size of the window of detection (maxWindowSize)')
@click.option('-t', '--tree_size', default=1000, show_default=True, is_flag=False, flag_value=1000, help='Maximum size of the trees (pruningSteps)')
@click.option('-c', "--config", default=False, show_default=True, is_flag=False, flag_value=False, help='(True/False) Configuration for generating sub-logs from the drifts identified')
@click.option('-f', '--file', help='Path to the XES log file.')
def main(config, file, window_size, tree_size, decay_lambda, noise):
    """This is the Prefix-Tree Concept Drift Detection algorithm."""

    configuration = config

    settings.init()
    endEventsDic = dict()
    window = Window(initWinSize=window_size)
    filePath = file

    streaming_ev_object = stream_xes_importer.apply(os.path.abspath(filePath), variant=stream_xes_importer.Variants.XES_TRACE_STREAM)

    # Process the log Trace-by-Trace
    for trace in streaming_ev_object:
        lastEvent = trace[-1]["concept:name"]
        timeStamp = trace[-1]["time:timestamp"]
        caseID = trace.attributes["concept:name"]
        endEventsDic[caseID] = [lastEvent, timeStamp]

    caseList = [] # Complete list of cases seen
    Dcase = OrderedDict() # Dictionary of cases that we're tracking.

    tree = PrefixTree(pruningSteps = tree_size, noiseFilter=noise, lambdaDecay=decay_lambda) # Create the prefix tree with the first main node empty
    adwin = ADWIN()
    ph = PageHinkley()

    pruningCounter = 0 # Counter to check if pruning needs to be done
    traceCounter = 0  # Counter to create the Heuristics Miner model

    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True, variant.value.Parameters.TIMESTAMP_KEY: 'time:timestamp'}
    log = xes_importer.apply(os.path.abspath(filePath), variant=variant)
    static_event_stream = stream_converter.apply(log, variant=stream_converter.Variants.TO_EVENT_STREAM, parameters=parameters)
    static_event_stream._list.sort(key=lambda x: x['time:timestamp'], reverse=False)

    eventCounter = 0 # Counter for number of events
    currentNode = tree.root  # Start from the root node

    start_time = time.time()


    for ev in static_event_stream:
        caseList, Dcase, currentNode, pruningCounter, traceCounter, window = tree.insertByEvent(caseList, Dcase, currentNode, ev, pruningCounter, traceCounter, endEventsDic, window)
        eventCounter += 1

        if window.cddFlag:  # If a complete new tree has been created
            if len(window.prefixTreeList) == window.WinSize:  # Maximum size of window reached, start concept drift detection within the window
                window.conceptDriftDetection(adwin, ph)
                window.WinSize = min(window.WinSize + 1, window.maxWindowSize)

                if len(window.prefixTreeList) == window.WinSize:  # If there was no drift detected within the window
                    window.prefixTreeList = deque(islice(window.prefixTreeList, 1, None))  # Drop the oldest tree

    end_time = time.time()
    print(end_time - start_time)

    if configuration:
        logProvider(static_event_stream, window)  # Provide the sub-logs of the drifts

def as_dict(object):
    dictionaryObject = object.__dict__
    return dictionaryObject


def logProvider(log, window):
    """Function to provide the sub-logs of each drift"""
    for drift in window.driftsIdentified:
        W0start = drift.eventsSeen - drift.refWinSize
        W0end = drift.eventsSeen
        W1start = drift.eventsSeen
        W1end = drift.eventsSeen + drift.testWinSize

        W0SubLog = log[W0start:W0end]
        W1SubLog = log[W1start:W1end]

        if not os.path.exists("results"):
            os.makedirs("results")

        variables = W0SubLog[0].__dict__['_dict'].keys()

        df1 = pd.DataFrame([[i.get(j) for j in variables] for i in W0SubLog], columns = variables)
        df2 = pd.DataFrame([[i.get(j) for j in variables] for i in W1SubLog], columns = variables)

        indexOfDrift = str(window.driftsIdentified.index(drift))

        xes_exporter.apply(df1, os.path.abspath('results\\drift' + indexOfDrift + "W0.xes"))
        xes_exporter.apply(df2, os.path.abspath('results\\drift' + indexOfDrift + "W1.xes"))

if __name__ == "__main__":
    main(sys.argv[1:])