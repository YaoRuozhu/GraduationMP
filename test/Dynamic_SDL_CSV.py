import sys
sys.path.append('/Users/meruozhu/Downloads/MP_data/MP_codes/MP')

import time
import pandas as pd
from scipy.stats import percentileofscore
from Data.data import Data
from edbn.Utils.LogFile import LogFile
import edbn.Predictions.setting as setting
from edbn import Methods
from collections import deque
from itertools import islice
from collections import OrderedDict
from PrefixTreeCDD.PrefixTreeClass import PrefixTree
import PrefixTreeCDD.settings as settings
from PrefixTreeCDD.CDD import Window
from skmultiflow.drift_detection import ADWIN, PageHinkley
import math
from numpy import log as ln
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def recCurve(a, b, x):
    return a * math.log(x+1, 10)**(b * math.log(x+12, 10)) + 5000


def recCurve2(x):
    return -2e-5*x**2 + 1000


def recCurveFinal(x,a,u_max):
    if x == 0:
        return 1
    return 1 - math.exp(a*(x/u_max-1))


def store_results(file, results):
    with open(file, "w") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")


def store_resultsa(file, results):
    with open(file, "a") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")


def store_timings(file, timings):
    with open(file, "w") as fout:
        for t in timings:
            fout.write(str(t) + "\n")


def store_timing(file, timing):
    with open(file, "w") as fout:
        fout.write(str(timing) + "\n")


def store_timinga(file, timing):
    with open(file, "a") as fout:
        fout.write(str(timing) + "\n")


def run(file, dataName, updateInt, updateWindow, trainPerc=0.5):
    #a = 0.1
    delay_val = 490 #MUST BE LESS THAN U_MAX! has been 1250
    u_min = 200 #has been 200
    u_max = 500 #has been 1500
    t_min = 500 #has been 500
    t_max = 750 #has been 750
    a = ln(0.2) / ((delay_val /u_max) - 1)
    updateArray = range(u_min, u_max)
    trainArray = range(t_max, t_min, -1)
    #b = 2.235
    file = file
    # file = "Data/BPIC15_1_sorted_new.csv"
    # file = "Data/BPIC15_ALL.csv"
    # data = pd.read_csv("Data/BPIC15_1_sorted_new.csv", low_memory=False)
    data = pd.read_csv(file, low_memory=False)
    timeformat = "%Y-%m-%d %H:%M:%S"
    numEvents = data.shape[0]
    print("Num events is {}".format(numEvents))

    d = Data(dataName,
             LogFile(filename=file, delim=",", header=0, rows=None, time_attr="completeTime", trace_attr="case",
                     activity_attr='event', convert=False))
    d.logfile.keep_attributes(['event', 'role', 'completeTime'])
    m = Methods.get_prediction_method("SDL")
    s = setting.STANDARD
    trainPerc = trainPerc
    s.train_percentage = trainPerc * 100
    # # #
    d.prepare(s)
    # d.create_batch("normal", timeformat)
    is_written = 0

    start_time = time.time()
    # print(d.train.contextdata)
    print("Test Context Data")
    print(d.test_orig.contextdata)
    basic_model = m.train(d.train)

    print("Runtime %s:" % m, time.time() - start_time)
    res = m.test(basic_model, d.test_orig)

    store_results("results/%s_%s_normal.csv" % (m.name, d.name), res)
    #b = 'month'
    # d.create_batch(b, timeformat)

    print("Baseline Complete")
    updateInt = updateInt
    updateWindow = updateWindow
    window_size = 10
    tree_size = 1000
    decay_lambda = 0.25
    noise = 1
    settings.init()
    endEventsDic = dict()
    window = Window(initWinSize=window_size)

    lastEvents = data.groupby(['case']).last()
    for _, row in lastEvents.iterrows():
        endEventsDic[_] = [str(row['event']), row['completeTime']]

    caseList = []  # Complete list of cases seen
    Dcase = OrderedDict()  # Dictionary of cases that we're tracking.
    # print("You are here")
    tree = PrefixTree(pruningSteps=tree_size, noiseFilter=noise,
                      lambdaDecay=decay_lambda)  # Create the prefix tree with the first main node empty
    adwin = ADWIN()
    ph = PageHinkley()

    pruningCounter = 0  # Counter to check if pruning needs to be done
    traceCounter = 0  # Counter to create the Heuristics Miner model

    eventCounter = 0  # Counter for number of events
    currentNode = tree.root  # Start from the root node
    testInd = round(numEvents * trainPerc)
    prevDriftCounter = 0
    drifts = {}
    new_drifts = {}
    severity = -88
    start_time = time.time()

    for _, event in data.iterrows():
        # print(event)
        # break
        # need to implement ev into form accepted from event in line above
        caseList, Dcase, currentNode, pruningCounter, traceCounter, window = tree.insertByEvent(caseList, Dcase,
                                                                                                currentNode, event,
                                                                                                pruningCounter,
                                                                                                traceCounter,
                                                                                                endEventsDic, window)
        eventCounter += 1

        if window.cddFlag:  # If a complete new tree has been created
            if len(window.prefixTreeList) == window.WinSize:
                # Maximum size of window reached, start concept drift detection within the window
                temp_drifts = window.conceptDriftDetection(adwin, ph, eventCounter)
                window.WinSize = min(window.WinSize + 1, window.maxWindowSize)
                for i in temp_drifts.keys():
                    if i not in drifts.keys():
                        new_drifts[i] = temp_drifts[i]
                        # Retrain model or smth here with condition index > train count
                        # Need to update the testing batch so it tests all the way back to testInd
                        # if _ > round(numEvents * trainPerc):
                        #     print("Performing Drift Update")
                        #     result, timing, basic_model = m.test_and_update_indices(basic_model, d,
                        #                                                             (_ - round(numEvents * trainPerc),
                        #                                                              max(_ - round(
                        #                                                                  numEvents * trainPerc) - updateWindow,
                        #                                                                  0)),
                        #                                                             testInd - round(
                        #                                                                 numEvents * trainPerc),
                        #                                                             _ - round(numEvents * trainPerc),
                        #                                                             reset=True)
                        #     if is_written:
                        #         store_resultsa(
                        #             "results/%s_%s_OTF_drift_%s_%s.csv" % (m.name, d.name, updateInt, updateWindow),
                        #             result)
                        #         store_timinga(
                        #             "results/%s_%s_OTF_drift_%s_%s_time.csv" % (
                        #             m.name, d.name, updateInt, updateWindow),
                        #             timing)
                        #     else:
                        #         store_results(
                        #             "results/%s_%s_OTF_drift_%s_%s.csv" % (m.name, d.name, updateInt, updateWindow),
                        #             result)
                        #         store_timing(
                        #             "results/%s_%s_OTF_drift_%s_%s_time.csv" % (
                        #             m.name, d.name, updateInt, updateWindow),
                        #             timing)
                        #     is_written = 1
                        #     testInd = _
                        #     print("Model Drift updated at event {}".format(str(_)))

                if len(window.prefixTreeList) == window.WinSize:  # If there was no drift detected within the window
                    window.prefixTreeList = deque(islice(window.prefixTreeList, 1, None))  # Drop the oldest tree

        #TODO: when drift occurs or other trigger is activated
        if len(list(drifts.keys()) + list(new_drifts.keys())) > prevDriftCounter and (_ > round(numEvents * trainPerc)):
            if len(list(drifts.keys())) >= 5:
                #TODO: Take quantile of any new drifts that occur
                #list of drift severities
                drift_sevs = [drifts[i]['treeDist'] for i in list(drifts.keys())]
                new_drift_sevs = [new_drifts[i]['treeDist'] for i in list(new_drifts.keys())]
                severity = percentileofscore(a = drift_sevs, score = sum(new_drift_sevs)/len(new_drift_sevs))/100 #* 5000
                print("Severity is {}".format(severity))
                #calculate new severity using percentilesum(

            else:
                #TODO: max severity level
                severity = 1
                print("Max Severity level")
        elif (_ - testInd > updateInt) and (_ > round(numEvents * trainPerc)):
            #TODO: last drift greater than predetermined retrain frequency -> assign moderate severity
            if severity == -1 or severity == -88:
                print("Max update interval level")
                severity = 0.75
        elif severity == -88 and (_ > round(numEvents * trainPerc)) and len(list(drifts.keys()))>0:# last drift was < 5000 events ago....then update with the most recent available information
            if _ - sorted([i for i in drifts.keys()])[-1] < 5000:
                severity = 1


        #TODO: become aware of drift severity above then make appropriate update below^^^
        #TODO: drift updating starts here, use this to set the updateInt from previous iterations
        if severity != -1 and severity != -88 and ((_ - testInd > updateInt) or len(list(new_drifts.keys())) > 0):
            severity = min(1, severity)
            print(severity)

            #TODO: Update these values and create a list using rec_curve_final
            #severity outputs a score between 0 and 1
            #need to use the update and train array to actually convert to a meaningful update/train value
            print("Update val index value is {}".format(round(recCurveFinal(severity, a, u_max) * len(updateArray)) - 1))
            updateVal = min(updateArray[round(recCurveFinal(severity, a, u_max) * len(updateArray)) - 1], u_max)
            winVal = max(t_min, trainArray[round(recCurveFinal(severity, a, u_max) * len(trainArray)) - 1])
            print("Update Value is {}".format(updateVal))
            print("Window Value is {}".format(winVal))
            updateInt, updateWindow = updateVal, winVal
            #TODO: determine new values for updateInt and updateWindow before running below
            print("Performing Maintenance Update")
            print("Size of hyphen is {}".format(_))
            print("Highest index calculation on test is {}".format(_ - round(numEvents * trainPerc)))
            if updateWindow > _ - round(numEvents * trainPerc):
                updateWindow = 0
            result, timing, basic_model = m.test_and_update_indices(basic_model, d, (_ - round(numEvents * trainPerc),
                                                                                     max(_ - round(
                                                                                         numEvents * trainPerc) - updateWindow, #maximally the total number in the test set to retrain
                                                                                         0)),
                                                                    testInd - round(numEvents * trainPerc),
                                                                    _ - round(numEvents * trainPerc),
                                                                    reset=False)
            testInd = _
            print("Model updated at event {}".format(str(_)))
            if is_written:
                store_resultsa("results/%s_%s_OTF_drift_%s.csv" % (m.name, d.name, 'dynamic'), result)
                store_timinga("results/%s_%s_OTF_drift_%s_time.csv" % (m.name, d.name, 'dynamic'),
                              timing)
            else:
                store_results("results/%s_%s_OTF_drift_%s.csv" % (m.name, d.name, 'dynamic'), result)
                store_timing("results/%s_%s_OTF_drift_%s_time.csv" % (m.name, d.name, 'dynamic'),
                             timing)
            is_written = 1
            severity += 0.1

        #reset drifts for next run's analysis
        drifts = {**drifts, **new_drifts}
        prevDriftCounter += len(list(new_drifts.keys()))
        new_drifts = {}
        if severity >= 1:
            severity = -1




    print("Performing Final Test")
    # print("Size of hyphen is {}".format(_))
    # print("Highest index calculation on test is {}".format(_ - round(numEvents * trainPerc)))
    result, timing, basic_model = m.test_and_update_indices(basic_model, d, (_ - round(numEvents * trainPerc),
                                                                             max(_ - round(
                                                                                 numEvents * trainPerc) - updateWindow,
                                                                                 0)),
                                                            testInd - round(numEvents * trainPerc),
                                                            _ - round(numEvents * trainPerc),
                                                            reset=False)
    testInd = _

    print("Model updated at event {}".format(str(_)))
    if is_written:
        store_resultsa("results/%s_%s_OTF_drift_%s.csv" % (m.name, d.name, 'dynamic'), result)
        store_timinga("results/%s_%s_OTF_drift_%s_time.csv" % (m.name, d.name, 'dynamic'), timing)
    else:
        store_results("results/%s_%s_OTF_drift_%s.csv" % (m.name, d.name, 'dynamic'), result)
        store_timing("results/%s_%s_OTF_drift_%s_time.csv" % (m.name, d.name, 'dynamic'), timing)

    end_time = time.time()


if __name__ == "__main__":
    intervals = [1500] #has been 1500
    windows = [750] #has been 750
    files = ['Data/BPIC17_FULL.csv']#['Data/BPIC15_1_sorted_new.csv', 'Data/BPIC15_2_sorted_new.csv', 'Data/BPIC15_3_sorted_new.csv', 'Data/BPIC15_4_sorted_new.csv', 'Data/BPIC15_5_sorted_new.csv',
             # 'Data/Helpdesk.csv', 'Data/BPIC11.csv', 'Data/BPIC12.csv']

    #names = ['BPIC15_1_OTF_OPT']
    names = ['BPIC17FULL'] #, 'BPIC15_2_OTF_OPT', 'BPIC15_3_OTF_OPT', 'BPIC15_4_OTF_OPT', 'BPIC15_5_OTF_OPT',
              # 'Helpdesk_OTF_OPT', 'BPIC11_OTF_OPT', 'BPIC12_OTF_OPT']
    for i in intervals:
        for w in windows:
            for f in range(len(files)):
                if i < 1000 and w < 1000:
                    run(files[f], names[f], i, w)
                elif i >= 1000:
                    run(files[f], names[f], i, w)
