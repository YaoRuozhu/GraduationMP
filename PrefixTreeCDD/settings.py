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
