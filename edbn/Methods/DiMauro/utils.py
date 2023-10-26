"""
@authors: Di Mauro, Appice and Basile
"""
from datetime import datetime

import numpy as np


def load_data_new(log):
    X = []
    X_t = []
    y = []

    casestarttime = None
    lasteventtime = None

    for case in log.get_cases():
        case_df = case[1]
        for row in case_df.iterrows():
            row = row[1]
            if log.time is None:
                t_raw = 0
            else:
                t_raw = row[log.time + "_Prev%i" % (log.k-1)]
            if t_raw != 0:
                try:
                    t = datetime.strptime(t_raw, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    t = datetime.strptime(t_raw, "%Y/%m/%d %H:%M:%S.%f")
                lasteventtime = t
            line = []
            times = []
            for i in range(log.k - 1, -1, -1):
                line.append(row["event_Prev%i" % i])
                if log.time is None:
                    t_raw = 0
                else:
                    t_raw = row[log.time + "_Prev%i" % i]
                if t_raw != 0:
                    try:
                        t = datetime.strptime(t_raw, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        t = datetime.strptime(t_raw, "%Y/%m/%d %H:%M:%S.%f")
                    if lasteventtime is None:
                        times.append(1)
                    else:
                        timesincelastevent = t - lasteventtime
                        timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds + timesincelastevent.microseconds/1000000
                        if timediff + 1 <= 0:
                            times.append(1)
                        else:
                            times.append(timediff+1)
                    lasteventtime = t
                else:
                    times.append(1) #to avoid zero
            X.append(line)
            X_t.append(times)
            y.append(row["event"])

    X = np.array(X)
    X_t = np.array(X_t)
    y = np.array(y)

    X_t = np.log(X_t)

    return X, X_t, y


def load_data(train, test, case_index = 0, act_index = 1):

    import numpy as np
    import csv
    from keras.preprocessing.sequence import pad_sequences

    ACT_INDEX = act_index
    CASE_INDEX = case_index

    vocabulary = set()

    csvfile = open(train, 'r')
    trainreader = csv.reader(csvfile, delimiter=',')
    next(trainreader, None)  # skip the headers

    csvfile = open(test, 'r')
    testreader = csv.reader(csvfile, delimiter=',')
    next(testreader, None)  # skip the headers



    train_lines = [] #these are all the activity seq
    test_lines = []

    numcases = 0
    max_length = 0

    lastcase = ''
    firstLine = True
    for row in trainreader:
        if row[CASE_INDEX]!=lastcase:  #'lastcase' is to save the last executed case for the loop
            lastcase = row[CASE_INDEX]
            if not firstLine:
                line.append(-1)
                train_lines.append(line)
                if len(line) > max_length:
                    max_length = len(line)
            line = []
            numcases += 1

        vocabulary.add(row[ACT_INDEX])
        line.append(row[ACT_INDEX])
        firstLine = False
    line.append(-1)
    train_lines.append(line)

    lastcase = ''
    firstLine = True
    for row in testreader:
        if row[CASE_INDEX]!=lastcase:  #'lastcase' is to save the last executed case for the loop
            lastcase = row[CASE_INDEX]
            if not firstLine:
                line.append(-1)
                test_lines.append(line)
                if len(line) > max_length:
                    max_length = len(line)
            line = []
            numcases += 1

        vocabulary.add(row[ACT_INDEX])
        line.append(row[ACT_INDEX])
        firstLine = False
    line.append(-1)
    test_lines.append(line)

    vocabulary.add(-1)
    vocabulary = {key: idx for idx, key in enumerate(vocabulary)}
    vocabulary = {key: int(key) - 1 for key in vocabulary}

    numcases += 1
    print("Num cases: ", numcases)

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    max_length = 0
    prefix_sizes_train = []
    prefix_sizes_test = []
    seqs = 0
    vocab = set()
    for seq in train_lines:
        code = []
        code.append(vocabulary[seq[0]])

        vocab.add(seq[0])

        for i in range(1,len(seq)):
            prefix_sizes_train.append(len(code))

            if len(code)>max_length:
                max_length = len(code)

            X_train.append(code[:])
            y_train.append(vocabulary[seq[i]])

            code.append(vocabulary[seq[i]])
            seqs += 1

            vocab.add(seq[i])

        if len(code) > max_length:
            max_length = len(code)

    for seq in test_lines:
        code = []
        code.append(vocabulary[seq[0]])

        vocab.add(seq[0])

        for i in range(1,len(seq)):
            prefix_sizes_test.append(len(code))

            if len(code)>max_length:
                max_length = len(code)
            X_test.append(code[:])
            y_test.append(vocabulary[seq[i]])

            code.append(vocabulary[seq[i]])
            seqs += 1

            vocab.add(seq[i])
        if len(code) > max_length:
            max_length = len(code)

    prefix_sizes_train = np.array(prefix_sizes_train)
    prefix_sizes_test = np.array(prefix_sizes_test)

    print("Num sequences:", seqs)

    print("Activities: ",vocab )
    vocab_size = len(vocab)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    combined_y = np.concatenate((y_train, y_test))

    y_unique = np.unique(combined_y)
    dict_y = {}
    i = 0
    for el in y_unique:
        dict_y[el] = i
        i += 1
    for i in range(len(y_train)):
        y_train[i] = dict_y[y_train[i]]
    for i in range(len(y_test)):
        y_test[i] = dict_y[y_test[i]]
    y_unique = np.unique(combined_y, return_counts=True)
    print("Classes: ", y_unique)
    n_classes = y_unique[0].shape[0]
    print("Num classes:", n_classes)
    # padding

    padded_X_train = pad_sequences(X_train, maxlen=max_length, padding='pre', dtype='float64')
    padded_X_test = pad_sequences(X_test, maxlen=max_length, padding='pre', dtype='float64')

    return ( padded_X_train, y_train, padded_X_test, y_test, vocab_size, max_length, n_classes, prefix_sizes_train)


def load_cases_new(log):
    X = []
    X_t = []
    y = []

    casestarttime = None
    lasteventtime = None

    for case in log.get_cases():
        trace = case[1]

        trace_ac = list(trace["event"])

        j = 0
        for row in trace.iterrows():
            row = row[1]
            if log.time is None:
                t_raw = 0
            else:
                t_raw = row[log.time + "_Prev%i" % (log.k-1)]
            if t_raw != 0:
                try:
                    t = datetime.strptime(t_raw, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    t = datetime.strptime(t_raw, "%Y/%m/%d %H:%M:%S.%f")
                lasteventtime = t
            line = []
            times = []
            for i in range(log.k - 1, -1, -1):
                line.append(row["event_Prev%i" % i])
                if log.time is None:
                    t_raw = 0
                else:
                    t_raw = row[log.time + "_Prev%i" % i]
                if t_raw != 0:
                    try:
                        t = datetime.strptime(t_raw, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        t = datetime.strptime(t_raw, "%Y/%m/%d %H:%M:%S.%f")
                    if lasteventtime is None:
                        times.append(1)
                    else:
                        timesincelastevent = t - lasteventtime
                        timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds + timesincelastevent.microseconds/1000000
                        if timediff + 1 <= 0:
                            times.append(1)
                        else:
                            times.append(timediff+1)
                    lasteventtime = t
                else:
                    times.append(1) #to avoid zero
            X.append(line)
            X_t.append(times)
            y.append(trace_ac[j:])

            j += 1

    X = np.array(X)
    X_t = np.array(X_t)
    y = np.array(y)

    X_t = np.log(X_t)

    return X, X_t, y


def load_cases(train, test, case_index = 0, act_index = 1):
    import numpy as np
    import csv
    from keras.preprocessing.sequence import pad_sequences

    ACT_INDEX = act_index
    CASE_INDEX = case_index

    vocabulary = set()

    csvfile = open(train, 'r')
    trainreader = csv.reader(csvfile, delimiter=',')
    next(trainreader, None)  # skip the headers

    csvfile = open(test, 'r')
    testreader = csv.reader(csvfile, delimiter=',')
    next(testreader, None)  # skip the headers

    train_lines = [] #these are all the activity seq
    test_lines = []

    numcases = 0
    max_length = 0

    lastcase = ''
    firstLine = True
    for row in trainreader:
        if row[CASE_INDEX]!=lastcase:  #'lastcase' is to save the last executed case for the loop
            lastcase = row[CASE_INDEX]
            if not firstLine:
                line.append(-1)
                train_lines.append(line)
                if len(line) > max_length:
                    max_length = len(line)
            line = []
            numcases += 1

        vocabulary.add(row[ACT_INDEX])
        line.append(row[ACT_INDEX])
        firstLine = False
    line.append(-1)
    train_lines.append(line)

    lastcase = ''
    firstLine = True
    for row in testreader:
        if row[CASE_INDEX]!=lastcase:  #'lastcase' is to save the last executed case for the loop
            lastcase = row[CASE_INDEX]
            if not firstLine:
                line.append(-1)
                test_lines.append(line)
                if len(line) > max_length:
                    max_length = len(line)
            line = []
            numcases += 1

        vocabulary.add(row[ACT_INDEX])
        line.append(row[ACT_INDEX])
        firstLine = False
    line.append(-1)
    test_lines.append(line)

    vocabulary.add(-1)
    vocabulary = {key: idx for idx, key in enumerate(vocabulary)}
    vocabulary = {key: int(key) - 1 for key in vocabulary}

    numcases += 1
    print("Num cases: ", numcases)

    cases_train = []
    cases_test = []

    max_length = 0
    prefix_sizes_train = []
    prefix_sizes_test = []
    seqs = 0
    vocab = set()
    for seq in train_lines:
        code = []
        code.append(vocabulary[seq[0]])
        vocab.add(seq[0])

        if len(seq) > max_length:
            max_length = len(seq)

        for i in range(1,len(seq)):
            code.append(vocabulary[seq[i]])
            seqs += 1
            vocab.add(seq[i])
        cases_train.append(code)

    for seq in test_lines:
        code = []
        code.append(vocabulary[seq[0]])
        vocab.add(seq[0])

        if len(seq) > max_length:
            max_length = len(seq)

        for i in range(1,len(seq)):
            code.append(vocabulary[seq[i]])
            seqs += 1
            vocab.add(seq[i])
        cases_test.append(code)

    print("Num sequences:", seqs)

    print("Activities: ",vocab )
    vocab_size = len(vocab)

    cases_train = np.array(cases_train)

    print(max_length)

    test_cases_X = []
    test_cases_y = []
    for test_case in cases_test:
        for i in range(1, len(test_case)):
            test_cases_X.append(test_case[:i])
            test_cases_y.append(test_case[i:])

    test_cases_X = np.array(test_cases_X)

    padded_cases_train = pad_sequences(cases_train, maxlen=max_length, padding='pre', dtype='float64')
    padded_cases_test_X = pad_sequences(test_cases_X, maxlen=max_length, padding='pre', dtype='float64')

    return (padded_cases_train, padded_cases_test_X, test_cases_y, vocab_size, max_length, prefix_sizes_train)

if __name__ == "__main__":
    print("Testing load data")
    train = "../Data/PredictionData/helpdesk/train_log.csv"
    test = "../Data/PredictionData/helpdesk/test_log.csv"

    (X_train, y_train,
     X_test, y_test,
     vocab_size,
     max_length,
     n_classes,
     prefix_sizes) = load_data(train, test)

    (train_cases,
     test_cases,
     vocab_size,
     max_length,
     n_classes,
     prefix_sizes) = load_cases(train, test)
