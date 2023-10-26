import os
import random

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

import Methods.Taymouri.event_prediction as ep
import Methods.Taymouri.preparation as pr
from Utils.LogFile import LogFile

device=torch.device('cuda:0')



class adapted_Input(pr.Input):
    def run(self, log, batch_size, train):
        self.log = log
        self.prefix_len = log.k
        self.batch = batch_size
        self.dataset_name = log.filename.split("/")[-1].split('.')[0]
        self.mode = "event_prediction"
        self.path = os.getcwd() + "/" + self.dataset_name + '/' + "event_prediction" + '/prefix_' + str(self.prefix_len)

        # data = log.data.copy(True)
        # print("SIZE:", data.shape)
        # print("Types:", data.dtypes)
        # # changing the data type from integer to category
        # data['event'] = data['event'].astype('category')
        # print("Types after:", data.dtypes)
        #
        # dat_group = data.groupby('case')
        # print("Original data:", data.head())
        # print("Group by data:", dat_group.head())

        unique_event = list(range(1, len(log.values["event"]) + 1))
        self.unique_event = [0] + unique_event

        # self.design_matrix = self.design_matrix_creation(data)
        self.prefix_creating(self.mode)

        #Determining the train,test, and validation sets
        if train:
            self.train_inds = random.sample(range(self.design_matrix_padded.size()[0]), k=round(self.design_matrix_padded.size()[0] * .9))
            self.validation_inds = list(set(range(self.design_matrix_padded.size()[0])).difference(set(self.train_inds)))

            train_data = TensorDataset(self.design_matrix_padded[self.train_inds], self.y[self.train_inds])
            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

            validation_data = TensorDataset(self.design_matrix_padded[self.validation_inds], self.y[self.validation_inds])
            validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=True)

            self.train_loader = train_loader
            self.validation_loader = validation_loader
            self.test_loader = None
        else:
            self.test_inds = list(range(self.design_matrix_padded.size()[0]))

            test_data = TensorDataset(self.design_matrix_padded[self.test_inds], self.y[self.test_inds])
            test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

            self.train_loader = None
            self.validation_loader = None
            self.test_loader = test_loader

    def prefix_creating(self, mode='event_prediction'):
        tmp = []
        tmp_y = []
        for index, row in tqdm(self.log.contextdata.iterrows()):
            row_prefix = []
            if row["event_Prev0"] == 0:
                continue
            for i in range(self.log.k - 1, -1, -1):
                new_row = np.zeros(len(self.unique_event))
                new_row[row["event_Prev%i" % i]] = 1
                row_prefix.append(new_row)


            tmp.append(torch.tensor(row_prefix, dtype=torch.float, requires_grad=False))
            tmp_y.append(torch.tensor([row["event"]], dtype=torch.float, requires_grad=False))

        self.design_matrix_padded = pad_sequence(tmp, batch_first=True)
        self.y = pad_sequence(tmp_y, batch_first=True)

def create_input(train_log, test_log, batch_size):
    train_obj = adapted_Input()
    train_obj.run(train_log, batch_size, True)
    test_obj = adapted_Input()
    test_obj.run(test_log, batch_size, False)
    return train_obj, test_obj


def train(train_log, epoch=1, batch_size=5):
    train_data = adapted_Input()
    train_data.run(train_log, batch_size, True)

    # Initializing a generator
    selected_columns = train_data.unique_event
    print("Selected columns:", selected_columns)
    rnnG = ep.LSTMGenerator(seq_len=train_data.prefix_len, input_size=len(selected_columns), batch=train_data.batch,
                            hidden_size=2 * len(selected_columns), num_layers=2, num_directions=1)
    optimizerG = torch.optim.Adam(rnnG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Initializing a discriminator
    rnnD = ep.LSTMDiscriminator(seq_len=train_data.prefix_len + 1, input_size=len(selected_columns), batch=train_data.batch,
                                hidden_size=2 * len(selected_columns), num_layers=2, num_directions=1)
    optimizerD = torch.optim.Adam(rnnD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training and testing
    ep.train(rnnD=rnnD, rnnG=rnnG, optimizerD=optimizerD, optimizerG=optimizerG, obj=train_data, epoch=epoch)

    return rnnG

def test(model, test_log, batch_size=5):
    test_data = adapted_Input()
    test_data.run(test_log, batch_size, False)

    rnng_validation = torch.load(test_data.path + "/rnnG(validation).m")
    print("EVAL model")
    acc = ep.model_eval_test(modelG=model, mode='test', obj=test_data)
    print("EVAL model from validation")
    ep.model_eval_test(modelG=rnng_validation, mode='test', obj=test_data)
    return acc

if __name__ == "__main__":
    from data import Data
    import setting
    from metric import ACCURACY

    d = Data("Helpdesk", LogFile("../../Data/Helpdesk.csv", ",", 0, None, "completeTime", "case", activity_attr="event", convert=False))
    d.logfile.keep_attributes(["event", "role", "completeTime"])
    d.prepare(setting.STANDARD)

    r = test(train(d.train), d.test_orig)
    print("Accuracy:", ACCURACY.calculate(r))
