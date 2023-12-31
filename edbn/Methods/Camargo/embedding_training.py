# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:23:55 2018

@author: Manuel Camargo
"""
import itertools
import math
import os
import random

import numpy as np
from tensorflow.keras.layers import Input, Embedding, Dot, Reshape
from tensorflow.keras.models import Model

from Methods.Camargo.support_modules import support as sup


def training_model(log, outfile):
    """Main method of the embedding training module.
    """

    # Index creation
    # ac_index = {}
    # index_ac = {}
    # i = 1
    # for val in log.values[log.activity]:
    #     ac_index[val] = i
    #     index_ac[i] = val
    #     i += 1
    # ac_index["None"] = 0
    # index_ac[0] = "None"
    #
    # rl_index = {}
    # index_rl = {}
    # i = 1
    # for val in log.values["role"]:
    #     rl_index[val] = i
    #     index_rl[i] = val
    #     i += 1
    # rl_index["None"] = 0
    # index_rl[0] = "None"

    ac_index = create_index(log.data, 'event')
    ac_index['start'] = 0
    ac_index['end'] = len(ac_index)
    index_ac = {v: k for k, v in ac_index.items()}

    rl_index = create_index(log.data, 'role')
    rl_index['start'] = 0
    rl_index['end'] = len(rl_index)
    index_rl = {v: k for k, v in rl_index.items()}

    # Define the number of dimensions as the 4th root of the number of categories
    dim_number = math.ceil(len(list(itertools.product(*[list(ac_index.items()),
                                                        list(rl_index.items())])))**0.25)

    ac_weights, rl_weights = train_embedded(log.contextdata, len(log.values["event"]) + 1,
                                               len(log.values["role"]) + 1, dim_number)

    sup.create_file_from_list(reformat_matrix(ac_weights), os.path.join(outfile, 'event.emb'))
    sup.create_file_from_list(reformat_matrix(rl_weights), os.path.join(outfile, 'role.emb'))

    return reformat_matrix(ac_weights), reformat_matrix(rl_weights)


# =============================================================================
# Pre-processing: embedded dimension
# =============================================================================

def train_embedded(log_df, ac_index, rl_index, dim_number):
    """Carry out the training of the embeddings"""
    print("TRAIN")
    # Iterate through each book
    #pairs = list()
    #for i in range(0, len(log_df)):
        # Iterate through the links in the book
        # pairs.append((log_df.iloc[i]['task'], log_df.iloc[i]['role']))
        # pairs.append((ac_index[log_df.iloc[i]['event']], rl_index[log_df.iloc[i]['role']]))
    pairs = list(zip(log_df.event, log_df.role))

    # convert_pair_func = functools.partial(convert_pair, ac_index=ac_index, rl_index=rl_index)
    #
    # with mp.Pool(mp.cpu_count()) as p:
    #     pairs = p.map(convert_pair_func, pairs)
    print("Pairs generated")

    model = ac_rl_embedding_model(ac_index, rl_index, dim_number)
    model.summary()

    n_positive = 1024
    gen = generate_batch(pairs, n_positive, negative_ratio=2)
    # Train
    model.fit_generator(gen, epochs=100,
                        steps_per_epoch=len(pairs) // n_positive,
                        verbose=2)


    # Extract embeddings
    ac_layer = model.get_layer('activity_embedding')
    rl_layer = model.get_layer('role_embedding')

    ac_weights = ac_layer.get_weights()[0]
    rl_weights = rl_layer.get_weights()[0]

    return ac_weights, rl_weights

def convert_pair(pair, ac_index, rl_index):
    return (ac_index[pair[0]], rl_index[pair[1]])

def generate_batch(pairs, n_positive=50, negative_ratio=1.0):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    pairs_set = set(pairs)
    # This creates a generator
    while True:
        # randomly choose positive examples
        idx = 0
        for idx, (activity, role) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (activity, role, 1)
        # Increment idx by 1
        idx += 1

        # Add negative examples until reach batch size
        while idx < batch_size:
            # random selection
            random_ac = 0
            random_rl = 0

            # Check to make sure this is not a positive example
            if (random_ac, random_rl) not in pairs_set:

                # Add to batch and increment index, label 0 due classification task
                batch[idx, :] = (random_ac, random_rl, 0)
                idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'activity': batch[:, 0], 'role': batch[:, 1]}, batch[:, 2]


def ac_rl_embedding_model(ac_index, rl_index, embedding_size):
    """Model to embed activities and roles using the functional API"""

    # Both inputs are 1-dimensional
    activity = Input(name='activity', shape=[1])
    role = Input(name='role', shape=[1])

    # Embedding the activity (shape will be (None, 1, embedding_size))
    activity_embedding = Embedding(name='activity_embedding',
                                   input_dim=ac_index,
                                   output_dim=embedding_size)(activity)

    # Embedding the role (shape will be (None, 1, embedding_size))
    role_embedding = Embedding(name='role_embedding',
                               input_dim=rl_index,
                               output_dim=embedding_size)(role)

    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name='dot_product', normalize=True, axes=2)([activity_embedding, role_embedding])

    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape=[1])(merged)

    # Loss function is mean squared error
    model = Model(inputs=[activity, role], outputs=merged)
    model.compile(optimizer='Adam', loss='mse')

    return model

# =============================================================================
# Support
# =============================================================================
def reformat_matrix(weigths):
    """Reformating of the embedded matrix for exporting.
    Args:
        index: index of activities or roles.
        weigths: matrix of calculated coordinates.
    Returns:
        matrix with indexes.
    """
    matrix = list()
    for i in range(len(weigths)):
        data = weigths[i]
        matrix.append(data)
    return matrix

def create_index(log_df, column):
    """Creates an idx for a categorical attribute.
    Args:
        log_df: dataframe.
        column: column name.
    Returns:
        index of a categorical attribute pairs.
    """
    temp_list = log_df[[column]].values.tolist()
    subsec_set = {(x[0]) for x in temp_list}
    subsec_set = sorted(list(subsec_set))
    alias = dict()
    for i, _ in enumerate(subsec_set):
        alias[subsec_set[i]] = i + 1
    return alias
