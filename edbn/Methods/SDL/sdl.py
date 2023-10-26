import numpy as np

from tensorflow.keras import utils as ku
from tensorflow.keras.layers import Embedding, Dense, Input, Concatenate, Softmax, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf


def learn_model(log, attributes, epochs, early_stop):
    num_activities = len(log.values[log.activity]) + 1
    # Input + Embedding layer for every attribute
    input_layers = []
    embedding_layers = []
    for attr in attributes:
        if attr not in log.ignoreHistoryAttributes and attr != log.time and attr != log.trace:
            for k in range(log.k):
                i = Input(shape=(1,), name=attr.replace(" ", "_").replace("(", "").replace(")","").replace(":","_") + "_Prev%i" % k)
                input_layers.append(i)
                # e = Embedding(len(log.values[attr]) + 1, 32, embeddings_initializer="zeros")(i)
                e = Embedding(len(log.values[attr]) + 5, len(log.values[attr]) + 5, embeddings_initializer="zeros")(i)
                embedding_layers.append(e)
    concat = Concatenate()(embedding_layers)

    drop = Dropout(0.2)(concat)
    dense2 = Dense(num_activities)(drop)

    flat = Flatten()(dense2)

    output = Softmax(name="output")(flat)

    model = Model(inputs=input_layers, outputs=[output])
    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999,
                epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    model.compile(loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
                  optimizer=opt)
    # model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop)

    outfile = 'tmp/model_{epoch:03d}-{val_loss:.2f}.h5'
    model_checkpoint = ModelCheckpoint(outfile,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto')

    x, y, vals = transform_data(log, [a for a in attributes if a != log.time and a != log.trace])
    if len(y) < 10:
        split = 0
    else:
        split = 0.2
    model.fit(x=x, y=y,
              validation_split=split,
              verbose=2,
              callbacks=[early_stopping],
              batch_size=32,
              epochs=epochs)
    return model


def transform_data(log, columns):
    num_activities = len(log.values[log.activity]) + 1

    col_num_vals = {}
    for col in columns:
        if col == log.activity:
            col_num_vals[col] = num_activities
        else:
            col_num_vals[col] = log.contextdata[col].max() + 2

    inputs = []
    for _ in range(len(columns) * log.k - len(log.ignoreHistoryAttributes) * log.k):
        inputs.append([])
    outputs = []
    for row in log.contextdata.iterrows():
        row = row[1]
        i = 0
        for attr in columns:
            if attr not in log.ignoreHistoryAttributes:
                for k in range(log.k):
                    inputs[i].append(row[attr + "_Prev%i" % k])
                    i += 1
        outputs.append(row[log.activity])

    outputs = ku.to_categorical(outputs, num_activities)
    for i in range(len(inputs)):
        inputs[i] = np.array(inputs[i])
    return inputs, outputs, col_num_vals


def train(log, epochs, early_stop):
    return learn_model(log, log.attributes(), epochs, early_stop)


# def update(model, log):
#     inputs, expected, _ = transform_data(log, [a for a in log.attributes() if a != log.time and a != log.trace])
#     print(inputs[0].shape)
#     print(expected[0].shape)
#     model.fit(inputs, y=expected,
#               validation_split=0,
#               batch_size=56,
#               verbose=0,
#               epochs=10)
#     return model

def update_old(model, log):
    inputs, expected, _ = transform_data(log, [a for a in log.attributes() if a != log.time and a != log.trace])
    
    batch_size = 32
    # Calculate the number of batches
    num_batches = len(expected) // batch_size
    print("num_batches:", num_batches)
    # Iterate over each batch
    for i in range(num_batches):
        # print("===========================================")
        # print("Batch:", i)

        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        # Check if it's the last batch and adjust the end index
        if i == num_batches - 1 and len(inputs) % batch_size != 0:
            end_idx = len(expected)
        
        # print("start_idx:", start_idx)
        # print("end_idx:", end_idx)
        # Extract the current batch
        batch_inputs = [_input[start_idx:end_idx] for _input in inputs]
        batch_expected = expected[start_idx:end_idx]

        
        # Train the model with the current batch
        model.fit(batch_inputs, y=batch_expected,
                  validation_split=0,
                  verbose=0,
                  batch_size=len(batch_expected),  # Adjust the batch size to fit the current batch
                  epochs=10)
    
    return model

def update_batch(model,log):
    batch_inputs_lst=[]
    batch_expected_lst = []

    loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    inputs, expected, _ = transform_data(log, [a for a in log.attributes() if a != log.time and a != log.trace])
    
    batch_size = 16
    # Calculate the number of batches
    num_batches = len(expected) // batch_size
    # print("num_batches:", num_batches)
    # Iterate over each batch
    for i in range(num_batches):
        # print("===========================================")
        # print("Batch:", i)

        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        # # Check if it's the last batch and adjust the end index
        # if i == num_batches - 1 and len(inputs) % batch_size != 0:
        #     end_idx = len(expected)
        
        batch_inputs = [_input[start_idx:end_idx] for _input in inputs]
        batch_expected = expected[start_idx:end_idx]
        
        # return batch_inputs,batch_expected
        batch_expected_lst.append(batch_expected)
        batch_inputs_lst.append(batch_inputs)
    
    # return batch_inputs_lst, batch_expected_lst
        
        # Train the model with the current batch
        for gs in range(10):
            with tf.GradientTape() as tape:
                y_pred = model([tf.convert_to_tensor(np.asarray(row).reshape(-1,1),dtype=tf.float32) for row in batch_inputs],training=True)
                y_sup = batch_expected
                recent_loss = loss_fn(y_sup, y_pred)
                total_loss = tf.reduce_sum(recent_loss)
                gradients = tape.gradient(total_loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return model

def update(model,log):

    loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    inputs, expected, _ = transform_data(log, [a for a in log.attributes() if a != log.time and a != log.trace])
    
    # batch_size = 16
    # Calculate the number of batches
    # num_batches = len(expected) // batch_size
    # print("num_batches:", num_batches)
    # Iterate over each batch
    # for i in range(num_batches):
    #     # print("===========================================")
    #     # print("Batch:", i)

    #     start_idx = i * batch_size
    #     end_idx = start_idx + batch_size
        
        # # Check if it's the last batch and adjust the end index
        # if i == num_batches - 1 and len(inputs) % batch_size != 0:
        #     end_idx = len(expected)
        
    # batch_inputs = [_input[start_idx:end_idx] for _input in inputs]
    # batch_expected = expected[start_idx:end_idx]
        
        # return batch_inputs,batch_expected
        # batch_expected_lst.append(batch_expected)
        # batch_inputs_lst.append(batch_inputs)
    
    # return batch_inputs_lst, batch_expected_lst
        
        # Train the model with the current batch
    for gs in range(20):
        with tf.GradientTape() as tape:
            y_pred = model([tf.convert_to_tensor(np.asarray(row).reshape(-1,1),dtype=tf.float32) for row in inputs],training=True)
            y_sup = expected
            recent_loss = loss_fn(y_sup, y_pred)
            total_loss = tf.reduce_sum(recent_loss)
            gradients = tape.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return model


def test(model, log):
    inputs, expected, _ = transform_data(log, [a for a in log.attributes() if a != log.time and a != log.trace])
    predictions = model.predict(inputs)
    predict_vals = np.argmax(predictions, axis=1)
    predict_probs = predictions[np.arange(predictions.shape[0]), predict_vals]
    expected_vals = np.argmax(expected, axis=1)
    expected_probs = predictions[np.arange(predictions.shape[0]), expected_vals]
    result = zip(expected_vals, predict_vals, predict_probs, expected_probs)
    # print accuracy
    accuracy = np.mean(predict_vals == expected_vals)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return result


def test_and_update(logs, model):
    results = []
    i = 0
    for t in logs:
        print(i, "/", len(logs))
        i += 1
        log = logs[t]["data"]
        results.extend(test(log, model))

        inputs, expected, _ = transform_data(log, [a for a in log.attributes() if a != log.time and a != log.trace])
        model.fit(inputs, y=expected,
                  validation_split=0,
                  verbose=0,
                  batch_size=1,
                  epochs=10)
    return results


def test_and_update_retain(test_logs, model, train_log):
    train_x, train_y, _ = transform_data(train_log, [a for a in train_log.attributes() if a != train_log.time and a != train_log.trace])

    results = []
    i = 0
    for t in test_logs:
        print(i, "/", len(test_logs))
        i += 1
        test_log = test_logs[t]["data"]
        results.extend(test(test_log, model))
        test_x, test_y, _ = transform_data(test_log, [a for a in test_log.attributes() if a != test_log.time and a != test_log.trace])
        for j in range(len(train_x)):
            train_x[j] = np.hstack([train_x[j], test_x[j]])
        train_y = np.concatenate((train_y, test_y))
        model.fit(train_x, y=train_y,
                  validation_split=0.2,
                  verbose=0,
                  batch_size=32,
                  epochs=1)
    return results


def test_and_update_full(test_log, model, train_logs):
    results = test(test_log, model)

    train = train_logs[0]
    train_x, train_y, _ = transform_data(train, [a for a in train.attributes() if a != train.time and a != train.trace])
    for t_idx in range(1, len(train_logs)):
        t = train_logs[t_idx]
        test_x, test_y, _ = transform_data(t, [a for a in t.attributes() if a != t.time and a != t.trace])
        for j in range(len(train_x)):
            train_x[j] = np.hstack([train_x[j], test_x[j]])
        train_y = np.concatenate((train_y, test_y))
    model.fit(train_x, y=train_y,
              validation_split=0.2,
              verbose=1,
              batch_size=32,
              epochs=1)
    return results


    # results = []
    # update_range = 1000
    # for i in range(0, len(test_x[0]), update_range):
    #     single_input = [el[i:i+update_range] for el in test_x]
    #     predictions = model.predict(single_input)
    #     predict_vals = np.argmax(predictions, axis=1)
    #     predict_probs = predictions[np.arange(predictions.shape[0]), predict_vals]
    #     if update_range == 1:
    #         expected_vals = np.argmax(test_y[i])
    #         results.append((expected_vals, predict_vals[0], predict_probs[0]))
    #     else:
    #         expected_vals = np.argmax(test_y[i:i+update_range], axis=1)
    #         results.extend(zip(expected_vals, predict_vals, predict_probs))
    #

    #
    #     model.fit(x=train_x, y=train_y,
    #               validation_split=0.2,
    #               verbose=0,
    #               batch_size=32,
    #               epochs=10)
    # return results

if __name__ == "__main__":
    import Predictions.setting
    import Predictions.setting as setting
    import Data
    import Methods
    from Predictions.metric import ACCURACY

    d = Data.get_data("BPIC12")
    m = Methods.get_prediction_method("SDL")
    s = setting.STANDARD
    d.prepare(s)

    basic_model = m.train(d.train)
    res = m.test(basic_model, d.test_orig)
    print("Acc:", ACCURACY.calculate(res))

