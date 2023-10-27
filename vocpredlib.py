# Wordboost : STGCN experiments
# This library contains all experiment functions.

# A Roxburgh 2021-2023
# Ver 2.1.1

# imports:
import csv
import itertools
import os
import sys
import time
from pprint import pprint

# import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import mysql.connector
import numpy
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.preprocessing import MinMaxScaler
from stellargraph.layer import GCN_LSTM
from tensorflow import keras
from tensorflow.keras import Model


def load_uk_cdi_questions_lookup(filename):
    print("loading cdi_questions_lookup...")
    column_names = ["UID",
                    "word",
                    "category"]

    col_dtypes = {'UID': 'string',
                  'word': 'string',
                  'category': 'string'
                  }
    our_dataframe = pd.read_csv(
        filename,
        header=None,  # no heading row
        sep=",",  # separation mark
        names=[*column_names],
        dtype=col_dtypes,
        encoding="ISO-8859-1",
        engine='python',
        index_col='UID'
    )
    # convert to capitals
    for columns in our_dataframe.columns:
        our_dataframe['word'] = our_dataframe['word'].str.upper()

    return (our_dataframe)


def load_cdi_replace(filename):
    print("loading CDI Replace dictionary...")
    column_names = ["find", "replace"]
    col_dtypes = {'find': 'string',
                  'replace': 'string'}
    our_dataframe = pd.read_csv(
        filename,
        sep=",",  # separation mark
        header=None,  # no heading row
        dtype=col_dtypes,
        names=[*column_names],
        encoding="ISO-8859-1",
        engine='python'
    )
    return (our_dataframe)


def load_english_american(filename):
    print("loading UK English/American English dictionary...")
    column_names = ["find", "replace"]
    col_dtypes = {'find': 'string',
                  'replace': 'string'}
    our_dataframe = pd.read_csv(
        filename,
        sep=",",  # separation mark
        header=None,  # no heading row
        dtype=col_dtypes,
        names=[*column_names],
        encoding="ISO-8859-1",
        engine='python'
    )
    return (our_dataframe)



def replace_from_dictionary(input_frame, replace_frame, firstcol, secondcol='NONE'):
    # search and replace word labels based on dictionary file
    dict_lookup = dict(zip(replace_frame['find'], replace_frame['replace']))
    cue_mask = input_frame[firstcol].isin(dict_lookup.keys())
    input_frame.loc[cue_mask, firstcol] = input_frame.loc[cue_mask, firstcol].map(dict_lookup)

    if secondcol != 'NONE':
        target_mask = input_frame[secondcol].isin(dict_lookup.keys())
        input_frame.loc[target_mask, secondcol] = input_frame.loc[target_mask, secondcol].map(dict_lookup)

    return (input_frame)


def standardwords(input_frame, cdi_replace_frame, firstcol, secondcol):
    dict_lookup = dict(zip(cdi_replace_frame['find'], cdi_replace_frame['replace']))
    cue_mask = input_frame[firstcol].isin(dict_lookup.keys())
    input_frame.loc[cue_mask, firstcol] = input_frame.loc[cue_mask, firstcol].map(dict_lookup)
    target_mask = input_frame[secondcol].isin(dict_lookup.keys())
    input_frame.loc[target_mask, secondcol] = input_frame.loc[target_mask, secondcol].map(dict_lookup)
    return (input_frame)


def segment_wordbank_data_in_db(credentials, sequence_length, startid, inputtable, outputtable):
    # first we segment the data using a rolling window
    # Set up connection to MySQL server
    mydb = mysql.connector.connect(
        host=credentials['host'],
        user=credentials['username'],
        password=credentials['password'],
        database=credentials['database']
    )

    mycursor = mydb.cursor(dictionary=True)  # using dictionary=True to get rows as dicts

    def get_data_for_child(child_id):
        mycursor.execute(f"SELECT * FROM {inputtable} WHERE child_id = {child_id} ORDER BY data_id")
        return mycursor.fetchall()

    def insert_into_target(data):
        for row in data:
            # Exclude the 'id' column from the row data.
            row_data = {k: v for k, v in row.items() if k != 'id'}

            columns = ', '.join(row_data.keys())
            placeholders = ', '.join(['%s'] * len(row_data))
            query = f"INSERT INTO {outputtable} ({columns}) VALUES ({placeholders})"
            mycursor.execute(query, list(row_data.values()))
        mydb.commit()

    def process_child(child_id, next_new_child_id):
        data = get_data_for_child(child_id)
        unique_data_ids = list({row["data_id"] for row in data})

        # If unique data_ids are less than sequence_length, do nothing
        if len(unique_data_ids) < sequence_length:
            return next_new_child_id

        # If they're exactly sequence_length, insert as is
        if len(unique_data_ids) == sequence_length:
            for row in data:
                row["child_id"] = next_new_child_id  # Use the counter here
                row["wbcid"] = child_id  # Store the original child_id
            insert_into_target(data)
            return next_new_child_id + 1

        # If more than sequence_length, slide and insert
        for i in range(len(unique_data_ids) - sequence_length + 1):
            subset_data_ids = unique_data_ids[i:i + sequence_length]
            to_insert = [row for row in data if row["data_id"] in subset_data_ids]
            for row in to_insert:
                row["child_id"] = next_new_child_id  # Use the counter here
                row["wbcid"] = child_id  # Store the original child_id
            insert_into_target(to_insert)
            # increment counter after window is copied
            next_new_child_id = next_new_child_id + 1
        # Increment the counter after each child is processed
        return next_new_child_id + 1

    # Get all unique child_ids
    mycursor.execute("SELECT DISTINCT child_id FROM " + inputtable)
    all_child_ids = [row['child_id'] for row in mycursor.fetchall()]

    next_new_child_id = startid
    # Process each child_id
    for child_id in all_child_ids:
        print(child_id, next_new_child_id)
        next_new_child_id = process_child(child_id, next_new_child_id)

    return 'done'


def get_wordbank_wordlist_from_mysql(credentials, targettable):
    # Set up connection to MySQL server
    mydb = mysql.connector.connect(
        host=credentials['host'],
        user=credentials['username'],
        password=credentials['password'],
        database=credentials['database']
    )

    # Execute the query and store the result in a DataFrame
    query = "SELECT standardised AS word FROM " + targettable + " GROUP BY standardised;"
    wordsframe = pd.read_sql(query, mydb)

    # Close the connection
    mydb.close()
    return wordsframe



# STGCN

# print function
def fullprint(*args, **kwargs):
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)


# Train/test split
def train_test_split(data, train_portion):
    time_len = data.shape[1]
    train_size = int(time_len * train_portion)
    train_data = np.array(data.iloc[:, :train_size])
    test_data = np.array(data.iloc[:, train_size:])
    return train_data, test_data


# at a fixed position to split at the end of a n-observation series
def train_test_split_fixed(data, fixed_size):
    train_size = fixed_size
    train_data = np.array(data.iloc[:, :train_size])
    test_data = np.array(data.iloc[:, train_size:])
    return train_data, test_data


# Scaling
# Rescale the data from the original range so that all values are within the range of 0 and 1.
def scale_data(train_data, test_data):
    max_feature_value = train_data.max()
    print(max_feature_value)
    min_feature_value = train_data.min()
    print(min_feature_value)
    train_scaled = (train_data - min_feature_value) / (max_feature_value - min_feature_value)
    test_scaled = (test_data - min_feature_value) / (max_feature_value - min_feature_value)
    return train_scaled, test_scaled


# Sequence data preparation for LSTM
# We first need to prepare the data to be fed into an LSTM.
# The LSTM model learns a function that maps a sequence of past observations as input to an output observation.
# As such, the sequence of observations must be transformed into multiple examples from which the LSTM can learn.
def sequence_data_preparation(sequence_length, prediction_length, train_data, test_data):
    trainX, trainY, = [], []
    testX, testY = [], []

    # train_data is a 2d ndarray
    for i in range(train_data.shape[1] - int(sequence_length + prediction_length - 1)):
        # a = [everything, 0:10]
        # a = [everything, 1:11].. etc
        a = train_data[:, i: i + sequence_length + prediction_length]

        # trainX is a list of arrays of (sequence_length) size
        trainX.append(a[:, :sequence_length])
        trainY.append(a[:, -1])

    for i in range(test_data.shape[1] - int(sequence_length + prediction_length - 1)):
        b = test_data[:, i: i + sequence_length + prediction_length]
        testX.append(b[:, :sequence_length])
        testY.append(b[:, -1])

    trainX = np.array(trainX)
    trainY = np.array(trainY)

    testX = np.array(testX)
    testY = np.array(testY)

    return trainX, trainY, testX, testY


# function to execute stgcn test
def run_test(name, epoch, batchsize, sequence_length, prediction_length,
             edges_file, nodes_directory, survey_variety, edges_directory,
             testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel,
             savemodel, output_directory, log_directory):
    edges_files_folder = edges_directory

    def plot_convergence(history):
        # Plot training and validation loss to show convergence.
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(name + ': Model Convergence')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    col_dtypes = {'word': 'string',
                  'UID': 'string',
                  'ProbOfSaying': 'int',
                  'ProbOfUnderstanding': 'int'}

    nodes_dataframe = pd.DataFrame(columns=[*col_dtypes])

    # todo: make the nodes folder part of the function args
    nodes_directory = nodes_directory + "wbank4" + "\\"

    # Define a custom sort function that will retrieve the serial number as an integer
    def custom_sort(filename):
        return int(filename.split("_")[0])

    # Sort files using the custom sort function
    sorted_files = sorted(os.listdir(nodes_directory), key=custom_sort)

    # load nodes files
    # files are in nodes folder and labelled e.g. a_b_c.csv
    # where a = serial num, b = sequence num, c = survey num
    print('loading nodes files....')
    for filename in sorted_files:
        # replaced with sorted version
        f = os.path.join(nodes_directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            timeslice_id = filename.split("_")[0]
            if len(timeslice_id) == 1:
                timeslice_id = "000" + timeslice_id
            if len(timeslice_id) == 2:
                timeslice_id = "00" + timeslice_id
            if len(timeslice_id) == 3:
                timeslice_id = "0" + timeslice_id

            nodes_file = pd.read_csv(
                f,
                sep=",",  # separation mark
                dtype=col_dtypes,
                encoding="ISO-8859-1",
                engine='python'
            )

            # deconstruct the nodes file
            word_list = nodes_file['word'].to_list()
            pou_list = nodes_file['ProbOfUnderstanding'].to_list()
            pous_list = nodes_file['ProbOfSaying'].to_list()
            # create new column to hold all word comprehension levels
            new_list = pous_list.copy()

            length = len(pou_list)
            print(length)
            # populate the comprehension level column
            # Iterating the index
            # for each word, convert the knowledge rating to a comprehension level
            for i in range(length):
                if ((pou_list[i] == 1) & (pous_list[i] == 1)):
                    new_list[i] = 4.0
                if ((pou_list[i] == 1) & (pous_list[i] == 0)):
                    new_list[i] = 1.0
                if ((pou_list[i] == 0) & (pous_list[i] == 1)):
                    new_list[i] = 2.0
                if ((pou_list[i] == 0) & (pous_list[i] == 0)):
                    new_list[i] = 0

            nodes_file.drop("UID", axis=1, inplace=True)
            nodes_file.drop("ProbOfUnderstanding", axis=1, inplace=True)
            nodes_file.drop("ProbOfSaying", axis=1, inplace=True)
            print('timeslice: ' + str(timeslice_id))

            # insert comprehension column into nodes_file dataframe
            nodes_file[str(timeslice_id)] = new_list
            # insert comprehension column into nodes_dataframe
            nodes_dataframe[str(timeslice_id)] = new_list

    # insert data into vocab_data dataframe
    # drop unwanted columns
    nodes_dataframe.drop("UID", axis=1, inplace=True)
    nodes_dataframe.drop("ProbOfUnderstanding", axis=1, inplace=True)
    nodes_dataframe.drop("ProbOfSaying", axis=1, inplace=True)
    nodes_dataframe.drop("word", axis=1, inplace=True)
    vocab_data = nodes_dataframe
    vocab_data = vocab_data.sort_index(axis=1)

    # load edge file
    print('loading edges file')
    ecol_dtypes = {'source': 'string',
                   'target': 'string',
                   'weight': 'float'}
    edges_dataframe = pd.read_csv(
        edges_files_folder + edges_file,
        sep=",",  # separation mark
        header=0,  # no heading row
        dtype=ecol_dtypes,
        encoding="ISO-8859-1",
        engine='python'
    )

    # create weights_adjacency_matrix
    # nodes must be numbers in a sequential range starting at 0 - so this is the
    # number of nodes.
    # turn all node labels into a list (all sources and all targets)
    nodes = edges_dataframe.iloc[:, 0].tolist() + edges_dataframe.iloc[:, 1].tolist()

    # sort and dedupe the list
    nodes = list(k for k, g in itertools.groupby(sorted(nodes)))
    nodes = list(nodes)
    nodes = sorted(nodes)

    # insert node id
    nodes = [(i, nodes[i]) for i in range(len(nodes))]

    # replace node names with numbers in edge_dataframe
    for i in range(len(nodes)):
        edges_dataframe = edges_dataframe.replace(nodes[i][1], nodes[i][0])

    # halve the values of each weight to compensate for the doubling up that coo_matrix does (dedupe would be better here)
    edges_dataframe.iloc[:, 2] = np.true_divide(edges_dataframe.iloc[:, 2], 2)

    # create coordinate matrix
    M = coo_matrix((edges_dataframe.iloc[:, 2], (edges_dataframe.iloc[:, 0], edges_dataframe.iloc[:, 1])), shape=(len(nodes), len(nodes)))

    # convert to dense matrix
    M = M.todense()

    # transpose
    weights_adjacency_matrix = M + M.T
    n = weights_adjacency_matrix.shape[0]

    # make diagonals = 1
    weights_adjacency_matrix[range(n), range(n)] = 1
    print('weights_adjacency_matrix.shape:')
    print(weights_adjacency_matrix.shape)
    weights_check = weights_adjacency_matrix.shape[0]

    num_nodes, time_len = vocab_data.shape

    print("No. of words:", num_nodes, "\nNo. of timesteps:", time_len)

    if num_nodes != weights_check:
        print('warning! words in nodes file does not match total words in edges file!')

    train_data, test_data = train_test_split_fixed(vocab_data, testtrainsplitpoint)
    print("Train data: ", train_data.shape)
    print("Test data: ", test_data.shape)

    train_scaled, test_scaled = scale_data(train_data, test_data)

    print("sequence data prep")
    trainX, trainY, testX, testY = sequence_data_preparation(sequence_length, prediction_length, train_scaled, test_scaled)
    print(" Train data: ")
    print(trainX.shape)
    print(trainY.shape)
    print("Test data: ")
    print(testX.shape)
    print(testY.shape)

    print("Seq length: ", sequence_length)
    print("weights_adjacency_matrix:")
    print(weights_adjacency_matrix)
    print(np.info(weights_adjacency_matrix))

    # define model
    gcn_lstm = GCN_LSTM(
        seq_len=sequence_length,
        adj=weights_adjacency_matrix,
        gc_layer_sizes=[16, 10],
        gc_activations=["relu", "relu"],
        lstm_layer_sizes=[200, 200],
        lstm_activations=["tanh", "tanh"]
    )

    x_input, x_output = gcn_lstm.in_out_tensors()

    # should we be loading the model from disk?
    if loadmodel == 'True':
        print()
        print()
        print("loading model...")
        # history: any()
        # history = load_model(name, model)
        mname = name.replace(' ', '')
        model = keras.models.load_model(mname)

    else:
        model = Model(inputs=x_input, outputs=x_output)
        # compile model
        model.compile(optimizer=m_optimizer, loss=m_loss, metrics=m_metrics)

        print("fitting...")
        history: any()

        # start training
        history = model.fit(
            trainX,
            trainY,
            epochs=epoch,
            batch_size=batchsize,
            shuffle=False,
            verbose=0,
            validation_data=(testX, testY),
            use_multiprocessing=True, workers=8)

    # are we saving the model weights?
    if savemodel == 'True':
        # save_model(name, model, history.history)
        mname = name.replace(' ', '')
        mname = mname + '_' + time.strftime('%d%m%y%H%M%S')
        model.save(mname)
        # serialize model to YAML
        model_yaml = model.to_yaml()
        fname = ".\\saved_models\\" + name + ".yaml"
        with open(fname, "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        mfname = ".\\saved_models\\" + name + ".h5"

        model.save_weights(mfname)
        # np.save(hname, history)
        print("Saved model to disk")

    model.summary()

    print("Train loss: ", history.history["loss"][-1], "\nTest loss:", history.history["val_loss"][-1], )

    ythat = model.predict(trainX)
    yhat = model.predict(testX)

    # # Rescale values
    max_feature_value = train_data.max()
    print(max_feature_value)
    min_feature_value = train_data.min()
    print(min_feature_value)

    # actual train and test values
    train_rescref = np.array(trainY * max_feature_value)
    print(train_rescref)
    test_rescref = np.array(testY * max_feature_value)
    print(test_rescref)

    # Rescale model predicted values
    train_rescpred = np.array((ythat) * max_feature_value)
    test_rescpred = np.array((yhat) * max_feature_value)

    print(train_rescpred)
    print(test_rescpred)

    # Naive prediction benchmark (using previous observed value)
    testnpred = np.array(testX)[:, :, -1]  # picking the last feature_value of the 10 sequence for each segment in each sample
    testnpredc = testnpred * max_feature_value

    # Performance measures
    seg_mael = []
    seg_masel = []
    seg_nmael = []

    for j in range(testX.shape[-1]):
        seg_mael.append(np.mean(np.abs(test_rescref.T[j] - test_rescpred.T[j])))  # Mean Absolute Error for NN
        seg_nmael.append(np.mean(np.abs(test_rescref.T[j] - testnpredc.T[j])))  # Mean Absolute Error for naive prediction
        if seg_nmael[-1] != 0:
            seg_masel.append(seg_mael[-1] / seg_nmael[-1])  # Ratio of the two: Mean Absolute Scaled Error
        else:
            seg_masel.append(np.NaN)

    newplotting = True

    if newplotting == True:
        plot_convergence(history)

    # Prediction data output -----
    # Note: The interim accuracy calculation output to the command prompt / summary files here is not accurate - needs correcting.
    # actual (correct) accuracy in the reporting functions uses the raw data from the output files.

    # initialise
    average_accuracy = 0.0
    dataslices = 0
    actually_used = 0
    ds_range = len(test_rescpred)
    total_coarse_accuracy = 0

    # write to vocab logfile
    print("writing to file...")

    mname = name.replace(' ', '')

    # summary file
    sname = name.replace(' ', '')
    mname = output_directory + '' + mname + '_' + str(sequence_length) + 'x' + str(prediction_length) + 'x' + str(batchsize) + '.csv'
    sname = output_directory + '' + sname + '_summary_' + str(sequence_length) + 'x' + str(prediction_length) + 'x' + str(batchsize) + '.csv'
    # summary file
    if os.path.exists(sname):
        os.remove(sname)  # this deletes the file
    if os.path.exists(mname):
        os.remove(mname)  # this deletes the file
    with open(sname, "a+") as summary_file_object:
        # main output graphs file
        with open(mname, "a+") as file_object:
            # Move read cursor to the start of file.
            file_object.seek(0)
            file_object.write('Timeslice' + "," +
                              'Word' + "," +
                              'Truth' + "," +
                              'Prediction' + "," +
                              'Previous' + "," +
                              'Accuracy' + "," +
                              'RawPrediction')
            file_object.write("\n")

            summary_file_object.seek(0)
            summary_file_object.write('Timeslice' + "," +
                                      'CorrectNewwordsCoarse' + "," +
                                      'CorrectNewWordsFine' + "," +
                                      'TotalNewWords' + "," +
                                      'AccCoarse' + "," +
                                      'AccFine' + "," +
                                      time.strftime('%d%m%y %H%M%S')
                                      )
            summary_file_object.write("\n")

            # for each timeslice
            for test_data_timeslice in range(0, ds_range):
                wrd_index = 0

                prediction_order = list()

                # populate prediction_order
                for wrd in word_list:
                    # if first timeslice, don't do prediction
                    if test_data_timeslice == 0:
                        prediction_order.append([wrd, test_rescpred[test_data_timeslice, wrd_index], test_rescref[test_data_timeslice, wrd_index], 0, wrd_index])
                    else:
                        prediction_order.append([wrd, test_rescpred[test_data_timeslice, wrd_index], test_rescref[test_data_timeslice, wrd_index], test_rescref[test_data_timeslice - 1, wrd_index], wrd_index])
                    wrd_index += 1

                # sort prediction order
                # sort in place
                prediction_order.sort(key=lambda tup: tup[2], reverse=True)

                total_words = 0
                wrd_index = 0
                coarse_correct_new_prediction = 0
                correct_new_prediction = 0
                new_words = 0
                displaywrd = list()
                display_index = 1
                length_of_vocab_list = len(prediction_order)
                print('vocab list length=' + str(length_of_vocab_list))
                correct = 0
                incorrect = 0

                for currwrd in prediction_order:
                    gndtruth = currwrd[2]
                    theword = currwrd[0]
                    prediction = round(currwrd[1], 0)
                    previous = currwrd[3]

                    if not ((gndtruth == 0) and (prediction == 0)):
                        # if pred=0 and truth=0, 100%
                        acc = round(((prediction / gndtruth) * 100), 1)
                        if (gndtruth == 0) and (previous == 0):
                            acc = 0
                        if (gndtruth == 0) and (previous > 0):
                            acc = 0
                        file_object.write(str(test_data_timeslice) + "," +
                                          theword + "," +
                                          str(gndtruth) + "," +
                                          str(prediction) + "," +
                                          str(previous) + "," +
                                          str(acc) + "," +
                                          str(currwrd[1]))
                        file_object.write("\n")

                    # if truth > previous (i.e. a new word is predicted to have been 'learned' by the child)
                    if gndtruth > previous:
                        # new word prediction total increased
                        new_words += 1
                        # is this prediction correct?
                        if prediction == gndtruth:
                            correct_new_prediction += 1
                        if (abs(gndtruth - prediction) < 2):
                            coarse_correct_new_prediction += 1

                    if prediction == gndtruth:
                        correct += 1
                    else:
                        incorrect += 1

                    displaywrd.append(currwrd)
                    total_words += 1
                    wrd_index += 1
                    display_index += 1

                # compensate for sequence lengths
                if (dataslices == 0) | (dataslices % 4 != 0):
                    average_accuracy = average_accuracy + round(((correct / total_words) * 100), 2)
                    actually_used += 1
                else:
                    # do nothing - used as debug breakpoint
                    actually_used = actually_used

                print("-----------------------------------")
                print("Timeslice " + str(test_data_timeslice))

                print(str(total_words) + " words in lexicon")
                print(str(new_words) + " new words in vocabulary")

                print(str(incorrect) + " incorrectly predicted")
                print(str(correct_new_prediction) + " correct new word predictions")
                print(str(coarse_correct_new_prediction) + " correct new word predictions (coarse)")

                if new_words > 0:
                    summary_file_object.write(str(test_data_timeslice) + "," +
                                              str(coarse_correct_new_prediction) + "," +
                                              str(correct_new_prediction) + "," +
                                              str(new_words) + "," +
                                              str((coarse_correct_new_prediction / new_words) * 100) + "," +
                                              str((correct_new_prediction / new_words) * 100))
                else:
                    summary_file_object.write(str(test_data_timeslice) + "," +
                                              str(coarse_correct_new_prediction) + "," +
                                              str(correct_new_prediction) + "," +
                                              str(new_words) + "," +
                                              '0' + "," +
                                              '0')

                summary_file_object.write("\n")
                if new_words > 0:
                    total_coarse_accuracy = total_coarse_accuracy + ((coarse_correct_new_prediction / new_words) * 100)
                    print(total_coarse_accuracy)
                    print(coarse_correct_new_prediction)
                    print(new_words)

                average_accuracy = average_accuracy + round(((correct / total_words) * 100), 2)

                dataslices += 1

                if new_words > 0:
                    average_nw_accuracy = str(round((correct_new_prediction / new_words) * 100, 2)) + "%"
                    print("Average New Word Accuracy = " + average_nw_accuracy)
                    average_nw_accuracy = str(round((coarse_correct_new_prediction / new_words) * 100, 2)) + "%"
                    print("Average New Word Accuracy (coarse) = " + average_nw_accuracy)

    average_accuracy = round(total_coarse_accuracy / test_data_timeslice, 2)
    print("========")
    print(str(average_accuracy) + " / " + str(actually_used))
    print("Average Vocab Accuracy = " + str(average_accuracy))
    print()
    print("epoch:" + str(epoch) + ", batchsize=" + str(batchsize) + ", seq=" + str(sequence_length) + ", pred="
          + str(prediction_length))
    print()
    return average_accuracy


def evaluate_model(name, nodes_directory, output_directory, logfile, edges_file, sequence_length,
                   prediction_length, batchsize, epoch, survey_type, edges_folder, testtrainsplitpoint,
                   m_optimizer, m_loss, m_metrics, loadmodel, savemodel, log_directory):
    print("running test..." + name)

    print(name, nodes_directory, output_directory, logfile, edges_file, sequence_length, prediction_length, batchsize, epoch, survey_type, edges_folder)
    output_directory = output_directory + str(epoch) + '\\'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    accur = run_test(name, epoch, batchsize, sequence_length, prediction_length, edges_file, nodes_directory, survey_type, edges_folder,
                     testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel, output_directory, log_directory)

    # Open the file in append & read mode ('a+')
    print("writing to logfile...")
    with open(log_directory + '\\' + logfile, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(time.strftime(
            '%d/%m/%Y %H:%M:%S') + "," +
                          name + "," +
                          survey_type + "," +
                          edges_folder + "," +
                          edges_file + "," +
                          str(accur) + "," +
                          str(epoch) + "," +
                          str(batchsize) + "," +
                          str(sequence_length) + "," +
                          str(prediction_length))

    print("evaluation finished..." + name)  # Ensembler


def left(s, n):
    return s[:n]


# Store Model outputs to mysql db for subsequent analysis
def import_stgcn_results_into_mysql(credentials, import_folder, model, table):
    # Set up connection to MySQL server
    mydb = mysql.connector.connect(
        host=credentials['host'],
        user=credentials['username'],
        password=credentials['password'],
        database=credentials['database']
    )

    mycursor = mydb.cursor()

    # Set up the name of the folder containing the CSV files
    folder_name = import_folder
    # Loop through each file in the folder
    for filename in os.listdir(folder_name):
        if (model == 'ALL') or (filename.split('_')[0] == model):
            if (filename.endswith(".csv") and not ("summary" in filename)):
                # print(filename)
                # Timeslice,Word,Truth,Prediction,Previous,Accuracy,RawPrediction
                f = os.path.join(folder_name, filename)
                with open(f, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    line_count = 0
                    for row in csv_reader:
                        if line_count == 0:
                            print(f'Column names are {", ".join(row)}')
                            line_count += 1

                        timeslice = row["Timeslice"]
                        word = row["Word"]
                        truth = row["Truth"]
                        prediction = row["Prediction"]
                        previous = row["Previous"]
                        raw_prediction = row["RawPrediction"]

                        line_count += 1
                        fn2 = filename.split('_')[1]
                        sq = fn2.split('x')[0]
                        pl = fn2.split('x')[1]
                        pbs = fn2.split('x')[2]
                        bs = pbs.split('.')[0]
                        ep = os.path.basename(os.path.dirname(folder_name))
                        config = str(sq) + 'x' + str(pl) + 'x' + str(bs) + 'x' + str(ep)
                        version = time.strftime('%d%m%Y%H%M')

                        # Extract the dataset name from the filename
                        dataset = filename.split('_')[0]
                        dg_group = dataset
                        if left(dataset, 9) == 'Lancaster':
                            dg_group = "Lancaster"
                        if left(dataset, 7) == 'Glasgow':
                            dg_group = "Glasgow"
                        if dataset == 'Mcrae':
                            dg_group = dataset
                        if dataset == 'Buchanan':
                            dg_group = dataset
                        if dataset == 'Nelson':
                            dg_group = dataset
                        if dataset == 'Rhyming':
                            dg_group == dataset

                        sql = "INSERT INTO " + table + " (id, timeslice_id, word, truth, prediction, `previous`, rawprediction, dataset, config, version, dataset_group) " \
                                                       "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                        val = (0, timeslice, word, truth, prediction, previous, raw_prediction, dataset, config, version, dg_group)

                        mycursor.execute(sql, val)

                # Commit the changes to the MySQL database
                mydb.commit()

    # Close the connection to the MySQL server
    mydb.close()
    return "results imported into db."


def get_weighted_average(predictions, model, ignoremodels, biasfactor):
    # reset weights
    weights = {'Buchanan': 1, 'LancasterAuditory': 1, 'LancasterFootLeg': 1, 'LancasterGustatory': 1,
               'LancasterHandArm': 1, 'LancasterHaptic': 1, 'LancasterHead': 1, 'LancasterInteroceptive': 1,
               'LancasterMouth': 1, 'LancasterOlfactory': 1, 'LancasterTorso': 1, 'LancasterVisual': 1,
               'Mcrae': 1, 'Nelson': 1, 'Rhyming': 1, 'GlasgowArousal': 1, 'GlasgowConcreteness': 1,
               'GlasgowDominance': 1, 'GlasgowFamiliarity': 1, 'GlasgowGender': 1, 'GlasgowImageability': 1,
               'GlasgowValence': 1, 'GlasgowSize': 1, 'GlasgowAOA': 1, model: biasfactor}

    # Calculate the weighted average
    weighted_sum = 0
    total_weight = 0
    for key in predictions.keys():
        # as long as we're not ignoring a model, calculate av weight
        if key not in ignoremodels:
            weighted_sum += predictions[key] * weights[key]
            # This takes into account the bias factor
            total_weight += weights[key]

    weighted_average = round(weighted_sum / total_weight, 0)

    return weighted_average


def reduced_ensemble(credentials, tablename, outputtable, total_timeslices, version, ignoremodels, biasfactor):
    # calculate ensembles from the model results table
    # input: out_stgcn output: new_out_ensemble

    mydb = mysql.connector.connect(
        host=credentials['host'],
        user=credentials['username'],
        password=credentials['password'],
        database=credentials['database']
    )
    cursor = mydb.cursor()
    print('querying database..')
    print("ensembling:")

    # Set model list
    # models = ['Buchanan',
    #           'LancasterAuditory',
    #           'LancasterFootLeg',
    #           'LancasterGustatory',
    #           'LancasterHandArm',
    #           'LancasterHaptic',
    #           'LancasterHead',
    #           'LancasterInteroceptive',
    #           'LancasterMouth',
    #           'LancasterOlfactory',
    #           'LancasterTorso',
    #           'LancasterVisual',
    #           'Mcrae',
    #           'Nelson',
    #           'Rhyming',
    #           'GlasgowArousal',
    #           'GlasgowConcreteness',
    #           'GlasgowDominance',
    #           'GlasgowFamiliarity',
    #           'GlasgowGender',
    #           'GlasgowImageability',
    #           'GlasgowValence',
    #           'GlasgowSize',
    #           'GlasgowAOA']

    models = ['LancasterAuditory',
              'LancasterTorso',
              'Mcrae',
              'Nelson',
              'GlasgowDominance',
              'GlasgowImageability',
              'GlasgowAOA']

    # Define a helper function to write chunks to the DB
    def write_to_db(values):
        query = "INSERT INTO " + outputtable + " (timeslice_id, word, truth, prediction, previous, accuracy, " \
                                               "dataset, config, version, dataset_group) " \
                                               "VALUES (%s, %s, %s, %s, " \
                                               "%s, %s, %s, %s, %s, %s)"
        cursor.executemany(query, values)
        mydb.commit()
        print('writing..')

    # Define a constant for the data chunk size to be written to the db
    CHUNK_SIZE = 1000

    # Loop through each timeslice
    for timeslice in range(total_timeslices):
        # Get a list of unique words in the table
        cursor.execute("SELECT word FROM " + tablename + " WHERE timeslice_id=" + str(timeslice) + " GROUP BY word")
        wordlist = cursor.fetchall()
        print("Timeslice: " + str(timeslice))
        values = []

        # Record the start time of the loop
        start_time = time.time()

        # Loop through each word in wordlist
        for word in wordlist:
            # Select all rows in the table for the current timeslice and word
            query = "SELECT * FROM " + tablename + " WHERE timeslice_id=%s AND word=%s"
            vals = (timeslice, word[0])
            cursor.execute(query, vals)
            results = cursor.fetchall()
            num_rows = cursor.rowcount

            weighted_averages = {}
            # If there are rows returned, calculate predictions for the word
            if num_rows != 0:
                # Convert the results to a numpy array
                array_results = np.array(results)
                # Copy the column containing the sentiment scores
                column_copy = array_results[:, 4].copy()
                dataset_names = array_results[:, 8].copy()
                # Convert the sentiment scores to integers
                column_copy = column_copy.astype(int)

                dict_data = {}
                # clear predictions
                predictions = {'LancasterAuditory': 0, 'LancasterTorso': 0,
                               'Mcrae': 0, 'Nelson': 0, 'GlasgowDominance': 0, 'GlasgowImageability': 0, 'GlasgowAOA': 0}
                # get all model predictions for word in timeslice
                for i in range(len(array_results)):
                    row_data = array_results[i]
                    sentiment_score = int(row_data[4])
                    dataset_name = row_data[7]
                    # predictions[dataset_name] = sentiment_score
                    predictions[dataset_name] = sentiment_score

                # ======== Calculate the average and maximum prediction scores
                avg = np.mean(column_copy)
                maxi = np.max(column_copy)

                # ======== weighted averages
                for model in models:
                    weighted_averages[model] = get_weighted_average(predictions, model, ignoremodels, biasfactor)

                # 'OR' Classifier
                # aka max - see above

                # 'AND' Classifier
                # all must be above zero
                if np.all(column_copy != 0):
                    andc = round(np.mean(column_copy), 0)
                else:
                    andc = 0
                # Majority Vote
                counts = np.bincount(column_copy)
                majv = np.argmax(counts)

                # Extract the ground truth, previous prediction, and other relevant information
                truth = int(array_results[0, 3])
                previous = int(array_results[0, 5])
                accuracy = int(0)
                config = ''
                dataset_group = 'ensemble'

                # write out

                dataset = 'ens_avg'
                # Append the prediction for the average prediction score to values
                prediction = int(round(avg, 0))
                values.append((
                    timeslice, word[0], truth, prediction, previous, accuracy, dataset, config, version, dataset_group))

                dataset = 'ens_max'
                prediction = int(round(maxi, 0))
                values.append((
                    timeslice, word[0], truth, prediction, previous, accuracy, dataset, config, version, dataset_group))

                dataset = 'ens_and'
                prediction = int(round(andc, 0))
                values.append((
                    timeslice, word[0], truth, prediction, previous, accuracy, dataset, config, version, dataset_group))

                dataset = 'ens_maj'
                prediction = int(round(majv, 0))
                values.append((
                    timeslice, word[0], truth, prediction, previous, accuracy, dataset, config, version, dataset_group))

                for model in models:
                    prediction = weighted_averages[model]
                    ndataset = 'ens_wav_' + model.replace('Lancaster', 'ln_')
                    mconfig = str(biasfactor) + 'x'

                    values.append((timeslice, word[0], truth, prediction, previous, accuracy, ndataset, mconfig, version, dataset_group))

        # If there are values in 'values', insert them into the input_gnn_table_name table
        if len(values) > 0:
            print("Query Size is " + str(len(values)))
            # Check if values length exceeds chunk size
            if len(values) >= CHUNK_SIZE:
                write_to_db(values)
                values = []  # reset the values list

        # Write any remaining values to the DB after the word loop finishes
        if len(values) > 0:
            write_to_db(values)

        # Record the end time of the loop and print the elapsed time
        end_time = time.time()
        print(str(len(values)) + " Elapsed timeslice time: {:.2f} seconds".format(end_time - start_time))

    # Close the database connection and print completion message
    cursor.close()
    mydb.close()
    print("Done!")


def ensemble(credentials, tablename, outputtable, total_timeslices, version, ignoremodels, biasfactor):
    # calculate ensembles from the model results table

    mydb = mysql.connector.connect(
        host=credentials['host'],
        user=credentials['username'],
        password=credentials['password'],
        database=credentials['database']
    )
    cursor = mydb.cursor()
    print('querying database..')
    print("ensembling:")

    # Set model list
    models = ['Buchanan',
              'LancasterAuditory',
              'LancasterFootLeg',
              'LancasterGustatory',
              'LancasterHandArm',
              'LancasterHaptic',
              'LancasterHead',
              'LancasterInteroceptive',
              'LancasterMouth',
              'LancasterOlfactory',
              'LancasterTorso',
              'LancasterVisual',
              'Mcrae',
              'Nelson',
              'Rhyming',
              'GlasgowArousal',
              'GlasgowConcreteness',
              'GlasgowDominance',
              'GlasgowFamiliarity',
              'GlasgowGender',
              'GlasgowImageability',
              'GlasgowValence',
              'GlasgowSize',
              'GlasgowAOA']

    # Define a helper function to write chunks to the DB
    def write_to_db(values):
        query = "INSERT INTO " + outputtable + " (timeslice_id, word, truth, prediction, previous, accuracy, " \
                                               "dataset, config, version, dataset_group) " \
                                               "VALUES (%s, %s, %s, %s, " \
                                               "%s, %s, %s, %s, %s, %s)"
        cursor.executemany(query, values)
        mydb.commit()
        print('writing..')

    # Define a constant for the data chunk size to be written to the db
    CHUNK_SIZE = 1000

    # Loop through each timeslice
    for timeslice in range(total_timeslices):
        # Get a list of unique words in the table
        cursor.execute("SELECT word FROM " + tablename + " WHERE timeslice_id=" + str(timeslice) + " GROUP BY word")
        wordlist = cursor.fetchall()
        print("Timeslice: " + str(timeslice))
        values = []

        # Record the start time of the loop
        start_time = time.time()

        # Loop through each word in wordlist
        for word in wordlist:
            # print(word)
            # Select all rows in the table for the current timeslice and word
            query = "SELECT * FROM " + tablename + " WHERE timeslice_id=%s AND word=%s"
            vals = (timeslice, word[0])
            cursor.execute(query, vals)
            results = cursor.fetchall()
            num_rows = cursor.rowcount

            weighted_averages = {}
            # If there are rows returned, calculate predictions for the word
            if num_rows != 0:
                # Convert the results to a numpy array
                array_results = np.array(results)
                # Copy the column containing the sentiment scores
                column_copy = array_results[:, 4].copy()
                dataset_names = array_results[:, 8].copy()
                # Convert the sentiment scores to integers
                column_copy = column_copy.astype(int)

                dict_data = {}
                # clear predictions
                predictions = {'Buchanan': 0, 'LancasterAuditory': 0, 'LancasterFootLeg': 0,
                               'LancasterGustatory': 0, 'LancasterHandArm': 0, 'LancasterHaptic': 0,
                               'LancasterHead': 0, 'LancasterInteroceptive': 0, 'LancasterMouth': 0,
                               'LancasterOlfactory': 0, 'LancasterTorso': 0, 'LancasterVisual': 0,
                               'Mcrae': 0, 'Nelson': 0, 'Rhyming': 0, 'GlasgowArousal': 0,
                               'GlasgowConcreteness': 0, 'GlasgowDominance': 0, 'GlasgowFamiliarity': 0,
                               'GlasgowGender': 0, 'GlasgowImageability': 0, 'GlasgowValence': 0,
                               'GlasgowSize': 0, 'GlasgowAOA': 0}
                # get all model predictions for word in timeslice
                for i in range(len(array_results)):
                    row_data = array_results[i]
                    sentiment_score = int(row_data[4])
                    dataset_name = row_data[7]
                    # predictions[dataset_name] = sentiment_score
                    predictions[dataset_name] = sentiment_score

                # ======== Calculate the average and maximum prediction scores
                avg = np.mean(column_copy)
                maxi = np.max(column_copy)

                # ======== weighted averages
                for model in models:
                    weighted_averages[model] = get_weighted_average(predictions, model, ignoremodels, biasfactor)

                # 'OR' Classifier
                # aka max - see above

                # 'AND' Classifier
                # all must be above zero
                if np.all(column_copy != 0):
                    andc = round(np.mean(column_copy), 0)
                else:
                    andc = 0
                # Majority Vote
                counts = np.bincount(column_copy)
                majv = np.argmax(counts)

                # Extract the ground truth, previous prediction, and other relevant information
                truth = int(array_results[0, 3])
                previous = int(array_results[0, 5])
                accuracy = int(0)
                config = ''
                dataset_group = 'ensemble'

                # write out

                dataset = 'ens_avg'
                # Append the prediction for the average prediction score to values
                prediction = int(round(avg, 0))
                values.append((
                    timeslice, word[0], truth, prediction, previous, accuracy, dataset, config, version, dataset_group))

                dataset = 'ens_max'
                prediction = int(round(maxi, 0))
                values.append((
                    timeslice, word[0], truth, prediction, previous, accuracy, dataset, config, version, dataset_group))

                dataset = 'ens_and'
                prediction = int(round(andc, 0))
                values.append((
                    timeslice, word[0], truth, prediction, previous, accuracy, dataset, config, version, dataset_group))

                dataset = 'ens_maj'
                prediction = int(round(majv, 0))
                values.append((
                    timeslice, word[0], truth, prediction, previous, accuracy, dataset, config, version, dataset_group))

                for model in models:
                    prediction = weighted_averages[model]
                    ndataset = 'ens_wav_' + model.replace('Lancaster', 'ln_')
                    mconfig = str(biasfactor) + 'x'

                    values.append((timeslice, word[0], truth, prediction, previous, accuracy, ndataset, mconfig, version, dataset_group))

        # If there are values in values, insert them into the input_gnn_table_name table
        if len(values) > 0:
            print("Query Size is " + str(len(values)))
            # Check if values length exceeds chunk size
            if len(values) >= CHUNK_SIZE:
                write_to_db(values)
                values = []  # reset the values list

        # Write any remaining values to the DB after the word loop finishes
        if len(values) > 0:
            write_to_db(values)

        # Record the end time of the loop and print the elapsed time
        end_time = time.time()
        print(str(len(values)) + " Elapsed timeslice time: {:.2f} seconds".format(end_time - start_time))

    # Close the database connection and print completion message
    cursor.close()
    mydb.close()
    print("Done!")


def collate_models_results(credentials, tablename, collate_table_correct, collate_table_incorrect, timeslices):
    # we need to establish correct and incorrect predictions and collate them into two tables
    #  (new_collate_correct and new_collate_incorrect)

    # Set up connection to MySQL server
    mydb = mysql.connector.connect(
        host=credentials['host'],
        user=credentials['username'],
        password=credentials['password'],
        database=credentials['database']
    )
    cursor = mydb.cursor()
    print('querying database..')

    # Fetch the data and store it in a list of dictionaries

    dictlist = [dict() for x in range(timeslices)]

    x = 0

    # COLLATE CORRECT PREDICTIONS ===================================
    #  run query to get correct word counts

    # get total timeslices

    # for each timeslice
    for timeslice in range(timeslices):
        query = "SELECT dataset, COUNT(*) AS 'CORRECTLY PREDICTED WORDS' FROM " + tablename + " WHERE timeslice_id = " + str(timeslice) + \
                " AND prediction > previous AND truth > previous " \
                "GROUP BY dataset ORDER BY dataset"
        cursor.execute(query)
        x = x + 1

        # for each returned row, add to appropriate dictionary (i.e. for this timeslice)
        dictCorrectlyPredictedNum = {}
        for row in cursor:
            print(row)
            dictCorrectlyPredictedNum[row[0]] = row[1]

        # set missing keys to default 0 value
        if "Buchanan" not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['Buchanan'] = 0
        if "Mcrae" not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['Mcrae'] = 0
        if 'LancasterHaptic' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['LancasterHaptic'] = 0
        if 'LancasterAuditory' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['LancasterAuditory'] = 0
        if 'LancasterFootLeg' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['LancasterFootLeg'] = 0
        if 'LancasterGustatory' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['LancasterGustatory'] = 0
        if 'LancasterHandArm' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['LancasterHandArm'] = 0
        if 'LancasterHead' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['LancasterHead'] = 0
        if 'LancasterInteroceptive' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['LancasterInteroceptive'] = 0
        if 'LancasterMouth' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['LancasterMouth'] = 0
        if 'LancasterOlfactory' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['LancasterOlfactory'] = 0
        if 'LancasterTorso' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['LancasterTorso'] = 0
        if 'LancasterVisual' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['LancasterVisual'] = 0
        if 'GlasgowAOA' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['GlasgowAOA'] = 0
        if 'GlasgowSize' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['GlasgowSize'] = 0
        if 'GlasgowImageability' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['GlasgowImageability'] = 0
        if 'GlasgowValence' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['GlasgowValence'] = 0
        if 'GlasgowConcreteness' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['GlasgowConcreteness'] = 0
        if 'GlasgowGender' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['GlasgowGender'] = 0
        if 'GlasgowArousal' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['GlasgowArousal'] = 0
        if 'GlasgowDominance' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['GlasgowDominance'] = 0
        if 'GlasgowFamiliarity' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['GlasgowFamiliarity'] = 0
        if 'Norelationship' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['Norelationship'] = 0
        if 'Nelson' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['Nelson'] = 0
        if 'Rhyming' not in dictCorrectlyPredictedNum:
            dictCorrectlyPredictedNum['Rhyming'] = 0

        # now add this dictionary into the array
        try:
            dictlist[timeslice] = dictCorrectlyPredictedNum

        except:
            print("An exception occurred")

    # Now work out the actual number of new (improved comprehension) words. Getting words list from Buchanan
    for timeslice in range(timeslices):
        query = "SELECT 'NewWords', COUNT(*) FROM " + tablename + " WHERE timeslice_id = " + str(timeslice) + \
                " AND dataset = 'Buchanan' AND truth > previous "
        cursor.execute(query)

        # for each returned row, add to appropriate dictionary (i.e. for this timeslice)
        dictActualNum = {}
        for row in cursor:
            # print(row)
            dictActualNum[row[0]] = row[1]

        # set missing keys to default 0 value
        if "NewWords" not in dictActualNum:
            dictlist[timeslice]['NewWords'] = 0
        # now add this dictionary into the array
        else:
            dictlist[timeslice][row[0]] = row[1]

    # now we write the collated data back to the database
    for timeslice in range(timeslices):

        columns_string = "timeslice_id"
        for key in dictlist[timeslice].keys():
            if columns_string == "":
                columns_string = columns_string + key
            else:
                columns_string = columns_string + ", " + key

        values_string = str(timeslice)
        for value in dictlist[timeslice].values():
            if values_string == "":
                values_string = values_string + str(value)
            else:
                values_string = values_string + ", " + str(value)

        query = "INSERT INTO " + collate_table_correct + " (" + columns_string + ") VALUES (" + values_string + "); "

        cursor.execute(query)
        mydb.commit()

    # -----------------------------------------------------
    # Now the incorrect (false pos)  predictions

    inc_dictlist = [dict() for x in range(timeslices)]
    # for each timeslice

    for timeslice in range(timeslices):
        query = "SELECT dataset, COUNT(*) AS 'INCORRECTLY PREDICTED WORDS' FROM " + tablename + " WHERE timeslice_id = " + str(timeslice) + \
                " AND prediction > previous AND NOT truth > previous " \
                "GROUP BY dataset ORDER BY dataset"

        cursor.execute(query)
        x = x + 1

        # for each returned row, add to appropriate dictionary (i.e. for this timeslice)
        dictInCorrectlyPredictedNum = {}
        for row in cursor:
            print(row)
            dictInCorrectlyPredictedNum[row[0]] = row[1]

        # set missing keys to default 0 value
        if "Buchanan" not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['Buchanan'] = 0
        if "Mcrae" not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['Mcrae'] = 0
        if 'LancasterHaptic' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['LancasterHaptic'] = 0
        if 'LancasterAuditory' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['LancasterAuditory'] = 0
        if 'LancasterFootLeg' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['LancasterFootLeg'] = 0
        if 'LancasterGustatory' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['LancasterGustatory'] = 0
        if 'LancasterHandArm' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['LancasterHandArm'] = 0
        if 'LancasterHead' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['LancasterHead'] = 0
        if 'LancasterInteroceptive' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['LancasterInteroceptive'] = 0
        if 'LancasterMouth' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['LancasterMouth'] = 0
        if 'LancasterOlfactory' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['LancasterOlfactory'] = 0
        if 'LancasterTorso' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['LancasterTorso'] = 0
        if 'LancasterVisual' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['LancasterVisual'] = 0
        if 'GlasgowAOA' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['GlasgowAOA'] = 0
        if 'GlasgowSize' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['GlasgowSize'] = 0
        if 'GlasgowImageability' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['GlasgowImageability'] = 0
        if 'GlasgowValence' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['GlasgowValence'] = 0
        if 'GlasgowConcreteness' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['GlasgowConcreteness'] = 0
        if 'GlasgowGender' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['GlasgowGender'] = 0
        if 'GlasgowArousal' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['GlasgowArousal'] = 0
        if 'GlasgowDominance' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['GlasgowDominance'] = 0
        if 'GlasgowFamiliarity' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['GlasgowFamiliarity'] = 0
        if 'Norelationship' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['Norelationship'] = 0
        if 'Nelson' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['Nelson'] = 0
        if 'Rhyming' not in dictInCorrectlyPredictedNum:
            dictInCorrectlyPredictedNum['Rhyming'] = 0

        # now add this dictionary into the array
        try:
            inc_dictlist[timeslice] = dictInCorrectlyPredictedNum

        except:
            print("An exception occurred")

    # Now work out the actual number of new (improved comprehension) words - use Buchanan for words list
    for timeslice in range(timeslices):
        query = "SELECT 'NewWords', COUNT(*) FROM " + tablename + " WHERE timeslice_id = " + str(timeslice) + \
                " AND dataset = 'Buchanan' AND truth > previous "
        cursor.execute(query)

        # for each returned row, add to appropriate dictionary (i.e. for this timeslice)
        dictActualNum = {}
        for row in cursor:
            # print(row)
            dictActualNum[row[0]] = row[1]

        # set missing keys to default 0 value
        if "NewWords" not in dictActualNum:
            inc_dictlist[timeslice]['NewWords'] = 0
        # now add this dictionary into the array
        else:
            inc_dictlist[timeslice][row[0]] = row[1]

    # now we write the collated data back to the database
    for timeslice in range(timeslices):

        columns_string = "timeslice_id"
        for key in inc_dictlist[timeslice].keys():
            if columns_string == "":
                columns_string = columns_string + key
            else:
                columns_string = columns_string + ", " + key

        values_string = str(timeslice)
        for value in inc_dictlist[timeslice].values():
            if values_string == "":
                values_string = values_string + str(value)
            else:
                values_string = values_string + ", " + str(value)

        query = "INSERT INTO " + collate_table_incorrect + " (" + columns_string + ") VALUES (" + values_string + "); "

        cursor.execute(query)
        mydb.commit()

    # Close the cursor and the connection
    cursor.close()
    mydb.close()
    # print(arrayAnalysedoutput)
    print("Done!")
    return ("collating done.")


def calculate_models_performance(credentials, report_filename, models_output_table, datversion, results_directory):
    # calculate performance from collated tables
    # Set up connection to MySQL server
    mydb = mysql.connector.connect(
        host=credentials['host'],
        user=credentials['username'],
        password=credentials['password'],
        database=credentials['database']
    )
    cursor = mydb.cursor()
    print('querying database..')
    # this outputs the performance metrics to std out

    # set this if we wish to ignore first-of-sequence, keep empty if not
    ignore_first_surveys = ""

    # true pos = predicts new knowledge, there is new knowledge
    cursor.execute(
        # "SELECT dataset, COUNT(*) AS 'truepos' FROM " + input_gnn_table_name + " WHERE version = '" + datversion + "' AND prediction > previous AND truth > previous " +
        "SELECT dataset, COUNT(*) AS 'truepos' FROM " + models_output_table + " WHERE prediction > previous AND truth > previous " +
        ignore_first_surveys + " GROUP BY dataset ORDER BY dataset")
    response = cursor.fetchall()
    tp = response

    # fp = predicts new knowledge, no new knowledge
    cursor.execute(
        # "SELECT dataset, COUNT(*) AS 'falsepos' FROM " + input_gnn_table_name + " WHERE version = '" + datversion + "' AND prediction > previous AND NOT (truth > previous) " +
        "SELECT dataset, COUNT(*) AS 'falsepos' FROM " + models_output_table + " WHERE prediction > previous AND NOT (truth > previous) " +
        ignore_first_surveys + " GROUP BY dataset ORDER BY dataset")
    response = cursor.fetchall()
    fp = response

    # tn = predicts no new knowledge, there is no new knowledge
    cursor.execute(
        # "SELECT dataset, COUNT(*) AS 'trueneg' FROM " + input_gnn_table_name + " WHERE version = '" + datversion + "' AND NOT (prediction > previous) AND NOT truth > previous " +
        "SELECT dataset, COUNT(*) AS 'trueneg' FROM " + models_output_table + " WHERE NOT (prediction > previous) AND NOT truth > previous " +
        ignore_first_surveys + " GROUP BY dataset ORDER BY dataset")
    response = cursor.fetchall()
    tn = response

    # fn = predicts no new knowledge but there is new knowledge
    cursor.execute(
        # "SELECT dataset, COUNT(*) AS 'falseneg' FROM " + input_gnn_table_name + " WHERE version = '" + datversion + "' AND NOT (prediction > previous) AND (truth > previous) " +
        "SELECT dataset, COUNT(*) AS 'falseneg' FROM " + models_output_table + " WHERE NOT (prediction > previous) AND (truth > previous) " +
        ignore_first_surveys + " GROUP BY dataset ORDER BY dataset")
    response = cursor.fetchall()
    fn = response

    # Precision = (Number of true positives) / (Number of true positives + Number of false positives)
    print("Precision:")
    for x in range(len(tp)):
        precision = (tp[x][1] / (tp[x][1] + fp[x][1]))
        modelName = str(tp[x][0])
        print(modelName + ", " + str(round(precision, 2)))

    # Recall = TruePositives / (TruePositives + FalseNegatives)
    print("Recall:")
    for x in range(len(tp)):
        recall = (tp[x][1] / (tp[x][1] + fn[x][1]))
        modelName = str(tp[x][0])
        print(modelName + ", " + str(round(recall, 2)))

    # F-Measure = (2 * Precision * Recall) / (Precision + Recall)
    print("F1:")
    for x in range(len(tp)):
        prec = (tp[x][1] / (tp[x][1] + fp[x][1]))
        recall = (tp[x][1] / (tp[x][1] + fn[x][1]))
        f1 = (2 * (prec * recall)) / (prec + recall)
        # f1 = 2 * (((tp[x][1] / (tp[x][1] + fp[x][1])) * (tp[x][1] / (tp[x][1] + fn[x][1]))) / (
        #            ((tp[x][1] / (tp[x][1] + fp[x][1]))) + ((tp[x][1] / (tp[x][1] + fn[x][1])))))
        modelName = str(tp[x][0])
        print(modelName + ", " + str(round(f1, 2)))

    # Accuracy = (TP+TN) / (Total number of samples)
    print("Accuracy:")
    for x in range(len(tp)):
        accuracy = ((tp[x][1] + tn[x][1]) / (tp[x][1] + tn[x][1] + fp[x][1] + fn[x][1]))
        modelName = str(tp[x][0])
        print(modelName + ", " + str(round(accuracy, 3)))

    models_data = []
    # Print table rows
    for x in range(len(tp)):
        modelName = str(tp[x][0])
        precision = tp[x][1] / (tp[x][1] + fp[x][1])
        recall = tp[x][1] / (tp[x][1] + fn[x][1])
        f1 = 2 * ((precision * recall) / (precision + recall))
        accuracy = (tp[x][1] + tn[x][1]) / (tp[x][1] + tn[x][1] + fp[x][1] + fn[x][1])
        models_data.append((modelName, precision, recall, f1, accuracy))

    # Sort the list by accuracy in descending order
    sorted_models = sorted(models_data, key=lambda x: x[4], reverse=True)

    # Print LaTeX table header
    print("\\begin{table}[htbp]")
    print("  \\centering")
    print("  \\caption{Individual Model Performance}")
    print("  \\label{tab:model_performance}")
    print("  \\begin{tabular}{lllll}")
    print("    \\toprule")
    print("    Model & Precision & Recall & f1 & Accuracy\\\\")
    print("    \\midrule")

    # Print table rows from the sorted list
    for model in sorted_models:
        modelName = model[0].replace('Glasgow', '').replace('Lancaster', '').replace('Norelationship', 'No Relationship').replace('Mcrae', 'Semantic (Mcrae)').replace('Nelson', 'Word Association').replace('Buchanan', 'Semantic (Buch.)')
        precision, recall, f1, accuracy = model[1], model[2], model[3], model[4]
        print("    {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(modelName, precision, recall, f1, accuracy))

    # Print LaTeX table footer
    print("    \\bottomrule")
    print("  \\end{tabular}")
    print("\\end{table}")

    # Close the cursor and the connection
    cursor.close()
    mydb.close()

    print("Done!")

    return ("model performance calculated.")


def calculate_ensemble_performance(credentials, report_filename, ensemble_table, datversion, configuration, results_directory):
    # calculate performance from collate tables
    # Set up connection to MySQL server
    mydb = mysql.connector.connect(
        host=credentials['host'],
        user=credentials['username'],
        password=credentials['password'],
        database=credentials['database']
    )
    cursor = mydb.cursor()
    print('querying database..')
    # this outputs the performance metrics to std out

    ignore_first_surveys = ""

    # true pos = predicts new knowledge, there is new knowledge
    cursor.execute(
        "SELECT dataset, COUNT(*) AS 'truepos' FROM " + ensemble_table + " WHERE prediction > previous AND truth > previous " +
        ignore_first_surveys + " GROUP BY dataset ORDER BY dataset")
    response = cursor.fetchall()
    tp = response

    # fp = predicts new knowledge, no new knowledge
    cursor.execute(
        "SELECT dataset, COUNT(*) AS 'falsepos' FROM " + ensemble_table + " WHERE prediction > previous AND NOT (truth > previous) " +
        ignore_first_surveys + " GROUP BY dataset ORDER BY dataset")
    response = cursor.fetchall()
    fp = response

    # tn = predicts no new knowledge, there is no new knowledge
    cursor.execute(
        "SELECT dataset, COUNT(*) AS 'trueneg' FROM " + ensemble_table + " WHERE NOT (prediction > previous) AND NOT truth > previous " +
        ignore_first_surveys + " GROUP BY dataset ORDER BY dataset")
    response = cursor.fetchall()
    tn = response

    # fn = predicts no new knowledge but there is new knowledge
    cursor.execute(
        "SELECT dataset, COUNT(*) AS 'falseneg' FROM " + ensemble_table + " WHERE NOT (prediction > previous) AND (truth > previous) " +
        ignore_first_surveys + " GROUP BY dataset ORDER BY dataset")
    response = cursor.fetchall()
    fn = response

    # Precision = (Number of true positives) / (Number of true positives + Number of false positives)
    print("Precision:")
    for x in range(len(tp)):
        precision = (tp[x][1] / (tp[x][1] + fp[x][1]))
        modelName = str(tp[x][0])
        print(modelName + ", " + str(round(precision, 3)))

    # Recall = TruePositives / (TruePositives + FalseNegatives)
    print("Recall:")
    for x in range(len(tp)):
        recall = (tp[x][1] / (tp[x][1] + fn[x][1]))
        modelName = str(tp[x][0])
        print(modelName + ", " + str(round(recall, 3)))

    print("F1:")
    for x in range(len(tp)):
        prec = (tp[x][1] / (tp[x][1] + fp[x][1]))
        recall = (tp[x][1] / (tp[x][1] + fn[x][1]))
        f1 = (2 * (prec * recall)) / (prec + recall)
        # f1 = 2 * (((tp[x][1] / (tp[x][1] + fp[x][1])) * (tp[x][1] / (tp[x][1] + fn[x][1]))) / (
        #            ((tp[x][1] / (tp[x][1] + fp[x][1]))) + ((tp[x][1] / (tp[x][1] + fn[x][1])))))
        modelName = str(tp[x][0])
        print(modelName + ", " + str(round(f1, 3)))

    # Accuracy = (TP+TN) / (Total number of samples)
    print("Accuracy:")
    for x in range(len(tp)):
        accuracy = ((tp[x][1] + tn[x][1]) / (tp[x][1] + tn[x][1] + fp[x][1] + fn[x][1]))
        modelName = str(tp[x][0])
        print(modelName + ", " + str(round(accuracy, 3)))

    models_data = []
    # Print table rows
    for x in range(len(tp)):
        modelName = str(tp[x][0])
        precision = tp[x][1] / (tp[x][1] + fp[x][1])
        recall = tp[x][1] / (tp[x][1] + fn[x][1])
        f1 = 2 * ((precision * recall) / (precision + recall))
        accuracy = (tp[x][1] + tn[x][1]) / (tp[x][1] + tn[x][1] + fp[x][1] + fn[x][1])
        models_data.append((modelName, precision, recall, f1, accuracy))

    # Sort the list by accuracy in descending order
    sorted_models = sorted(models_data, key=lambda x: x[4], reverse=True)

    # Print LaTeX table header
    print("\\begin{table}[htbp]")
    print("  \\centering")
    print("  \\caption{Ensemble Performance}")
    print("  \\label{tab:ensemble_performance}")
    print("  \\begin{tabular}{lllll}")
    print("    \\toprule")
    print("    Model & Precision & Recall & f1 & Accuracy\\\\")
    print("    \\midrule")

    # Print table rows from the sorted list
    for model in sorted_models:
        modelName = model[0].replace('Glasgow', '').replace('_ln', '').replace('Norelationship', 'No Relationship').replace('Mcrae', 'Semantic (Mcrae)').replace('Nelson', 'Word Association').replace('Buchanan', 'Semantic (Buch.)')
        precision, recall, f1, accuracy = model[1], model[2], model[3], model[4]
        print("    {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(modelName, precision, recall, f1, accuracy))

    # Print LaTeX table footer
    print("    \\bottomrule")
    print("  \\end{tabular}")
    print("\\end{table}")

    # Close the cursor and the connection
    cursor.close()
    mydb.close()

    print("Done!")

    return "ensemble performance calculated."
