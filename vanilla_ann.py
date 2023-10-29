#
# Using a vanilla artificial neural net on observational data
# (based on the spec of the Beckage CDI Word model)
# This is written in Python using Keras/Tensorflow
# (whereas the original was written in Lua.)
# v.1.3
# A.Roxburgh 2020

import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


# at a fixed position to split at the end of a n-observation series
def train_test_split_fixed(data, fixed_size):
    train_size = fixed_size
    train_data = np.array(data.iloc[:, :train_size])
    test_data = np.array(data.iloc[:, train_size:])
    return train_data, test_data


# Scaling
# Rescale the data from the original range so that all values are within the range of 0 and 1.
def scale_data(train_data, test_data):
    # Ensure both datasets are numpy arrays
    if not isinstance(train_data, np.ndarray):
        train_data = train_data.values
    if not isinstance(test_data, np.ndarray):
        test_data = test_data.values

    # Get the global max and min values
    global_max = max(train_data.max(), test_data.max())
    global_min = min(train_data.min(), test_data.min())

    # Scale the data
    train_scaled = (train_data - global_min) / (global_max - global_min)
    test_scaled = (test_data - global_min) / (global_max - global_min)

    return train_scaled, test_scaled


####################################################################################
# Models specs :
#  trained via stochastic gradient descent
#  single hidden layer
#  optimized in size for each trained model
#  variable number of input features based on the vocabulary representation
#  logistic transformation (i.e. sigmoid) on the output layer
#    such that the probability of learning a speciﬁc word was returned by the model.
#   Model      inputs   α       hu   batch   α decay (in epochs)    m   avg?
# -------------------------------------------------------------------------------------
#   CDI Word    677+6   0.8     500   25      200                   .9          words
####################################################################################

# experimental parameters
col_dtypes = {'word': 'string',
              'UID': 'string',
              'ProbOfSaying': 'int',
              'ProbOfUnderstanding': 'int'}

nodes_dataframe = pd.DataFrame(columns=[*col_dtypes])
nodes_directory = os.getcwd() + "\\data\\nodes\\Set3\\"
testtrainsplitpoint = 339  # 365

# ---------  hyperparameters  -----------
learning_rate = 0.8  # 0.8
batch_size = 1  # 25
momentum = 0.9
alpha_decay = 200
epochs = 1000

# --------  architecture  -------------
total_hidden_units = 500
total_input_features = 418
total_network_outputs = 418
num_hidden_layers = 1  # Set this to 1 or 2

# Define a custom sort function that will retrieve the serial number as an integer
def custom_sort(filename):
    return int(filename.split("_")[0])


# Sort files using the custom sort function
sorted_files = sorted(os.listdir(nodes_directory), key=custom_sort)

# load nodes files
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
        # print(length)
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

        # print('timeslice: ' + str(timeslice_id))

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
vocab_data_transposed = vocab_data.transpose()

train_data_t, test_data_t = train_test_split_fixed(vocab_data, testtrainsplitpoint)
train_data = train_data_t.transpose()
test_data = test_data_t.transpose()
X, Y = [], []

for i in range(len(train_data) - 1):
    X.append(train_data[i])
    Y.append(train_data[i + 1])

train_data = pd.DataFrame(X)
target_data = pd.DataFrame(Y)

# print("Train data: ", train_data.shape)
# print("Test data: ", test_data.shape)

train_scaled, test_scaled = scale_data(train_data, test_data)

train_set = train_scaled
target_set = target_data
test_data = test_scaled

# Build the model.

# Input layer
inputs = tf.keras.Input(
    shape=(total_input_features,),
    name="vocabulary",
    dtype=tf.float32)

# Hidden Layer1
x = tf.keras.layers.Dense(
    total_hidden_units,
    activation=tf.nn.relu,
    name="hidden",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="random_uniform",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None)(inputs)

# If num_hidden_layers is 2 or 3, add the second hidden layer
if num_hidden_layers >= 2:
    x = tf.keras.layers.Dense(
        total_hidden_units,
        activation=tf.nn.relu,
        name="hidden2",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="random_uniform",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None)(x)

# If num_hidden_layers is 3, add the third hidden layer
if num_hidden_layers == 3:
    x = tf.keras.layers.Dense(
        total_hidden_units,
        activation=tf.nn.relu,
        name="hidden3",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="random_uniform",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None)(x)

# Output Layer
outputs = tf.keras.layers.Dense(
    total_network_outputs,
    activation=tf.nn.sigmoid,
    name="prediction",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None)(x)

# Assemble
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model.
model.compile(
    optimizer='adam',  # adam or sgd
    loss='mae',  # mse
    metrics=['mse', 'accuracy'],
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
)

# load weights?
# model.load_weights('modelTEST6.h5')

log_dir = ".\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model.
history = model.fit(
    x=train_set,
    y=target_set,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    callbacks=[tensorboard_callback],
    validation_split=0.2,
    validation_data=None,
    shuffle=False,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_freq=1,
    max_queue_size=10,
    workers=8,
    use_multiprocessing=True,
)

# Evaluate the model.
model.evaluate(
    x=train_set,
    y=target_set,
    batch_size=batch_size,
    verbose=1,
    sample_weight=None,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=8,
    use_multiprocessing=True,
)

# Save the model
model.save_weights('newmod.h5')

predictions = model.predict(test_data, verbose=1)

testSize = test_data.size
print(test_data.take)
for i in range(len(predictions)):
    print("X=%s, Predicted=%s" % (i, predictions[i]))

# Setting up the figure
plt.figure(figsize=(12, 6))

# Plotting the training and validation loss on the first subplot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()

# Plotting the training and validation accuracy on the second subplot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Trn vs Val Acc:' + ' Hdn Layrs:' + str(num_hidden_layers) + '  hu:' +
          str(total_hidden_units) + ' lr:' + str(learning_rate) + ' bs:' +
          str(batch_size) + ' m:' + str(momentum) + ' ad:' + str(alpha_decay) +
          ' ep:' + str(epochs))
plt.legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
