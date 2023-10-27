# STGCN Vocab Prediction Expts
# Processes input graphs
# Latest update Jul 2023
# v.2.1.4

# Notes ------------
# This assumes input graphs have been generated (see graph generator code)
# Imports settings.ini for controlling the experiments
# ------------------

# imports
import stellargraph as sg
import tensorflow as tf
from tensorflow.python.client import device_lib

import vocpredlib as ev
import helper

# initialise
print("TF version: ")
print(tf.version.VERSION)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()
print(sg.__version__)

# get all config parameters
config = helper.read_config()
credentials = dict({'host': config['database']['host'],
                    'username': config['database']['username'],
                    'password': config['database']['password'],
                    'database': config['database']['database']})
# relationship files

# adjustment files
cdi_replace_file = config['folders']['cleansing_dictionaries'] + config['files']['cdi_replace']
english_american_file = config['folders']['cleansing_dictionaries'] + config['files']['english_american']

# misc files
cdi_question_lookup_file = config['folders']['cleansing_dictionaries'] + config['files']['cdi_question_lookup']
new_words_file = config['folders']['cleansing_dictionaries'] + config['files']['new_words']

# outputs
edges_files_folder = config['folders']['edges_files_folder']
node_files_folder = config['folders']['nodes_files_folder']
stgcn_output_folder = config['folders']['stgcn_output_folder']
# to do - this is a temp thing:
root_edges_files_folder = config['folders']['edges_files_folder'] + '2000\\'
# switches
output_mcrae = config['testing_configuration']['mcrae']
output_nelson = config['testing_configuration']['nelson']
output_lancaster = config['testing_configuration']['lancaster']
output_buchanan = config['testing_configuration']['buchanan']
output_rhyming = config['testing_configuration']['rhyming']
output_glasgow = config['testing_configuration']['glasgow']
output_norelationship = config['testing_configuration']['norelationship']
verbose_mode = config['testing_configuration']['verbose_mode']
survey_mode = config['misc']['survey_mode']

use_wordbank = eval(config['make_graphs']['use_wordbank'])

generate_node_files_from_wordbank = config['wordbank_process']['generate_node_files']
append_new_surveys = config['wordbank_process']['append_new_surveys']

run_wordbank_tests = config['testing_configuration']['runwordbanktests']
loadmodel = config['testing_configuration']['loadmodel']
savemodel = config['testing_configuration']['savemodel']
testing = eval(config['testing_configuration']['testing'])
buchanan = eval(config['testing_configuration']['buchanan'])
mcrae = eval(config['testing_configuration']['mcrae'])
lancaster = eval(config['testing_configuration']['lancaster'])
glasgow = eval(config['testing_configuration']['glasgow'])
rhyming = eval(config['testing_configuration']['rhyming'])
nelson = eval(config['testing_configuration']['nelson'])
vanilla = eval(config['testing_configuration']['vanilla'])
norelationship = eval(config['testing_configuration']['norelationship'])

logfolder = config['logging']['logfilepath']
epochs = config['testing_configuration']['epochs'].split()
res = [eval(i) for i in epochs]
epochs = res

sequences = config['testing_configuration']['sequences'].split()
res = [eval(i) for i in sequences]
sequences = res

pred_lengths = config['testing_configuration']['pred_lengths'].split()
res = [eval(i) for i in pred_lengths]
pred_lengths = res

batch_sizes = config['testing_configuration']['batch_sizes'].split()
res = [eval(i) for i in batch_sizes]
batch_sizes = res

testtrainsplitpoint = int(config['testing_configuration']['testtrainsplitpoint'])

m_optimizer = config['stgcn_hyperparameters']['m_optimizer']
m_loss = config['stgcn_hyperparameters']['m_loss']
m_metrics = config['stgcn_hyperparameters']['m_metrics'].split()
output_logfile = config['stgcn_hyperparameters']['output_logfile']

import_stgcn_to_mysql = config['evaluation_configuration']['import_stgcn_to_mysql']

calculate_ensembles = config['ensemble_configuration']['calculate_ensembles']
collate_results = config['evaluation_configuration']['collate_results']
generate_reports = config['report_configuration']['generate_reports']

# load search-and-replace file
cdi_replace_frame = ev.load_cdi_replace(cdi_replace_file)
english_american_frame = ev.load_english_american(english_american_file)

if use_wordbank:
    # Set up connection to MySQL server
    purecdi_dataframe = ev.get_wordbank_wordlist_from_mysql(credentials, 'wordbank_target')

# ============== STGCN ======================================
print("STGCN Tests phase ")
for epoch in epochs:
    for sequence in sequences:
        for pred_len in pred_lengths:
            for batchsize in batch_sizes:
                print('epoch:' + str(epoch))
                if nelson:
                    ev.evaluate_model("Nelson", node_files_folder, stgcn_output_folder, output_logfile, 'nelson_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel, logfolder)
                if rhyming:
                    ev.evaluate_model("Rhyming", node_files_folder, stgcn_output_folder, output_logfile, 'rhyming_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel, logfolder)
                if lancaster:
                    ev.evaluate_model("Lancaster Gustatory", node_files_folder, stgcn_output_folder, output_logfile, 'lancaster_gustatory_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel,
                                      savemodel, logfolder)
                    ev.evaluate_model("Lancaster Foot Leg", node_files_folder, stgcn_output_folder, output_logfile, 'lancaster_foot_leg_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel,
                                      logfolder)
                    ev.evaluate_model("Lancaster Hand Arm", node_files_folder, stgcn_output_folder, output_logfile, 'lancaster_hand_arm_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel,
                                      logfolder)
                    ev.evaluate_model("Lancaster Haptic", node_files_folder, stgcn_output_folder, output_logfile, 'lancaster_haptic_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel,
                                      logfolder)
                    ev.evaluate_model("Lancaster Head", node_files_folder, stgcn_output_folder, output_logfile, 'lancaster_head_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel,
                                      logfolder)
                    ev.evaluate_model("Lancaster Interoceptive", node_files_folder, stgcn_output_folder, output_logfile, 'lancaster_interoceptive_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel,
                                      savemodel, logfolder)
                    ev.evaluate_model("Lancaster Mouth", node_files_folder, stgcn_output_folder, output_logfile, 'lancaster_mouth_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel,
                                      logfolder)
                    ev.evaluate_model("Lancaster Olfactory", node_files_folder, stgcn_output_folder, output_logfile, 'lancaster_olfactory_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel,
                                      savemodel, logfolder)
                    ev.evaluate_model("Lancaster Torso", node_files_folder, stgcn_output_folder, output_logfile, 'lancaster_torso_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel,
                                      logfolder)
                    ev.evaluate_model("Lancaster Visual", node_files_folder, stgcn_output_folder, output_logfile, 'lancaster_visual_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel,
                                      logfolder)
                    ev.evaluate_model("Lancaster Auditory", node_files_folder, stgcn_output_folder, output_logfile, 'lancaster_auditory_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel,
                                      logfolder)
                if glasgow:
                    ev.evaluate_model("Glasgow AOA", node_files_folder, stgcn_output_folder, output_logfile, 'glasgow_aoa_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel, logfolder)
                    ev.evaluate_model("Glasgow Arousal", node_files_folder, stgcn_output_folder, output_logfile, 'glasgow_arousal_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel,
                                      logfolder)
                    ev.evaluate_model("Glasgow Concreteness", node_files_folder, stgcn_output_folder, output_logfile, 'glasgow_concreteness_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel,
                                      savemodel, logfolder)
                    ev.evaluate_model("Glasgow Dominance", node_files_folder, stgcn_output_folder, output_logfile, 'glasgow_dominance_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel,
                                      logfolder)
                    ev.evaluate_model("Glasgow Familiarity", node_files_folder, stgcn_output_folder, output_logfile, 'glasgow_familiarity_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel,
                                      savemodel, logfolder)
                    ev.evaluate_model("Glasgow Gender", node_files_folder, stgcn_output_folder, output_logfile, 'glasgow_gend_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel, logfolder)
                    ev.evaluate_model("Glasgow Imageability", node_files_folder, stgcn_output_folder, output_logfile, 'glasgow_imageability_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel,
                                      savemodel, logfolder)
                    ev.evaluate_model("Glasgow Size", node_files_folder, stgcn_output_folder, output_logfile, 'glasgow_size_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel, logfolder)
                    ev.evaluate_model("Glasgow Valence", node_files_folder, stgcn_output_folder, output_logfile, 'glasgow_valence_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel,
                                      logfolder)
                if buchanan:
                    ev.evaluate_model("Buchanan", node_files_folder, stgcn_output_folder, output_logfile, 'buchanan_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel, logfolder)
                if mcrae:
                    ev.evaluate_model("Mcrae", node_files_folder, stgcn_output_folder, output_logfile, 'mcrae_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel, logfolder)
                if norelationship:
                    ev.evaluate_model("Norelationship", node_files_folder, stgcn_output_folder, output_logfile, 'norelationship_edges.csv', sequence, pred_len, batchsize, epoch, 'optimistic', root_edges_files_folder, testtrainsplitpoint, m_optimizer, m_loss, m_metrics, loadmodel, savemodel,
                                      logfolder)
# ============== RESULTS EVALUATION ==============================
# put results into mysql database
if run_wordbank_tests == 'True':

    # load from wordbank
    wordnet_dataframe = ev.load_from_wordnet(credentials)
    # from mysql - wordbank

    for columns in wordnet_dataframe.columns:
        wordnet_dataframe['item_definition'] = wordnet_dataframe['item_definition'].str.upper()

    wordnet_dataframe['item_definition'] = wordnet_dataframe['item_definition'].str.replace(' / ', '/')

    wordnet_dataframe = ev.replace_from_dictionary(wordnet_dataframe, cdi_replace_frame, 'item_definition')  # standardise CDI

    column_to_modify = 'item_definition'
    category_column = 'category'
    category_value = 'animals'

    # Disambiguation. List of words to change: (category_value, original_word, replacement_word)
    words_to_change = [('animals', 'FISH', 'FISH (ANIMAL)'),
                       ('animals', 'CHICKEN', 'CHICKEN (ANIMAL)'),
                       ('clothing', 'BATH', 'BATH (FURNITURE)'),
                       ('action_words', 'CLEAN', 'CLEAN (ACTION)'),
                       ('action_words', 'DRINK', 'DRINK (ACTION)'),
                       ('action_words', 'SWING', 'SWING (ACTION)'),
                       ('action_words', 'WATCH', 'WATCH (ACTION)'),
                       ('outside', 'WATER', 'WATER (OUTSIDE)'),
                       ('games_routines', 'TEA', 'TEA (DINNER)'),
                       ('prepositions', 'BACK', 'BACK (PLACE)')]

    # Iterate over the list of words and update the DataFrame
    for word in words_to_change:
        cat_value, search_value, replacement_value = word
        wordnet_dataframe.loc[(wordnet_dataframe[category_column] == cat_value) & (wordnet_dataframe[column_to_modify] == search_value), column_to_modify] = replacement_value

if import_stgcn_to_mysql == 'True':
    print("importing results from csv to mysql...")
    # specify the model to be imported into mysql (ALL for all in folder)
    modelname = "Norelationship"
    modelname = "ALL"  # none=import nothing
    for epoch in epochs:
        print(ev.import_stgcn_results_into_mysql(credentials, stgcn_output_folder + str(epoch) + '\\', modelname, 'out_stgcn'))

# ========== ENSEMBLES ==========
# calculate ensemble results based on the STGCN output table
# then put into mysql table new_ensemble_results
# timeslices of test set
total_timeslices = config['ensemble_configuration']['testslices']

if calculate_ensembles == 'True':
    print("calculating ensembles...")

    versiontag = '0508230000'
    ignoremodels = ['Norelationship', 'Buchanan', 'LancasterFootLeg', 'LancasterGustatory',
                    'LancasterHandArm', 'LancasterHaptic', 'LancasterHead', 'LancasterInteroceptive',
                    'LancasterMouth', 'LancasterOlfactory', 'LancasterVisual', 'Rhyming', 'GlasgowArousal',
                    'GlasgowConcreteness', 'GlasgowFamiliarity', 'GlasgowGender', 'GlasgowValence', 'GlasgowSize']
    # weighted average bias factor : x5 etc
    biasfactor = config['ensemble_configuration']['biasfactor']
    # print(ev.ensemble(credentials, 'out_stgcn', 'new_out_ensemble', total_timeslices, versiontag, ignoremodels, biasfactor))
    print(ev.reduced_ensemble(credentials, 'out_stgcn', 'new_out_ensemble', total_timeslices, versiontag, ignoremodels, biasfactor))

# ========== COLLATE RESULTS =====
# collate the results from all models into collate tables
# so we can ensemble them easily
# Note : these are usually new_collate_correct and new_collate_incorrect
datversion = config['testing_configuration']['datversion']
configuration = config['testing_configuration']['configuration']
results_directory = config['folders']['results_directory']

if collate_results == 'True':
    print("collating model results...")
    print(ev.collate_models_results(credentials, 'out_stgcn', 'new_collated_correct', 'new_collated_incorrect', total_timeslices))
    print(ev.calculate_ensemble_performance(credentials, config['report_configuration']['filename'], 'new_out_ensemble',  datversion, configuration, results_directory))

# ========== PERFORMANCE ==========
if generate_reports == 'True':
    print("calculating performance...")
    # first calculate the individual models performance
    # we take the data from the collate tables for this
    # calculate precision, f1, accuracy, recall for each model and ensemble
    print(ev.calculate_ensemble_performance(credentials, config['report_configuration']['filename'], 'new_out_ensemble', datversion, configuration, results_directory))
    print(ev.calculate_models_performance(credentials, config['report_configuration']['filename'], 'out_stgcn', datversion, results_directory))

print("FINISHED")
