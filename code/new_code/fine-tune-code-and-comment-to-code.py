# Import statements
import functools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import time
import warnings

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from t5.data import postprocessors as t5_postprocessors
from t5.seqio import Feature,SentencePieceVocabulary
import t5
import gin
import math
from mesh_tensorflow.transformer.learning_rate_schedules import slanted_triangular 

from mesh_tensorflow.transformer.learning_rate_schedules import truncated_rsqrt
 
from tensorflow.keras.optimizers.schedules import PolynomialDecay
import datetime
import argparse

from transformers import T5Config, T5ForConditionalGeneration, load_tf_weights_in_t5
from transformers.utils import logging

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd

tf.autograph.set_verbosity(1)
warnings.filterwarnings("ignore", category=DeprecationWarning)


current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
print("current_time:", current_time)


# Directory Paths
root_path = "/workspace/itsec/itsec1/test/ssh_temp/"
task_name1 = "code&comment-to-code"
task_name2 = "code&comment2code: "
task_name3 = "code_comment_to_code_new_large"
task_name4 = "code_and_comment2code"

train_path = root_path + "automating_code_review/dataset/fine-tuning/new_large/" + task_name1 + "/train.tsv"
val_path = root_path + "automating_code_review/dataset/fine-tuning/new_large/" + task_name1 + "/val.tsv"


#Model vocab and path
vocab_model_path = root_path + "automating_code_review/tokenizer/TokenizerModel.model"
vocab_path = root_path + "automating_code_review/tokenizer/TokenizerModel.vocab"

# Model cehckpoint path
MODEL_DIR = root_path + "model_checkpoints/" + task_name4 + "/check_" + current_time
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Gin config path
GIN_PATH = root_path + 'automating_code_review/utils/operative_config_constant.gin'


# Read the data
nq_tsv_path_code_code_large = {
    "train":     train_path,
    "validation": val_path
}

data_train = len([line for line in open(train_path, 'r')])
data_val = len([line for line in open(val_path, 'r')])

num_nq_examples_code_code_large = dict(train=data_train, validation=data_val)


# Helper functions
def get_default_vocabulary():
  return SentencePieceVocabulary(vocab_model_path, 100)

#Setting up code to code task on new large dataset
def nq_dataset_code_code_large(split, shuffle_files=True):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(nq_tsv_path_code_code_large[split])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["string","string"],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  ds = ds.map(lambda *ex: dict(zip(["input", "output"], ex)))
  return ds


def code_code_preprocessing(ds):
  def to_inputs_and_targets(ex):
        inputs = tf.strings.join([task_name2 + ex['input']], separator=' ')
        class_label = tf.strings.join([ex['output']], separator=' ')
        return {'inputs': inputs, 'targets': class_label }
    
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

#Setting up fine tuning tasks
def _rate_num_input_examples(task):
  if "train" in task.splits:
    return float(task.num_input_examples("train"))
  elif "validation" in task.splits:
    return float(task.num_input_examples("validation"))
  else:
    raise ValueError("Task %s does not have a train or validation split." % (task.name))


# Features of Inputs & Outputs
DEFAULT_OUTPUT_FEATURES = {
    "inputs": Feature(
        vocabulary=get_default_vocabulary(), add_eos=True, required=False),

    "targets": Feature(
        vocabulary=get_default_vocabulary(), add_eos=True)
}


# T5 Task Registry
t5.data.TaskRegistry.remove(task_name3)
t5.data.TaskRegistry.add(
    task_name3,
    dataset_fn=nq_dataset_code_code_large,
    splits=["train", "validation"],
    text_preprocessor=[code_code_preprocessing],
    output_features = DEFAULT_OUTPUT_FEATURES,
    metric_fns=[t5.evaluation.metrics.accuracy],
    num_input_examples=num_nq_examples_code_code_large
)

nq_task = t5.data.TaskRegistry.get(task_name3)
ds = nq_task.get_dataset(split="train", sequence_length={"inputs": 512, "targets": 512})


# Init T5
t5.data.MixtureRegistry.remove(task_name3)
t5.data.MixtureRegistry.add(
    task_name3,
    [task_name3],
    default_rate=_rate_num_input_examples
)

MODEL_SIZE = "small"
fine_tuning = "fine-tuning_without_pre-training/"
dataset = "new_large"

model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 128, 200),
    "base": (2, 128, 8),
    "large": (8, 64, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]


# Parameters
# Learning rate scheduler

learning_rate_scheduler_picker = "constant"

if learning_rate_scheduler_picker == "slanted":
  selected_learning_rate_scheduler = slanted_triangular
elif learning_rate_scheduler_picker == "isr":
  selected_learning_rate_scheduler = truncated_rsqrt
elif learning_rate_scheduler_picker == "constant":
  selected_learning_rate_scheduler = 0.001
  PATH_GIN_FILE = GIN_PATH



# Training
batch_size = 8

dataset_size = 134238 # code2comment
epochs = 6
checkpoints_save = 1000

number_of_steps = int((dataset_size/batch_size)*epochs)
print("No of steps: ", number_of_steps)



model = t5.models.MtfModel(
    model_dir=MODEL_DIR,
    tpu=None,
    tpu_topology=None,
    model_parallelism=model_parallelism,
    batch_size= batch_size,
    learning_rate_schedule = selected_learning_rate_scheduler,
    sequence_length={"inputs": 512, "targets": 512},
    save_checkpoints_steps=checkpoints_save,
    keep_checkpoint_max=keep_checkpoint_max,
    iterations_per_loop=100,
)


# Start training
with gin.unlock_config():    
    gin.parse_config_file(GIN_PATH)
    TRAIN_STEPS = number_of_steps
    model.train(task_name3, steps=number_of_steps)