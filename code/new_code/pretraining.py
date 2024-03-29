import functools
from operator import le
import os
import time
import warnings

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import t5
from contextlib import contextmanager
import logging as py_logging
from t5.data import postprocessors as t5_postprocessors
from t5.seqio import Feature,SentencePieceVocabulary
from mesh_tensorflow.transformer.learning_rate_schedules import learning_rate_schedule_noam
import gin
import datetime

masked_pretraining_dataset_path = "data/automating_code_review/automating_code_review/dataset/pre-training/pre-training.tsv"

current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
print("current_time:", current_time)

nq_tsv_path = {
    "train": masked_pretraining_dataset_path
}

vocab_model_path = "data/automating_code_review/automating_code_review/tokenizer/TokenizerModel.model"
vocab_path = "data/automating_code_review/automating_code_review/tokenizer/TokenizerModel.vocab"

TaskRegistry = t5.data.TaskRegistry
TfdsTask = t5.data.TfdsTask

def get_default_vocabulary():
  print('----------finsihed get_default_vocabulary-----------')
  return SentencePieceVocabulary(vocab_model_path, 100)

def nq_dataset_fn(split, shuffle_files=True):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(nq_tsv_path[split])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["string","string"],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(lambda *ex: dict(zip(["input", "output"], ex)))
  print('----------finsihed nq_dataset_fn-----------')
  return ds

def preprocessing(ds):
  def to_inputs_and_targets(ex):
        inputs = tf.strings.join([ ex['input']], separator=' ')
        class_label = tf.strings.join([ex['output']], separator=' ')
        return {'inputs': inputs, 'targets': class_label }        
  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


DEFAULT_OUTPUT_FEATURES = {
    "inputs": Feature(
        vocabulary=get_default_vocabulary(), add_eos=False, required=True),

    "targets": Feature(
        vocabulary=get_default_vocabulary(), add_eos=False)
}


print("A few raw train examples...")
for ex in tfds.as_numpy(nq_dataset_fn("train").take(3)):
  print(ex)


#Create a new training task
print("Task registry add-------------------")
t5.data.TaskRegistry.remove('pretraining')
t5.data.TaskRegistry.add(
    "pretraining",
    t5.data.Task,
    dataset_fn=nq_dataset_fn,
    splits=["train", "validation"],
    text_preprocessor=[preprocessing],
    output_features = DEFAULT_OUTPUT_FEATURES,
    metric_fns=[t5.evaluation.metrics.accuracy],
)

nq_task = t5.data.TaskRegistry.get("pretraining")
ds = nq_task.get_dataset(split="train", sequence_length={"inputs": 512, "targets": 512})
print("A  preprocessed training example...")

for ex in tfds.as_numpy(ds.take(1)):
  print(ex)

MODEL_SIZE = "small"  
task_name = "pretraining_code2code"
root_path = "/mnt/nas/meghana/dlmcr/data/"

MODEL_DIR = root_path + "model_checkpoints/" + task_name + "/check_" + current_time

model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 16),
    "base": (2, 128, 8),
    "large": (8, 64, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]

tf.io.gfile.makedirs(MODEL_DIR)
print('--------train_batch_size----------:', train_batch_size)

print('--------model initialization--------------')
model = t5.models.MtfModel(
    model_dir=MODEL_DIR,
    tpu=None,
    tpu_topology=None,
    model_parallelism=model_parallelism,
    batch_size=4,
    sequence_length={"inputs": 512, "targets": 512},
    learning_rate_schedule = learning_rate_schedule_noam,
    save_checkpoints_steps=5000,
    keep_checkpoint_max=None
)

# We used 200000 TRAIN_STEPS
PATH_GIN_FILE = "/mnt/nas/meghana/dlmcr/deep_learning_modern_code_review_ovgu/code/utils/operative_config.gin"

with gin.unlock_config():  
    print('-----------train begins----------------')  
    gin.parse_config_file(PATH_GIN_FILE)
    TRAIN_STEPS = 201357
    model.train("pretraining", steps=TRAIN_STEPS)
    print('-----------train ends----------------')  