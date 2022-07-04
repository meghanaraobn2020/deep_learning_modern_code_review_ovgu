import functools
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

nq_tsv_path_code_code_large = {
    "train":      'data/automating_code_review/automating_code_review/dataset/fine-tuning/new_large/code-to-code/train.tsv',
    "validation": 'data/automating_code_review/automating_code_review/dataset/fine-tuning/new_large/code-to-code/val.tsv'
}

data_train = len([line for line in open('data/automating_code_review/automating_code_review/dataset/fine-tuning/new_large/code-to-code/train.tsv', 'r')])
data_val = len([line for line in open('data/automating_code_review/automating_code_review/dataset/fine-tuning/new_large/code-to-code/val.tsv', 'r')])

num_nq_examples_code_code_large = dict(train=data_train, validation=data_val)

#Model vocab and path
vocab_model_path = "data/automating_code_review/automating_code_review/tokenizer/TokenizerModel.model"
vocab_path = "data/automating_code_review/automating_code_review/tokenizer/TokenizerModel.vocab"

TaskRegistry = t5.data.TaskRegistry
TfdsTask = t5.data.TfdsTask

def get_default_vocabulary():
  return SentencePieceVocabulary(vocab_model_path, 100)

DEFAULT_OUTPUT_FEATURES = {
    "inputs": Feature(
        vocabulary=get_default_vocabulary(), add_eos=True, required=False),

    "targets": Feature(
        vocabulary=get_default_vocabulary(), add_eos=True)
}

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

print("A few raw validation examples...")
for ex in tfds.as_numpy(nq_dataset_code_code_large("validation").take(2)):
  print(ex)
print("A few raw training examples...")
for ex in tfds.as_numpy(nq_dataset_code_code_large("train").take(2)):
  print(ex)

def code_code_preprocessing(ds):
  def to_inputs_and_targets(ex):
        inputs = tf.strings.join(['code2code: ' + ex['input']], separator=' ')
        class_label = tf.strings.join([ex['output']], separator=' ')
        return {'inputs': inputs, 'targets': class_label }
    
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
t5.data.TaskRegistry.remove('code_to_code_new_large')
t5.data.TaskRegistry.add(
    "code_to_code_new_large",
    dataset_fn=nq_dataset_code_code_large,
    splits=["train", "validation"],
    text_preprocessor=[code_code_preprocessing],
    output_features = DEFAULT_OUTPUT_FEATURES,
    metric_fns=[t5.evaluation.metrics.accuracy],
    num_input_examples=num_nq_examples_code_code_large
)

nq_task = t5.data.TaskRegistry.get("code_to_code_new_large")
ds = nq_task.get_dataset(split="train", sequence_length={"inputs": 512, "targets": 512})
print("A few preprocessed training examples...")
for ex in tfds.as_numpy(ds.take(3)):
  print(ex)


#Setting up fine tuning tasks
def _rate_num_input_examples(task):
  if "train" in task.splits:
    return float(task.num_input_examples("train"))
  elif "validation" in task.splits:
    return float(task.num_input_examples("validation"))
  else:
    raise ValueError("Task %s does not have a train or validation split." % (task.name))

t5.data.MixtureRegistry.remove("code_to_code_new_large")
t5.data.MixtureRegistry.add(
    "code_to_code_new_large",
    ["code_to_code_new_large"],
    default_rate=_rate_num_input_examples
)

MODEL_SIZE = "small"
fine_tuning = "fine-tuning_without_pre-training/"
dataset = "new_large"

MODEL_DIR = 'data/automating_code_review/automating_code_review/model_dumps/fine-tuning_without_pre-training/new_large_dataset/code-to-code/'

model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 128, 200),
    "base": (2, 128, 8),
    "large": (8, 64, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]

# Learning rate scheduler

learning_rate_scheduler_picker = "constant"

if learning_rate_scheduler_picker == "slanted":
  selected_learning_rate_scheduler = slanted_triangular
  #PATH_GIN_FILE = 'gs://' + bucket_name + '/automating_code_review/utils/operative_config_slanted.gin'
elif learning_rate_scheduler_picker == "isr":
  selected_learning_rate_scheduler = truncated_rsqrt
  #PATH_GIN_FILE = 'gs://' + bucket_name + '/automating_code_review/utils/operative_config_isr.gin'
elif learning_rate_scheduler_picker == "polynomial":
  selected_learning_rate_scheduler = learning_rate_fn
  #PATH_GIN_FILE = 'gs://' + bucket_name + '/automating_code_review/utils/operative_config_polynomial.gin'
elif learning_rate_scheduler_picker == "constant":
  selected_learning_rate_scheduler = 0.001
  PATH_GIN_FILE = 'data/automating_code_review/automating_code_review/utils/operative_config_constant.gin'

number_of_steps = 500

tf.io.gfile.makedirs(MODEL_DIR)

model = t5.models.MtfModel(
    model_dir=MODEL_DIR,
    tpu=None,
    tpu_topology=None,
    model_parallelism=model_parallelism,
    batch_size= 2,
    learning_rate_schedule = selected_learning_rate_scheduler,
    sequence_length={"inputs": 512, "targets": 512},
    save_checkpoints_steps=10000,
    keep_checkpoint_max=keep_checkpoint_max,
    iterations_per_loop=100,
)

#Start training

# NON PRETRAINED
with gin.unlock_config():    
    gin.parse_config_file("data/automating_code_review/automating_code_review/utils/operative_config_constant.gin")
    TRAIN_STEPS = number_of_steps
    model.train('code_to_code_new_large', steps=number_of_steps)

#Evaluation

# model.batch_size = 1024
# model.eval(
#     mixture_or_task_name='code_to_code_new_large',
#     # -1 will evaluate the last checkpoint, you can also provide 
#     # a list of checkpoints with the following format : [10000, 20000, 30000]
#     checkpoint_steps=-1,
#     split="validation" #nq_tsv_path_code_code_large[split]
#     )

# #Confidence score
# inputs_file = './source.txt'
# pred_file = 'predictions.txt'
# score_file = 'score'
# model.score(inputs=inputs_file,
#             targets=pred_file,
#             scores_file=score_file,
#             checkpoint_steps='best',
#             vocabulary=get_default_vocabulary())

# confidence_score = [math.exp(float(line.split)) for line in open('./score.score', 'r')]

# f = open('./confidence_score.txt', 'w+')
# for i in range(len(confidence_score)):
#   f.write(str(confidence_score[i]) + '\n')
# f.close()
