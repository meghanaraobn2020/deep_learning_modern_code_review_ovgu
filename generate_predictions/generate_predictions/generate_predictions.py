from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class EvalDataset(torch.utils.data.Dataset):
    samples = []

    def __init__(self, data_dir_path, task):

        self.samples = []

        df = pd.read_csv('deep_learning_modern_code_review_ovgu/test_dataset/test_new.tsv', sep='\t', names=['source', 'target'])
        source = df['source']
        target = df['target']

        f_source = open('test.source', 'w+', encoding='utf-8')
        f_target = open('test.target', 'w+', encoding='utf-8')
        for j in range(len(df)):
            f_source.write(task + source[j] + '\n')
            f_target.write(target[j] + '\n')
        f_source.close()
        f_target.close()

        input_file = open('test.source', 'r', encoding='utf-8')
        output_file = open('test.target', 'r', encoding='utf-8')

        lines_input = input_file.readlines()
        output_lines = output_file.readlines()
        print('data: ', len(lines_input))

        for (inp, out) in zip(lines_input, output_lines):
            self.samples.append((inp.rstrip(), out.rstrip()))

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


beam_size = 10
batch_size = 2
task = 'code2comment: '  # possible options: 'code2code: ', 'code&comment2code: ', 'code2comment: '
data_dir = "data/dataset/dataset/fine-tuning/new_large/code-to-comment/"  # change the path if needed- ../../dataset/fine-tuning/large/code-to-code/"
tokenizer_name = "deep_learning_modern_code_review_ovgu/tokenizer/tokenizer/TokenizerModel.model" #"../../tokenizer/TokenizerModel.model" 
model_name_or_path ="deep_learning_modern_code_review_ovgu/dumps/pre_training_code2comment/isr_learning_rate/pytorch_model.bin" #"./dumps/pytorch_model.bin" 
config_name = "deep_learning_modern_code_review_ovgu/generate_predictions/generate_predictions/config.json" #"./config.json" 

dataset = EvalDataset(data_dir, task)
dloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu',)  # GPU recommended

t5_tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

t5_config = T5Config.from_pretrained(config_name)
t5_mlm = T5ForConditionalGeneration.from_pretrained(model_name_or_path, config=t5_config).to(DEVICE)

# GENERATE PREDICTIONS
f_pred = open(data_dir + 'predictions/pre_training_code2comment/isr_learning_rate/new_data/without_finetune/beam_size_10/predictions_' + str(beam_size) + '.txt', 'w+')
predictions = []

# indexes for batches
old = 0
new = batch_size * beam_size

for batch in tqdm(dloader):
    
    encoded = t5_tokenizer.batch_encode_plus(batch[0], add_special_tokens=False, return_tensors='pt', padding=True)

    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    outputs = t5_mlm.generate(
        input_ids=input_ids,
        max_length=512,  # Change here
        num_beams=beam_size,
        attention_mask=attention_mask,
        early_stopping=True,
        num_return_sequences=beam_size).to(DEVICE)
    
    

    predictions.extend(t5_tokenizer.batch_decode(outputs, skip_special_tokens=True))

    to_analyze = predictions[old:new]
    target_list = batch[1]
    input_list = batch[0]

    idx = 0
    for (input_item, target_item) in zip(input_list, target_list):
        target_item = " ".join(target_item.split(' '))
        for i in range(beam_size):
            f_pred.write(to_analyze[idx] + '\n')
            idx += 1

    old = new
    new = new + (batch_size * beam_size)


f_pred.close()
