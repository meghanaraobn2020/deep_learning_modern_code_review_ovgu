import statistics
import subprocess
from tqdm import tqdm
import javalang
import pandas as pd


root_path = '/mnt/nas/meghana/dlmcr/'


def tokenize_java(code):
    token_gen = javalang.tokenizer.tokenize(code)
    tokens = []
    indexes = []
    while (True):
        try:
            token = next(token_gen)
        except:
            break
        tokens.append(token)

    pure_tokens = [token.value for token in tokens]

    return pure_tokens


def code_bleu(ref, hyp):
    f = open(root_path + 'data/dataset/dataset/fine-tuning/new_large/code-to-code/predictions/isr_learning_rate/ref.txt', 'w')
    f.write(ref)
    f.close()
    f = open(root_path + 'data/dataset/dataset/fine-tuning/new_large/code-to-code/predictions/isr_learning_rate/hyp.txt', 'w')
    f.write(hyp)
    f.close()

    f = open('codeBleu.sh', 'w')
    f.write('#!/usr/bin/env bash\n')
    f.write('cd ./CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU && ' +
            'python3 calc_code_bleu.py --refs ' + path_ref +
            ' --hyp ' + path_hyp + ' --lang java --params 0.25,0.25,0.25,0.25' +
            ' > bleu.log')
    f.close()

    subprocess.run('./codeBleu.sh', shell=True)
    result = [line.strip() for line in open(path_result)]
    if len(result) == 3:
        f = open('codeBleu.sh', 'w')
        f.write('#!/usr/bin/env bash\n')
        f.write('cd ./CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU && ' +
                'python3 calc_code_bleu.py --refs ' + path_ref +
                ' --hyp ' + path_hyp + ' --lang java --params 0.25,0.25,0.25,0' +
                ' > bleu.log')
        f.close()
        subprocess.run('./codeBleu.sh', shell=True)
        result = [line.strip() for line in open(path_result)]
        return float(result[-1].split()[2])
    try:
        print(result)
        return float(result[-1].split()[2])
    except:
        print('.............................................WARNING.............................................')
        return 0


# for BEAM_SIZE in [1, 3, 5, 10]:
for BEAM_SIZE in [10]:

    print('BEAM SIZE: ', BEAM_SIZE)

    # change the following path with your correct paths to:
    # - path_targets : targets file
    # - path_predictions : predictions file
    # - path_statistics : the file where the statistics will be saved

    path_targets = root_path + 'data/dataset/dataset/fine-tuning/new_large/code-to-code/test.tsv'
    path_predictions = root_path + 'data/dataset/dataset/fine-tuning/new_large/code-to-code/predictions/isr_learning_rate/predictions_' + str(BEAM_SIZE) + '.txt'
    path_statistics = root_path + 'data/dataset/dataset/fine-tuning/new_large/code-to-code/predictions/isr_learning_rate/statistics_' + str(BEAM_SIZE) + '.txt'

    df = pd.read_csv(path_targets, sep='\t', names=['source', 'target'])

    tgt = df['target']
    pred = [line.strip() for line in open(path_predictions)]

    path_ref = root_path + 'data/dataset/dataset/fine-tuning/new_large/code-to-code/predictions/isr_learning_rate/ref.txt'
    path_hyp = root_path + 'data/dataset/dataset/fine-tuning/new_large/code-to-code/predictions/isr_learning_rate/hyp.txt'
    path_result = root_path + 'deep_learning_modern_code_review_ovgu/generate_predictions/generate_predictions/CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU/bleu.log'

    count_perfect = 0
    # BLEUscore = [1,2,3,4]
    BLEUscore = []

    for i in tqdm(range(len(tgt))):
        best_BLEU = 0
        target = tgt[i]
        for prediction in pred[i*BEAM_SIZE:i*BEAM_SIZE+BEAM_SIZE]:
            # when BEAM_SIZE > 1 select the best prediction
            try:
                current_pred = " ".join(tokenize_java(prediction))
                current_tgt = " ".join(tokenize_java(target))
            except:
                current_pred = prediction
                current_tgt = target

            if " ".join(current_pred.split()) == " ".join(current_tgt.split()):
                count_perfect += 1
                try:
                    best_BLEU = code_bleu(current_tgt, current_pred)
                except:
                    current_pred = prediction
                    current_tgt = target
                    best_BLEU = code_bleu(current_tgt, current_pred)
                break
            try:
                current_BLEU = code_bleu(current_tgt, current_pred)
            except:
                current_pred = prediction
                current_tgt = target
                current_BLEU = code_bleu(current_tgt, current_pred)
            if current_BLEU > best_BLEU:
                best_BLEU = current_BLEU
        BLEUscore.append(best_BLEU)

    print(f'PP    : %d/%d (%s%.2f)' % (count_perfect, len(tgt), '%', (count_perfect * 100) / len(tgt)))
    print(f'BLEU mean              : ', statistics.mean(BLEUscore))
    print(f'BLEU median            : ', statistics.median(BLEUscore))
    print(f'BLEU stdev             : ', statistics.stdev(BLEUscore))

    f = open(path_statistics, 'w+')
    f.write(f'PP     : %d/%d (%s%.2f)' % (count_perfect, len(tgt), '%', (count_perfect * 100) / len(tgt)))
    f.write('\nBLEU mean              : ' + str(statistics.mean(BLEUscore)))
    f.write('\nBLEU median            : ' + str(statistics.median(BLEUscore)))
    f.write('\nBLEU stdev             : ' + str(statistics.stdev(BLEUscore)))
    f.close()
