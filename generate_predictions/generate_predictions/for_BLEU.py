import statistics
from nltk.translate import bleu_score
from tqdm import tqdm


chencherry = bleu_score.SmoothingFunction()

for BEAM_SIZE in [1, 3, 5, 10]:

    print('BEAM SIZE: ', BEAM_SIZE)

    # change the following path with your correct paths to:
    # - path_targets : targets file
    # - path_predictions : predictions file
    # - path_statistics : the file where the statistics will be saved

    path_targets = '../../dataset/fine-tuning/large/code-to-comment/test.tsv'
    path_predictions = '../../dataset/fine-tuning/large/code-to-comment/predictions_' + str(BEAM_SIZE) + '.txt'
    path_statistics = '../../dataset/fine-tuning/large/code-to-comment/statistics_' + str(BEAM_SIZE) + '.txt'

    tgt = [line.strip() for line in open(path_targets)]
    pred = [line.strip() for line in open(path_predictions)]

    count_perfect = 0
    BLEUscore = []
    for i in tqdm(range(len(tgt))):
        best_BLEU = 0
        target = tgt[i]
        for prediction in pred[i*BEAM_SIZE:i*BEAM_SIZE+BEAM_SIZE]:
            # when BEAM_SIZE > 1 select the best prediction
            if " ".join(prediction.split()) == " ".join(target.split()):
                count_perfect += 1
                best_BLEU = bleu_score.sentence_bleu([target], prediction, smoothing_function=chencherry.method1)
                break
            current_BLEU = bleu_score.sentence_bleu([target], prediction, smoothing_function=chencherry.method1)
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
