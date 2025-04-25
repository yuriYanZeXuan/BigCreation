import json
import logging
import time
import torch
import fire
import numpy as np
from datasets import load_dataset
from bert_score import score as bert_score  # 新增导入
from nltk.translate.meteor_score import meteor_score  # 导入METEOR
import os 
os.environ['MOVERSCORE_MODEL'] = "albert-base-v2"
try:
    import jieba  # type: ignore
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore
    from rouge_chinese import Rouge  # type: ignore
    from moverscore_v2 import get_idf_dict, word_mover_score 
    from collections import defaultdict

    jieba.setLogLevel(logging.CRITICAL)
    jieba.initialize()
except ImportError:
    print("Please install llamafactory with `pip install -e .[metrics]`.")
    raise


def compute_metrics(sample):
    hypothesis = list(jieba.cut(sample["predict"]))
    reference = list(jieba.cut(sample["label"]))

    bleu_score = sentence_bleu(
        [list(sample["label"])],
        list(sample["predict"]),
        smoothing_function=SmoothingFunction().method3,
    )

    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]

    metric_result = {}
    for k, v in result.items():
        metric_result[k] = round(v["f"] * 100, 4)

    metric_result["bleu-4"] = round(bleu_score * 100, 4)
    
    meteor = meteor_score([sample["label"].split()], sample["predict"].split())  # 使用原始文本计算METEOR
    metric_result["meteor"] = round(meteor * 100, 4)  # 添加METEOR得分

    return metric_result


def main(filename: str):
    start_time = time.time()
    dataset = load_dataset("json", data_files=filename, split="train")
    
    # 先提取原始数据
    predictions = [s["predict"] for s in dataset]
    references = [s["label"] for s in dataset]
    
    # idf_dict_hyp = get_idf_dict(predictions) # idf_dict_hyp = defaultdict(lambda: 1.)
    # idf_dict_ref = get_idf_dict(references) # idf_dict_ref = defaultdict(lambda: 1.)

    # scores = word_mover_score(references, predictions, idf_dict_ref, idf_dict_hyp, \
    #                         stop_words=[], n_gram=1, remove_subwords=True)
    # print(f"Word Mover Score: {np.mean(scores):.4f}")
    # 计算BERTScore
    print("Calculating BERTScore...")
    _, _, bert_f1 = bert_score(
        predictions,
        references,
        lang="zh" if any("\u4e00" <= c <= "\u9fff" for c in "".join(predictions)) else "en",
        model_type="bert-base-chinese" if any("\u4e00" <= c <= "\u9fff" for c in "".join(predictions)) else "roberta-large",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=128
    )
    bert_score_avg = round(np.mean(bert_f1.numpy()) * 100, 4)

    # 汇总结果
    average_score = {}
    # 计算基础指标
    dataset = dataset.map(compute_metrics, num_proc=8, remove_columns=dataset.column_names)
    score_dict = dataset.to_dict()

    for task, scores in sorted(score_dict.items(), key=lambda x: x[0]):
        print(f"{task}: {sum(scores) / len(scores):.4f}")
        average_score[task] = sum(scores) / len(scores)
    
    average_score["bertscore"] = bert_score_avg
    print(f"Average BERTScore: {average_score['bertscore']:.4f}")

    with open("predictions_score.json", "w", encoding="utf-8") as f:
        json.dump(average_score, f, indent=4)

    print(f"\nDone in {time.time() - start_time:.3f}s.\nScore file saved to predictions_score.json")


if __name__ == "__main__":
    fire.Fire(main)
