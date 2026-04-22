import json, os
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.translate.bleu_score  import corpus_bleu, SmoothingFunction
from nltk.translate.nist_score  import corpus_nist
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer as rouge_lib
from pycocoevalcap.cider.cider import Cider

def tokenize(text):
    """tokenization used for BLEU/NIST/METEOR."""
    return nltk.word_tokenize(text.lower())

def compute_bleu(preds, refs):
    refs_tok  = [[tokenize(r) for r in ref_list] for ref_list in refs]
    preds_tok = [tokenize(p) for p in preds]
    score = corpus_bleu(refs_tok, preds_tok,
                        smoothing_function=SmoothingFunction().method1)
    return round(score * 100, 2)

def compute_nist(preds, refs):
    refs_tok  = [[tokenize(r) for r in ref_list] for ref_list in refs]
    preds_tok = [tokenize(p) for p in preds]
    try:
        return round(corpus_nist(refs_tok, preds_tok, n=5), 4)
    except:
        return 0.0

def compute_meteor(preds, refs):
    scores = []
    for pred, ref_list in zip(preds, refs):
        pred_tok = tokenize(pred)
        # Take max METEOR across all references for this MR
        best = max(meteor_score([r], pred) for r in ref_list)
        scores.append(best)
    return round(np.mean(scores) * 100, 2)

def compute_rouge_l(preds, refs):
    scorer = rouge_lib.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for pred, ref_list in zip(preds, refs):
        best = max(scorer.score(ref, pred)['rougeL'].fmeasure for ref in ref_list)
        scores.append(best)
    return round(np.mean(scores) * 100, 2)

def compute_cider(preds, refs):
    gts = {i: ref_list for i, ref_list in enumerate(refs)}
    res = {i: [preds[i]]  for i in range(len(preds))}
    score, _ = Cider().compute_score(gts, res)
    return round(score, 4)
