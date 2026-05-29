def compute_metrics(preds, refs):
    out = {}
    try:
        import sacrebleu
        out['BLEU'] = float(sacrebleu.corpus_bleu(preds, [refs]).score)
    except Exception:
        out['BLEU'] = 0.0
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        vals = [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)]
        out['ROUGE_L'] = float(sum(vals) / max(1, len(vals)))
    except Exception:
        out['ROUGE_L'] = 0.0
    return out
