from rouge import Rouge
import json
import logging
import numpy as np
import sys
sys.setrecursionlimit(10000)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def metrics_report(reference, hypothesis):

    rouge = Rouge()
    _hypothesis = [" ".join(list(sent)) for sent in hypothesis]
    _reference = [" ".join(list(sent)) for sent in reference]

    report = rouge.get_scores(_hypothesis, _reference, avg=True)
    focus_score = (report["rouge-1"]["f"] +
                       report["rouge-2"]["f"] +
                       report["rouge-l"]["f"])/3
    return report, focus_score

if __name__ == "__main__":
    file_pred = 'data/result/result.jsonl'
    file_ref = 'data/test.jsonl'

    pred_all = [json.loads(line) for line in open(file_pred, "r")]
    ref_all = [json.loads(line) for line in open(file_ref, "r")]

    pred_summ = []; ref_summ = []
    for p, r in zip(pred_all, ref_all):
        pred_summ.append(''.join(p['summary']))
        ref_summ.append(''.join(r['summary']))
    
    reports = []
    for p, r in zip(pred_summ, ref_summ):
        report, focus_score = metrics_report([p], [r])
        reports.append(report)
    
    r1 = []; r2 = []; rl = []
    for r in reports:
        r1.append(r['rouge-1']['f'])
        r2.append(r['rouge-2']['f'])
        rl.append(r['rouge-l']['f'])
    
    print('Score:')
    print('     Rouge-1 = ', np.mean(r1))
    print('     Rouge-2 = ', np.mean(r2))
    print('     Rouge-l = ', np.mean(rl))
    print('')
