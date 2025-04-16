import pandas as pd
import argparse
from mt_metrics_eval import meta_info, data, tasks
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


def add_metrics_to_eval_sets(evs_dict, lps, metric_name, sys_scores_df, seg_scores_df):
    for lp in lps:
        evs = evs_dict[('wmt24', lp)]
        for refname in evs.all_refs:
            sys_scores = sys_scores_df[
                (sys_scores_df.langpair == lp) & (sys_scores_df.reference == refname)
            ][['sys', 'score']].groupby('sys').mean()
            sys_scores = sys_scores.groupby('sys')['score'].apply(list).to_dict()

            seg_scores = seg_scores_df[
                (seg_scores_df.langpair == lp) & (seg_scores_df.reference == refname)
            ][['sys', 'seg_id', 'score']].groupby(['sys', 'seg_id']).mean()
            seg_scores = seg_scores.groupby('sys')['score'].apply(list).to_dict()

            evs.AddMetric(metric_name, {refname}, 'sys', sys_scores, replace=True)
            evs.AddMetric(metric_name, {refname}, 'seg', seg_scores, replace=True)

def process_evalset(evalset):
    if evalset == "mqm":
        lps = ['en-de', 'en-es', 'ja-zh']
    elif evalset == "esa":
        lps = ['cs-uk', 'en-cs', 'en-hi', 'en-is', 'en-ja', 'en-ru', 'en-uk', 'en-zh']
    else:
        raise ValueError("Invalid evalset name. Choose 'mqm' or 'esa'.")

    evs_dict = {('wmt24', lp): data.EvalSet('wmt24', lp, True) for lp in lps}
    return lps, evs_dict

def load_and_add_scores(lps, evs_dict, new_metrics):
    for metric_name in new_metrics:
        sys_scores_df = pd.read_csv(f"./scores/{metric_name}.sys.score", sep='\t', header=None)
        sys_scores_df.columns = ['metric', 'langpair', 'testset', 'domain', 'reference', 'sys', 'score']
        sys_scores_df = sys_scores_df[sys_scores_df.testset == 'generaltest2024']

        seg_scores_df = pd.read_csv(f"./scores/{metric_name}.seg.score", sep='\t', header=None)
        if seg_scores_df.shape[1] > 9:
            seg_scores_df = seg_scores_df.iloc[:, :9]
        seg_scores_df.columns = ['metric', 'langpair', 'testset', 'domain', 'doc', 'reference', 'sys', 'seg_id', 'score']
        seg_scores_df = seg_scores_df[seg_scores_df.testset == 'generaltest2024']

        add_metrics_to_eval_sets(evs_dict, lps, metric_name, sys_scores_df, seg_scores_df)
    return evs_dict

def run_eval(evalset, lps, evs_dict, metrics, k=1000):
    for evs in evs_dict.values():
        evs.SetPrimaryMetrics(metrics)

    if evalset == "mqm":
        wmt24_tasks, task_weights = tasks.WMT24(lps, k=k)
        results = wmt24_tasks.Run(eval_set_dict=evs_dict)
        if k==0:
            avg_corrs = results.AverageCorrs(task_weights)
        else:
            avg_corrs, matrix = results.AverageCorrMatrix(task_weights)

    elif evalset == "esa":
        esa_tasks, task_weights = tasks.WMT24(k=k, primary=True, lps=lps)
        results = esa_tasks.Run(eval_set_dict=evs_dict)
        if k==0:
            avg_corrs = results.AverageCorrs(task_weights)
        else:
            avg_corrs, matrix = results.AverageCorrMatrix(task_weights)
        
    all_lps = ','.join(sorted(lps))

    print(results.Table(
        metrics=list(avg_corrs),
        initial_column=avg_corrs,
        initial_column_header='avg-corr',
        attr_list=['lang', 'level', 'corr_fcn'],
        nicknames={'KendallWithTiesOpt': 'acc-eq', 'pce': 'SPA', all_lps: 'all'},
        fmt='latex',
        baselines_metainfo=meta_info.WMT24
    ))
    if k!=0:
        print()
        print(tasks.MatrixString(avg_corrs, matrix, probs=True))

def perm_input_test(metric_scores, human_scores, n_samples=200, p=0.05):
    #TODO FOR CHALLENGE SETS
    metric_scores = np.array(metric_scores)
    human_scores = np.array(human_scores)

    true_corr, _ = spearmanr(metric_scores, human_scores)
    null_distribution = []

    for _ in range(n_samples):
        permuted = np.random.permutation(human_scores)
        corr, _ = spearmanr(metric_scores, permuted)
        null_distribution.append(corr)

    null_distribution = np.array(null_distribution)
    p_val = np.mean(np.abs(null_distribution) >= np.abs(true_corr))

    is_significant = p_val < p
    return {
        "true_correlation": true_corr,
        "p_value": p_val,
        "significant": is_significant,
        "null_distribution": null_distribution
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evalset", type=str, choices=["wmt24_prim", "wmt24_sec", "afrimte", "indicmte"], required=True,
        help="Which eval set to process"
    )
    args = parser.parse_args()
    new_metrics = [
        #'comet_20', 
        'comet_xlmr', 'comet_bert', 'comet_mt5', 'comet_nllb',
        'comet_glot500', 'comet_labse', 'comet_reg_xlmr', 'comet_reg_bert',
        'comet_reg_mt5', 'comet_reg_nllb', 'comet_reg_glot500', 'comet_reg_labse'
    ]

    if "wmt24" in args.evalset:
        metrics = {
        'BERTScore', 'BLEU', 'BLEURT-20', 'metametrics_mt_mqm_hybrid_kendall',
        'XCOMET', 'gemba_esa', 'COMET-22', 'MetricX-24-Hybrid'
        }
        metrics.update(new_metrics)
        if args.evalset=="wmt24_prim": #Primary meta-evaluation
            lps, evs_dict = process_evalset("mqm")
            evs_dict=load_and_add_scores(lps, evs_dict, new_metrics)
            run_eval("mqm", lps, evs_dict, metrics, k=0)

        elif args.evalset=="wmt24_sec": #Secondary meta-evaluation
            lps, evs_dict = process_evalset("esa")
            evs_dict=load_and_add_scores(lps, evs_dict, new_metrics)
            run_eval("esa", lps, evs_dict, metrics)
    
    elif args.evalset=='afrimte':
        print('afrimte')

    elif args.evalset=='afrimte':
        print('indicmte')