import os
import pandas as pd
import argparse
from mt_metrics_eval import meta_info, data, tasks
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm import tqdm 


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
    for metric_name, clean_name in new_metrics.items():
        sys_scores_df = pd.read_csv(f"./results/scores/{metric_name}.sys.score", sep='\t', header=None)
        sys_scores_df.columns = ['metric', 'langpair', 'testset', 'domain', 'reference', 'sys', 'score']
        sys_scores_df = sys_scores_df[sys_scores_df.testset == 'generaltest2024']

        seg_scores_df = pd.read_csv(f"./results/scores/{metric_name}.seg.score", sep='\t', header=None, dtype={9: str})
        if seg_scores_df.shape[1] > 9:
            seg_scores_df = seg_scores_df.iloc[:, :9]
        seg_scores_df.columns = ['metric', 'langpair', 'testset', 'domain', 'doc', 'reference', 'sys', 'seg_id', 'score']
        seg_scores_df = seg_scores_df[seg_scores_df.testset == 'generaltest2024']

        add_metrics_to_eval_sets(evs_dict, lps, clean_name, sys_scores_df, seg_scores_df)
    return evs_dict

def run_eval(evalset, lps, evs_dict, metrics, k=0):
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

def perm_input_test(metric_name, clean_metric_name, challenge_set, n_samples=200, p=0.05):
    seg_scores_df= pd.read_csv(f"./results/scores/{metric_name}.seg.score", sep='\t', header=None,  dtype={9: str})
    if seg_scores_df.shape[1] > 10:
            seg_scores_df = seg_scores_df.iloc[:, :10]
    seg_scores_df.columns = ['metric', 'langpair', 'testset', 'domain', 'doc', 'reference', 'sys', 'seg_id', 'score', 'human_score']
    if challenge_set=='challenge_bioMQM':
        seg_scores_df=seg_scores_df[~seg_scores_df.langpair.isin(['pt-en','ru-en','zh-en'])]
    challenge_df = seg_scores_df[seg_scores_df.testset == challenge_set]
    language_pairs = challenge_df['langpair'].unique()
    rows = []

    def compute_perm(metric_scores, human_scores):
        def clean(x):
            x = str(x).strip().replace('`', '')  # Remove accidental backticks and spaces
            return x

        metric_scores = np.array([clean(x) for x in metric_scores], dtype=float)
        human_scores = np.array([clean(x) for x in human_scores], dtype=float)

        mask = (~np.isnan(metric_scores)) & (~np.isnan(human_scores))
        metric_scores = metric_scores[mask]
        human_scores = human_scores[mask]

        results = {}
        for name, corr_fn in [("spearman", spearmanr), ("kendall", kendalltau), ("pearson", pearsonr)]:
            true_corr, _ = corr_fn(metric_scores, human_scores)

            null_distribution = []
            for _ in range(n_samples):
                permuted = np.random.permutation(human_scores)
                corr, _ = corr_fn(metric_scores, permuted)
                null_distribution.append(corr)

            null_distribution = np.array(null_distribution)
            p_val = np.mean(np.abs(null_distribution) >= np.abs(true_corr))
            is_significant = p_val < p

            results[name] = (true_corr, p_val, is_significant)

        return results
        
    # Compute over ALL segments
    correlations = compute_perm(challenge_df['score'], challenge_df['human_score'])
    for corr_type, (true_corr, p_val, is_significant) in correlations.items():
        rows.append({
            "metric_name": clean_metric_name,
            "language_pair": "all",
            "correlation_type": corr_type,
            "true_correlation": true_corr,
            "p_value": p_val,
            "significant": is_significant
        })

    # Compute per-language pair
    for lp in language_pairs:
        lp_df = challenge_df[challenge_df['langpair'] == lp]

        correlations = compute_perm(lp_df['score'], lp_df['human_score'])
        for corr_type, (true_corr, p_val, is_significant) in correlations.items():
            rows.append({
                "metric_name": clean_metric_name,
                "language_pair": lp,
                "correlation_type": corr_type,
                "true_correlation": true_corr,
                "p_value": p_val,
                "significant": is_significant
            })
    
    df = pd.DataFrame(rows)
    return df

def print_latex_summary(df):
    all_df = df[df['language_pair'] == 'all']
    pivot_df = all_df.pivot(index='metric_name', columns='correlation_type', values='true_correlation')
    pivot_df = pivot_df[['spearman', 'kendall', 'pearson']]
    pivot_df['average'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values(by='average', ascending=False)
    latex_table = pivot_df.drop(columns='average').to_latex(float_format="%.3f")
    print(latex_table)
    row_order = pivot_df.index.tolist()
    return row_order

def print_latex_all(df, row_order):
    # Pivot the table: rows = metric_name, columns = (language_pair, correlation_type), values = true_correlation
    pivot_df = df.pivot_table(index='metric_name', columns=['language_pair', 'correlation_type'], values='true_correlation')
    columns_order = []
    for lp in df['language_pair'].unique():
        columns_order.extend([(lp, 'spearman'), (lp, 'kendall'), (lp, 'pearson')])

    pivot_df = pivot_df[columns_order]
    pivot_df = pivot_df.loc[row_order]

    def bold_max_in_column(col):
        max_value = col.max()
        return col.apply(lambda x: f"\\textbf{{{x:.3f}}}" if x == max_value else f"{x:.3f}")
    
    pivot_df = pivot_df.apply(bold_max_in_column)
    
    latex_table = pivot_df.to_latex(float_format="%.3f")
    print(latex_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evalset", type=str, choices=["wmt24_prim", "wmt24_sec", "afrimte", "indicmte", 'biomqm'], required=True,
        help="Which eval set to process"
    )
    parser.add_argument(
        "--k", default=0,
        help="Number of resampling runs for WMT24"
    )
    args = parser.parse_args()
    new_metrics_dict = {'comet_xlmr':'COMET-T XLMR', 
         'comet_bert': 'COMET-T mBERT', 
         'comet_mt5': 'COMET-T mT5', 
         'comet_nllb': 'COMET-T NLLB', 
         'comet_glot500':'COMET-T GloT-500', 
         'comet_labse': 'COMET-T LaBSE', 
         'comet_indic': 'COMET-T IndicBERT', 
         'comet_afro': 'COMET-T Afro-XLMR',
         'comet_reg_xlmr': 'COMET-E XLMR', 
         'comet_reg_bert': 'COMET-E mBERT', 
         'comet_reg_mt5': 'COMET-E mT5', 
         'comet_reg_nllb': 'COMET-E NLLB',
         'comet_reg_glot500': 'COMET-E GloT-500',
         'comet_reg_labse': 'COMET-E LaBSE', 
         'comet_reg_indic': 'COMET-E IndicBERT', 
         'comet_reg_afro': 'COMET-E Afro-XLMR'}

    MAP_CHALLENGE_SETS = {'afrimte':'challenge_AfriMTE',
    "indicmte":"challenge_IndicMTE", 
    'biomqm':'challenge_bioMQM'}
    

    if "wmt24" in args.evalset:
        new_metrics_dict.update({'comet_20':"COMET-20"})
        metrics = {
        'BERTScore', 'BLEU', 'BLEURT-20', 
        'COMET-22', 'MetricX-24-Hybrid'
        }
        metrics.update(new_metrics_dict.values())
        if args.evalset=="wmt24_prim": #Primary meta-evaluation
            lps, evs_dict = process_evalset("mqm")
            evs_dict=load_and_add_scores(lps, evs_dict, new_metrics_dict)
            run_eval("mqm", lps, evs_dict, metrics, k=args.k)

        elif args.evalset=="wmt24_sec": #Secondary meta-evaluation
            lps, evs_dict = process_evalset("esa")
            evs_dict=load_and_add_scores(lps, evs_dict, new_metrics_dict)
            run_eval("esa", lps, evs_dict, metrics, k=args.k)
    
    elif args.evalset=='afrimte' or args.evalset=='biomqm' or args.evalset=='indicmte':
        new_metrics_dict.update({'llm':"Gemma-inference"})
        challenge_set=MAP_CHALLENGE_SETS[args.evalset]
        OUTPUT_PATH=f"./results/meta_evaluation/{args.evalset}.csv"
        if os.path.exists(OUTPUT_PATH):
            all_results=pd.read_csv(OUTPUT_PATH)
        else:
            all_df=[]
            for metric_name, clean_name in tqdm(new_metrics_dict.items()):
                all_df.append(perm_input_test(metric_name, clean_name, challenge_set))
            all_results=pd.concat(all_df)
            all_results.to_csv(OUTPUT_PATH, index=False)
        
        row_order=print_latex_summary(all_results)
        print_latex_all(all_results, row_order)
        
