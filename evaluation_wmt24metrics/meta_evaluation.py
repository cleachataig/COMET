from mt_metrics_eval import meta_info
from mt_metrics_eval import data
from mt_metrics_eval import tasks

import pandas as pd
# @title Add metric scores to EvalSets

# Compute scores for each language pair, and add to the appropriate EvalSet.
# Setting replace=True makes this work if we want to iterate over different
# versions of the metric.
new_metrics= ['comet_xlmr', 'comet_bert', 'comet_mt5', 'comet_nllb', 'comet_glot500', 
              'comet_labse', 'comet_reg_xlmr', 'comet_reg_bert', 'comet_reg_mt5', 'comet_reg_nllb', 'comet_reg_glot500', 'comet_reg_labse']

wmt24_lps = ['en-de', 'en-es', 'ja-zh']
evs_dict = {('wmt24', lp): data.EvalSet('wmt24', lp, True) for lp in wmt24_lps}

for metric_name in new_metrics:
    sys_scores_df=pd.read_csv(f"./scores/{metric_name}.sys.score", sep='\t', header=None)
    sys_scores_df.columns=['metric', 'langpair', 'testset', 'domain', 'reference', 'sys', 'score']
    sys_scores_df=sys_scores_df.loc[sys_scores_df.testset=='generaltest2024', :]

    seg_scores_df=pd.read_csv(f"./scores/{metric_name}.seg.score", sep='\t', header=None)
    seg_scores_df.columns=['metric', 'langpair', 'testset', 'domain', 'doc', 'reference', 'sys', 'seg_id', 'score']
    seg_scores_df=seg_scores_df.loc[seg_scores_df.testset=='generaltest2024', :]

    for lp in wmt24_lps:
        evs = evs_dict[('wmt24', lp)]
        for refname, ref in evs.all_refs.items():
            sys_scores = sys_scores_df.loc[(sys_scores_df.langpair==lp)&(sys_scores_df.reference==refname), :]
            sys_scores = sys_scores[['sys', 'score']].groupby('sys').mean()
            sys_scores=sys_scores.groupby('sys')['score'].apply(list).to_dict()

            seg_scores=seg_scores_df.loc[(seg_scores_df.langpair==lp)&(seg_scores_df.reference==refname), :]
            seg_scores = seg_scores[['sys', 'seg_id', 'score']].groupby(['sys', 'seg_id']).mean()
            seg_scores=seg_scores.groupby('sys')['score'].apply(list).to_dict()

            evs.AddMetric(metric_name, {refname}, 'sys', sys_scores, replace=True)
            evs.AddMetric(metric_name, {refname}, 'seg', seg_scores, replace=True)

# Add new metric to the primary lists, so it will get picked up when tasks get
# run with primary=True (avoiding having to evaluate all contrastive
# submissions as well).

metrics={'BERTScore', 'BLEU', 'BLEURT-20', 'metametrics_mt_mqm_hybrid_kendall', 'XCOMET', 'gemba_esa', 'COMET-22'}
metrics.update(new_metrics)

for evs in evs_dict.values():
  evs.SetPrimaryMetrics(metrics)

# @title Generate results with new metric

wmt24_tasks, main_task_weights = tasks.WMT24(wmt24_lps, k=1000)
new_results = wmt24_tasks.Run(eval_set_dict=evs_dict)

# @title Print results

# Results show all primary metrics, along with the new 'lendiff' metric.

avg_corrs, matrix = new_results.AverageCorrMatrix(main_task_weights)

table = new_results.Table(
    metrics=list(avg_corrs),
    initial_column=avg_corrs,
    initial_column_header='avg-corr',
    attr_list=['lang', 'level', 'corr_fcn'],
    nicknames={'KendallWithTiesOpt': 'acc-eq', 'pce': 'SPA'},
    fmt='text',
    baselines_metainfo=meta_info.WMT24)

print(table)
print()
print(tasks.MatrixString(avg_corrs, matrix, probs=True))
