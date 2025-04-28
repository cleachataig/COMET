'''Script adapted from the WMT24 shared metrics task
Assume the data folder "metrics_inputs" has been downloaded in the DATA_FOLDER'''

import argparse
import os

import pandas as pd
from sacrebleu import corpus_bleu, corpus_chrf, sentence_bleu, sentence_chrf
import numpy as np
from tqdm import tqdm
from typing import Dict, List

from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch
import openai

DATA_FOLDER="./data/"
OUTPUT_FOLDER='./results/scores/'

SRC_PATH = DATA_FOLDER+"metrics_inputs/txt/{}/sources/{}.{}.src.{}"
REFA_PATH = DATA_FOLDER+"metrics_inputs/txt/{}/references/{}.{}.ref.refA.{}"
REFB_PATH = DATA_FOLDER+"metrics_inputs/txt/{}/references/{}.{}.ref.refB.{}"
CS_REF_PATH = DATA_FOLDER+"metrics_inputs/txt/{}/references/{}.{}.ref.ref1.{}"
CS_HUMAN_PATH = DATA_FOLDER+"metrics_inputs/txt/{}/human_scores/{}.{}.{}.score.txt"
SYSTEM_FOLDER = DATA_FOLDER+"metrics_inputs/txt/{}/system_outputs/"
METADATA_PATH = DATA_FOLDER+"metrics_inputs/txt/{}/metadata/{}.tsv"
METADATA_PATH_challenge_bioMQM = DATA_FOLDER+"metrics_inputs/txt/{}/metadata/{}.{}.docID.csv" 

LANGUAGE_PAIRS = ['cs-uk',
 'en-cs',
 'en-de',
 'en-es',
 'en-hi',
 'en-is',
 'en-ja',
 'en-ru',
 'en-uk',
 'en-zh',
 'ja-zh']
 
METADATA_LANGUAGES = LANGUAGE_PAIRS

CHALLENGE_SETS = [
 'challenge_AfriMTE',
 #'challenge_MSLC24-A',
 #'challenge_MSLC24-B',
 'challenge_bioMQM',]
 #'challenge_dfki']
 
 
CHALLENGE_SETS_LPS = {'challenge_bioMQM': ['de-en',
  'en-de',
  'en-es',
  'en-fr',
  'en-ru',
  'en-zh',
  'es-en',
  'fr-en',],
  #'pt-en',
  #'ru-en',
  #'zh-en'],}
'challenge_AfriMTE': ['ary-fr',
  'en-arz',
  'en-fr',
  'en-hau',
  'en-ibo',
  'en-kik',
  'en-luo',
  'en-som',
  'en-swh',
  'en-twi',
  'en-xho',
  'en-yor',
  'yor-en'],
  'challenge_IndicMTE': ['en-ta',
  'en-gu',
  'en-hi',
  'en-mr',
  'en-ml',],
 #'challenge_MSLC24-A': ['en-de', 'en-es', 'ja-zh'],
 #'challenge_MSLC24-B': ['en-de', 'en-es', 'ja-zh'],
 }
 #'challenge_dfki': ['en-de', 'en-ru']}


def load_llm_gemma_pipeline(model_name="google/gemma-3-27b-it"):
    model = Gemma3ForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return (model, tokenizer)

def load_comet(model="Unbabel/wmt20-comet-da",
    model_storage_path=None):
    model_path = model if model.endswith(".ckpt") and os.path.exists(model) else download_model(model, saving_directory=model_storage_path)
    print("Loading checkpoint from ", model_path)
    model = load_from_checkpoint(model_path)
    model.eval()
    return model

def score_comet(samples,
    model,
    batch_size=16,
    gpus=1,
    num_workers=None,
    disable_length_batching=False,
    quiet=False,):

    output = model.predict(
            samples=samples,
            batch_size=batch_size,
            gpus=gpus,
            progress_bar=not quiet,
            accelerator="cpu" if gpus == 0 else "auto",
            num_workers=num_workers,
            length_batching=not disable_length_batching,
        )
    return output.scores

def llm_score_segment(source: str, translation: str, reference: str, model_tuple, max_new_tokens=100):

    prompt = f"""You are a professional translator. You should assess the machine translation adequacy on a continuous scale [0-100] based on critical points described below:

    [0]: Nonsense/No meaning preserved: Nearly all information is lost between the translation and source.
    [34]: Some meaning preserved: The translation preserves some of the meaning of the source but misses significant parts.
    [67]: Most meaning preserved: The translation retains most of the meaning of the source.
    [100]: Perfect meaning: The meaning of the translation is completely consistent with the source.

    Note that your score should lie in between two critical points, inclusive of the points themselves.

    Presented below are the source sentence, its machine translation, and the corresponding reference translation:
    Source sentence: {source}
    Machine translation: {translation}
    Reference translation: {reference}

    Please assess the above machine translation based on the source sentence and the reference translation. You should only output the final score."""

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]
    if model_tuple:
        model, tokenizer= model_tuple
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        try:
            generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generation = generation[0][input_len:]
            score_str = tokenizer.decode(generation, skip_special_tokens=True)
            try:
                score = float(score_str.split()[0])
            except:
                print(score_str)
                score = np.nan
        except Exception as e:
            print(f"Generation failed: {e}")
            score = np.nan

    else:    
        try:
            client = openai.OpenAI()  # Automatically uses OPENAI_API_KEY from environment
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            score_str = response.choices[0].message.content.strip()
            try:
                score = float(score_str.split()[0])
            except Exception:
                print("Unexpected output:", score_str)
                score = np.nan
        except Exception as e:
            print(f"ChatGPT call failed: {e}")
            score = np.nan
    return score

def segment_level_scoring(samples: Dict[str, List[str]], metric: str, model=None):
    """ Function that takes source, translations and references along with a metric and returns
    segment level scores.
    
    :param samples: Dictionary with 'src', 'mt', 'ref' keys containing source sentences, translations and 
        references respectively.
    :param metric: String with the metric name. 
        If 'BLEU' runs sentence_bleu from sacrebleu. 
        If chrF runs chrF from sacrebleu    
    """
    if metric == "chrF":
        scores = run_sentence_chrf(samples["mt"], samples["ref"])
        
    elif metric == "BLEU":
        scores = run_sentence_bleu(samples["mt"], samples["ref"])
        
    elif metric == "random":
        scores = np.random.random(size = len(samples["ref"]))
    
    elif metric.startswith("comet"):
        data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(samples["src"], samples["mt"], samples['ref'])]
        scores=score_comet(data, model)
    
    elif 'llm' in metric:
        scores = [
            llm_score_segment(s, m, r, model)
            for s, m, r in zip(samples["src"], samples["mt"], samples["ref"])
        ]
    
    else:
        raise Exception(f"{metric} segment_scoring is not implemented!!")

    return scores


# System-level scoring function
def system_level_scoring(samples: Dict[str, List[str]], metric: str, scores=List[float]):
    """ Function that takes source, translations and references along with a metric and returns
    system level scores.
    
    :param samples: Dictionary with 'src', 'mt', 'ref' keys containing source sentences, translations and 
        references respectively.
    :param metric: String with the metric name. 
        If 'BLEU' runs sentence_bleu from sacrebleu. 
        If chrF runs chrF from sacrebleu
    :param scores: List with segment level scores coming from the segment_level_scoring function.  
        Change this function if your metric DOES NOT use a simple average across segment level scores   
    """
    if metric == "chrF":
        system_score = corpus_chrf(samples["mt"], [samples["ref"]]).score

    elif metric == "BLEU":
        system_score = corpus_bleu(samples["mt"], [samples["ref"],]).score
            
    else:
        system_score = sum(scores)/len(scores)

    return system_score


def read_data(testset_name: str, language_pair: str):
    src_lang, trg_lang = language_pair.split("-")
    testset_type = 'challengesets2024' if testset_name.startswith('challenge') else testset_name
    
    sources = [s.strip() for s in open(SRC_PATH.format(testset_type, testset_name, language_pair, src_lang)).readlines()]
    references = {}
    human_scores=None
    if os.path.isfile(REFA_PATH.format(testset_type, testset_name, language_pair, trg_lang)):
        references["refA"] = [s.strip() for s in open(REFA_PATH.format(testset_type, testset_name, language_pair, trg_lang)).readlines()]
        assert len(references["refA"]) == len(sources)

    if os.path.isfile(REFB_PATH.format(testset_type, testset_name, language_pair, trg_lang)):
        references["refB"] = [s.strip() for s in open(REFB_PATH.format(testset_type, testset_name, language_pair, trg_lang)).readlines()]
        assert len(references["refB"]) == len(sources)
        
    if os.path.isfile(CS_REF_PATH.format(testset_type, testset_name, language_pair, trg_lang)):
        references["ref1"] = [s.strip() for s in open(CS_REF_PATH.format(testset_type, testset_name, language_pair, trg_lang)).readlines()]
        assert len(references["ref1"]) == len(sources)
        
    lp_systems = [
        (SYSTEM_FOLDER.format(testset_type) +s,  ".".join(s.split(".")[3:-1]))
        for s in os.listdir(SYSTEM_FOLDER.format(testset_type) ) if language_pair in s and testset_name in s
    ]    
    
    
    
    system_outputs = {}
    human_scores={}
    for system_path, system_name in lp_systems:
        if "ref" in system_name:
            continue
        if os.path.isfile(CS_HUMAN_PATH.format(testset_type, testset_name, language_pair, system_name)):
            human_scores[system_name] = [s.strip() for s in open(CS_HUMAN_PATH.format(testset_type, testset_name, language_pair, system_name)).readlines()]
            assert len(human_scores[system_name]) == len(sources)
        system_outputs[system_name] = [s.strip() for s in open(system_path).readlines()]
        assert len(system_outputs[system_name]) == len(sources)

    if  testset_type == 'generaltest2024':
        metadata = [s.strip().split() for s in open(METADATA_PATH.format(testset_type, language_pair)).readlines()]
        assert len(metadata) == len(sources)
    elif testset_name == 'challenge_bioMQM':
        metadata = [('all', s.strip()) for s in open(METADATA_PATH_challenge_bioMQM.format(testset_type, testset_name, language_pair)).readlines()]
        assert len(metadata) == len(sources)
    else:
        metadata = None
        
    return sources, references, system_outputs, metadata, human_scores
 
 
def run_sentence_bleu(candidates: list, references: list) -> list:
    """ Runs sentence BLEU from Sacrebleu. """
    assert len(candidates) == len(references)
    bleu_scores = []
    for i in tqdm(range(len(candidates)), desc="Running BLEU..."):
        bleu_scores.append(sentence_bleu(candidates[i], [references[i]]).score)
    return bleu_scores


def run_sentence_chrf(candidates: list, references: list) -> list:
    """ Runs sentence chrF from Sacrebleu. """
    assert len(candidates) == len(references)
    chrf_scores = []
    for i in tqdm(range(len(candidates)), desc="Running chrF..."):
        chrf_scores.append(
            sentence_chrf(hypothesis=candidates[i], references=[references[i]]).score
        )
    return chrf_scores


def segment_scores(source, references, system_outputs, metadata, human_scores, language_pair, metric_name, testset="generaltest2024", model=None):
    segment_scores = []
    system_scores = []
    all_domains = set()
    for ref in references:
        for hyp in system_outputs:
            print (f"Scoring {testset}-{language_pair} system {hyp} with {ref}:")
            samples = {"src": source, "mt": system_outputs[hyp], "ref": references[ref]}
            scores = segment_level_scoring(samples, metric_name, model)
            assert len(scores) == len(references[ref])
            assert len(references[ref]) == len(system_outputs[hyp])
            assert len(system_outputs[hyp]) == len(source)
            
            # Save Segment Scores
            for i in range(len(source)):
                if metadata is not None:
                    domain = metadata[i][0]
                    all_domains.add(domain)
                    document = metadata[i][1]
                else:
                    domain = "all"
                    document = "-"
                
                segment_scores.append({
                    "METRIC": metric_name,
                    "LANG-PAIR": language_pair,
                    "TESTSET": testset,
                    "DOMAIN": domain,
                    "DOCUMENT": document,
                    "REFERENCE": ref,
                    "SYSTEM_ID": hyp,
                    "SEGMENT_ID": i+1,
                    "SEGMENT_SCORE": scores[i],
                    "HUMAN_SCORE": human_scores[hyp][i] if hyp in human_scores else None
                })
            
            # Compute and save System scores for all domains.
            system_score = system_level_scoring(samples, metric_name, scores)            
            system_scores.append({
                "METRIC": metric_name,
                "LANG-PAIR": language_pair,
                "TESTSET": testset,
                "DOMAIN": "all",
                "REFERENCE": ref,
                "SYSTEM_ID": hyp,
                "SYSTEM_LEVEL_SCORE": system_score
            })

            # Compute and save System scores for each domain.
            if metadata is not None:
                for domain in all_domains:
                    domain_idx = [i for i in range(len(metadata)) if metadata[i][0] == domain]
                    domain_src = [source[idx] for idx in domain_idx]
                    domain_ref = [references[ref][idx] for idx in domain_idx]
                    domain_hyp  = [system_outputs[hyp][idx] for idx in domain_idx]
                    domain_scores  = [scores[idx] for idx in domain_idx]
                    domain_samples = {"src": domain_src, "mt": domain_hyp, "ref": domain_ref}
                    system_score = system_level_scoring(domain_samples, metric_name, domain_scores)
                    system_scores.append({
                        "METRIC": metric_name,
                        "LANG-PAIR": language_pair,
                        "TESTSET": testset,
                        "DOMAIN": domain,
                        "REFERENCE": ref,
                        "SYSTEM_ID": hyp,
                        "SYSTEM_LEVEL_SCORE": system_score
                    })
                
        for alt_ref in references.keys():
            if ref != alt_ref:
                print (f"Scoring {testset}-{language_pair} system {alt_ref} with {ref}:")
                samples = {"src": source, "mt": references[alt_ref], "ref": references[ref]}
                # Compute and Save Segment Scores
                scores = segment_level_scoring(samples, metric_name, model)
                for i in range(len(source)):
                    if metadata is not None:
                        domain = metadata[i][0]
                        document = metadata[i][1]
                    else:
                        domain = "all"
                        document = "-"
                    
                    segment_scores.append({
                        "METRIC": metric_name,
                        "LANG-PAIR": language_pair,
                        "TESTSET": testset,
                        "DOMAIN": domain,
                        "DOCUMENT": document,
                        "REFERENCE": ref,
                        "SYSTEM_ID": alt_ref,
                        "SEGMENT_ID": i+1,
                        "SEGMENT_SCORE": scores[i]
                    })

                # Compute and save System scores for all domains.
                system_score = system_level_scoring(samples, metric_name, scores)           
                system_scores.append({
                    "METRIC": metric_name,
                    "LANG-PAIR": language_pair,
                    "TESTSET": testset,
                    "DOMAIN": "all",
                    "REFERENCE": ref,
                    "SYSTEM_ID": alt_ref,
                    "SYSTEM_LEVEL_SCORE": system_score
                })
                
                # Compute and save System scores for each domain.
                if metadata is not None:
                    for domain in all_domains:
                        domain_idx = [i for i in range(len(metadata)) if metadata[i][0] == domain]
                        domain_src = [source[idx] for idx in domain_idx]
                        domain_ref = [references[ref][idx] for idx in domain_idx]
                        domain_hyp  = [references[alt_ref][idx] for idx in domain_idx]
                        domain_scores  = [scores[idx] for idx in domain_idx]
                        domain_samples = {"src": domain_src, "mt": domain_hyp, "ref": domain_ref}
                        system_score = system_level_scoring(domain_samples, metric_name, domain_scores)
                        system_scores.append({
                            "METRIC": metric_name,
                            "LANG-PAIR": language_pair,
                            "TESTSET": testset,
                            "DOMAIN": domain,
                            "REFERENCE": ref,
                            "SYSTEM_ID": alt_ref,
                            "SYSTEM_LEVEL_SCORE": system_score
                        })

    return pd.DataFrame(segment_scores), pd.DataFrame(system_scores)


def score_indicmte(language_pair, metric_name, model):
    file_path=DATA_FOLDER+f'metrics_inputs/IndicMTE/{language_pair}.tsv'
    indicmte_df=pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
    systems=indicmte_df.model.unique()
    segment_scores = []
    system_scores = []
    for sys in systems:
        system_df=indicmte_df[indicmte_df.model==sys]
        source=system_df.Source.tolist()
        translation=system_df.Translation.tolist()
        ref=system_df.Reference.tolist()
        samples = {"src": source, "mt": translation, "ref": ref}
        scores = segment_level_scoring(samples, metric_name, model)
        for seg_idx, (i, row) in enumerate(system_df.iterrows()):
            segment_scores.append({
                            "METRIC": metric_name,
                            "LANG-PAIR": language_pair,
                            "TESTSET": 'challenge_IndicMTE',
                            "DOMAIN": "all",
                            "DOCUMENT": "-",
                            "REFERENCE": "ref1",
                            "SYSTEM_ID": sys,
                            "SEGMENT_ID": seg_idx+1,
                            "SEGMENT_SCORE": scores[seg_idx],
                            "HUMAN_SCORE":row['Human_scores']
                        })

        system_score = system_level_scoring(samples, metric_name, scores)            
        system_scores.append({
            "METRIC": metric_name,
            "LANG-PAIR": language_pair,
            "TESTSET": 'challenge_IndicMTE',
            "DOMAIN": "all",
            "REFERENCE": "ref1",
            "SYSTEM_ID": sys,
            "SYSTEM_LEVEL_SCORE": system_score
        })

    return pd.DataFrame(segment_scores), pd.DataFrame(system_scores)

def process_language_pair(lp, metric, model):
    seg_path = OUTPUT_FOLDER + f"partial/{metric}.generaltest2024.{lp}.seg.tsv"
    sys_path = OUTPUT_FOLDER + f"partial/{metric}.generaltest2024.{lp}.sys.tsv"

    if os.path.exists(seg_path) and os.path.exists(sys_path):
        segments = pd.read_csv(seg_path, sep="\t", header=None)
        systems = pd.read_csv(sys_path, sep="\t", header=None)
    else:
        source, references, system_outputs, metadata, human_scores = read_data('generaltest2024', lp)
        segments, systems = segment_scores(source, references, system_outputs, metadata, human_scores, lp, metric, model=model)
        segments.to_csv(seg_path, index=False, header=False, sep="\t")
        systems.to_csv(sys_path, index=False, header=False, sep="\t")
    
    return segments, systems

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scores Newstest2020 segments."
    )
    parser.add_argument(
        "--baseline",
        help="Metric to run.",
        type=str,
    )

    parser.add_argument(
        "--checkpoint",
        help="COMET checkpoint.",
        type=str,
    )

    parser.add_argument(
        "--set",
        default="all",
        help="Sets to evaluate.",
        type=str,
    )

    args = parser.parse_args()
    segment_data, system_data = [], []
    metric=args.baseline
    model=None 

    # Directory to store individual results
    os.makedirs(OUTPUT_FOLDER+"partial/", exist_ok=True)

    existing_files = set([f for f in os.listdir(OUTPUT_FOLDER+"partial/") if f.startswith(metric + ".")])

    if metric.startswith("comet"):
        model=load_comet(args.checkpoint)
    #elif 'llm' in metric:
        #model =load_llm_gemma_pipeline()
    
    if args.set=='all' or args.set=='challenge':

        for challengeset_name, lps in tqdm(CHALLENGE_SETS_LPS.items(), desc="Processing Challenge Sets"):
            for lp in lps:
                seg_file = f"{metric}.{challengeset_name}.{lp}.seg.tsv"
                sys_file = f"{metric}.{challengeset_name}.{lp}.sys.tsv"
                seg_path = OUTPUT_FOLDER + "partial/" + seg_file
                sys_path = OUTPUT_FOLDER + "partial/" + sys_file
                if seg_file in existing_files and sys_file in existing_files:
                    segments = pd.read_csv(seg_path, sep="\t", header=None)
                    systems = pd.read_csv(sys_path, sep="\t", header=None)
                else:
                    if challengeset_name=='challenge_IndicMTE':
                        segments, systems=score_indicmte(lp, metric, model=model)
                    else:
                        print("starting scoring")
                        source, references, system_outputs, metadata, human_scores = read_data(challengeset_name, lp)
                        print("data loaded")
                        segments, systems = segment_scores(source, references, system_outputs, metadata, human_scores, lp, metric, testset=challengeset_name, model=model)
                    segments.to_csv(seg_path, index=False, header=False, sep="\t")
                    systems.to_csv(sys_path, index=False, header=False, sep="\t")
                segment_data.append(segments)
                system_data.append(systems)
    
    if args.set=='all' or args.set=='wmt24':

        for lp in tqdm(LANGUAGE_PAIRS, desc="Processing Language Pairs"):
            seg_path = OUTPUT_FOLDER+f"partial/{metric}.generaltest2024.{lp}.seg.tsv"
            sys_path = OUTPUT_FOLDER+f"partial/{metric}.generaltest2024.{lp}.sys.tsv"

            if os.path.exists(seg_path) and os.path.exists(sys_path):
                segments = pd.read_csv(seg_path, sep="\t", header=None)
                systems = pd.read_csv(sys_path, sep="\t", header=None)
            else:
                source, references, system_outputs, metadata, human_scores = read_data('generaltest2024', lp)
                segments, systems = segment_scores(source, references, system_outputs, metadata, human_scores, lp, metric, model=model)
                segments.to_csv(seg_path, index=False, header=False, sep="\t")
                systems.to_csv(sys_path, index=False, header=False, sep="\t")
            segment_data.append(segments)
            system_data.append(systems)
    
    segment_data = pd.concat(segment_data, ignore_index=True)
    segment_data.to_csv(OUTPUT_FOLDER+"{}.seg.score".format(metric), index=False, header=False, sep="\t")
    
    system_data = pd.concat(system_data, ignore_index=True)
    system_data.to_csv(OUTPUT_FOLDER+"{}.sys.score".format(metric), index=False, header=False, sep="\t")

