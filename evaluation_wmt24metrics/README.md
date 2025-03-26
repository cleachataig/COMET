## README

This contains the inputs for the metrics task at WMT 2024, both the main task ( from WMT general translation task) and the challenge sets subtask (created specifically to test MT metrics). Please score both. You can use the provided prepare_scores.py script to iterate over all files. 

The files can be found in text format only, where one line corresponds to one segment.  

The txt dir contains one subdir for generaltest2024 and challengesets2024, each of which contains:
- sources (dir with all the source files, filenames look like <testsetname>.<src>-<tgt>.ref.<refid>.<tgt>)
- references (dir with all the ref files, filenames look like <testsetname>.<src>-<tgt>.src.<src>)
- system_outputs
   |_ <src>-<tgt> (filenames look like  <testsetname>.<src>-<tgt>.hyp.<systemid>.<tgt>) 
   
In addition, the generaltest2024 contains a metadata subdir with information of domain and document id for each segment.


# Scoring Scripts:

Along with the inputs we are releasing a script to help everyone score the data in the correct format.

To adapt that to your metric you just need to change the segment_level_scoring function and the system_level_scoring according to your metric. These methods receive a dictionary called samples containing the following format:

samples = {
    "src": ["hello world!", "welcome to Metrics task", ...],
    "mt": ["ola mundo!", "bem vindos à tarefa de metricas", ...],
    "ref": ["hello world!", "bem vindos à Metrics task", ...]
}
and you can run the script with random, BLEU, chrF with the following command (first verify if you have sacrebleu installed!):

```bash
python prepare_scores.py --baseline chrF
```

The output files will be stored in `scores/`.

For QE-as-a-metric we provide a similar script to run random, COMET-QE baseline (first verify if you have unbabel-comet installed!):

```bash
python prepare_scores.py --baseline COMET-QE
```
