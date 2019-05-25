# nlu-project2
## Setup
- Download the cloze datasets (eval and train) and place them in a folder
named "data". Also, create a folder "data/processed/".
- Run process_train.py
- Run 
```bash
python3 process_eval.py eval_stories.csv data/tokenizer.pickle
```
- If you're using skip-thoughts, you need to download to you `data/processed/`
folder these two files:
```bash
wget --user=nethz_username --ask-password https://polybox.ethz.ch/remote.php/webdav/nlu-project2/eval_stories_skip_thoughts.tfrecords
```
```bash
wget --user=nethz_username --ask-password https://polybox.ethz.ch/remote.php/webdav/nlu-project2/train_stories_skip_thoughts.tfrecords
```
- If you're using word2vec with 100 dimensional word embeddings:
```bash
wget --user=nethz_username --ask-password https://polybox.ethz.ch/index.php/s/mFkjmC9EmPKDzg1/download
```

## Project Structure

- train.py -> Main train class

- data/cloze_train.csv -> training dataset
- data/cloze_eval.csv -> evaluation dataset

- models/MODEL -> model

## Links
- Skip-thought: https://github.com/ryankiros/skip-thoughts


## Things we want to test (21-05-2019)

Discriminator
 - Mean vs. sum in word embedding --> Submit job now
    - Set pad vector to 0 ?
    
 - Skip-thought sentence embedding --> Arthur
 - Multiple endings: 6
    - 3/3 backwards/random --> Gabriel & Hidde
 - Dropout --> Ji
 - Introduce features
    
Generator
 - Implement near-generation
 - Implement VAE
 - Implement LM
    - LM project 1
    - Pretrained ?
    
Misc
 - Data cleaning (numbers, contractions)