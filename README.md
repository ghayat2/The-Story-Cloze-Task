# nlu-project2
## Setup
- Download the cloze datasets (eval and train) and place them in a folder
named "data". Also, create a folder "data/processed/".
- Run process_train.py
- Run `python3 process_eval.py eval_stories.csv tokenizer.pickle`
- Go to **data/embeddings/skip_thoughts/** and run:
    - wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
    - wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
    - wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
    - wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
    - wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
    - wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
    - wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
- When running on Leonhard, you'll need to install `punkt` manually by doing:
    - python
    - ``` import nltk```
    - ``` nltk.download('punkt')```
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