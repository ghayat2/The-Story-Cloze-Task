# nlu-project2
## Setup
- Go to **data/embeddings/skip_thoughts/** and run:
    - wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
    - wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
    - wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
    - wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
    - wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
    - wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
    - wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
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
    