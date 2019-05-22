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
## Project Structure

- train.py -> Main train class

- data/cloze_train.csv -> training dataset
- data/cloze_eval.csv -> evaluation dataset

- models/MODEL -> model

## Links
- Skip-thought: https://github.com/ryankiros/skip-thoughts