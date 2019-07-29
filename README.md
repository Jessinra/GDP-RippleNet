# GDP-RippleNet

***Last edit : 29 July 2019***

Recommender system using [RippleNet](http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf), trained using custom [MovieLens-20M](https://grouplens.org/datasets/movielens/20m/) dataset.

This repository is the implementation of RippleNet ([arXiv](https://arxiv.org/abs/1803.03467)):
> RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems  
Hongwei Wang, Fuzheng Zhang, Jialin Wang, Miao Zhao, Wenjie Li, Xing Xie, Minyi Guo  
The 27th ACM International Conference on Information and Knowledge Management (CIKM 2018)

![](https://github.com/hwwang55/RippleNet/blob/master/framework.jpg)

RippleNet is a deep end-to-end model that naturally incorporates the knowledge graph into recommender systems.
Ripple Network overcomes the limitations of existing embedding-based and path-based KG-aware recommendation methods by introducing preference propagation, which automatically propagates users' potential preferences and explores their hierarchical interests in the KG.

A PyTorch re-implementation of RippleNet by Qibin Chen et al. is [here](https://github.com/qibinc/RippleNet-PyTorch).

# Domain of problems
*Given a user and an item, predict how likely the user will interact with the item*

# Contents
- `/data` : contains dataset to use in training
    - `/book`: (not used)
    - **`/movie`** : custom ml-20m dataset where only movies shows up in Ripple-Net's knowledge graph used.    
- **`/log`** : contains training result stored in single folder named after training timestamp.
- **`/test`** : contains jupyter notebook used in testing the trained models
- `/src` : implementations of RippleNet.

### Note
    *italic* means this folder is ommited from git, but necessary if you need to run experiments
    **bold** means this folder has it's own README, check it for detailed information :)

# Preparing 
## Installing dependencies 

    pip3 install -r requirements.txt

## Where to download the dataset
You can download the intersect-20m dataset [here](https://github.com/Jessinra/GDP-KG-Dataset). Copy all items inside and put into `/movie` folder **except the Preprocess.ipynb**.

*Note : Dataset is put on separate repository because it's shared among models.*

## How to prepare data
1. To create missing `kg_final.txt` & `ratings_final.txt` files:
- either use this [jupyter notebook](./data/movie/Preprocess.ipynb) (`Preprocess.ipynb` inside the `./data/movie`).
- or simply run :
     ```
     python3 src/preprocess.py
     ```
2. To create the rest of the files :
    
    Simply provide `kg_final.txt` v and `ratings_final.txt` then run 
    ~~~
    python3 src/main.py
    ~~~
    the script will preprocess it (and save the result for cache) before training begin.

# How to run
1. Prepare the dataset and the preprocessed version (check section below this)
2. Run the training script
    ~~~
    python3 src/main.py
    ~~~

## **! Caching warning !**
To start using new dataset, or if dataset changed, those file need to be deleted, and re preprocess it, otherwise it will use old dataset:
- `data/movie/ratings_final.npy`
- `data/movie/kg_final.npy`
- `data/movie/preprocessed_data_info_32` or `_64`

# Training
## How to change hyper parameter
There are several ways to do this :
1. Open `src/main.py` and change the args parser default value
2. run `src/main.py` with arguments required.

# Testing / Evaluation
## How to check training result
1. Find the training result folder inside `/log` (find the latest), copy the folder name.
2. Create copy of latest jupyter notebook inside `/test` folder.
3. Rename folder to match a folder in `/log` (for traceability purpose).
4. Replace `TESTING_CODE` at the top of the notebook.
5. Run the notebook

# Final result
| Evaluation size     | Prec@10 | Distinct@10 |  Unique items |
|---------------------|---------|-------------|---------------|
| Eval on  500 user   | 0.26719 |  0.04640    |   232         |
| Eval on 1000 user   | 0.26939 |  0.02860    |   286         |
| Eval on 3000 user   | 0.26379 |  0.01317    |   395         |
| Eval on 5000 user   | 0.26573 |  0.00914    |   457         |


# Other findings
- Looking from the result of Ripplenet-1M, the usage of KG might turn out to be quite promising, especially to improve the diversity of suggestion.
  
- The downside of using RippleNet is that the train and test both takes a lot of time (almost 1 hr/epoch for training, and even longer for testing). This probably caused by the nature of RippleNet which supposed to predict 

    >*Given a user and an item, how likely the user will interact with the item*
    
    and not

    >*Given a user and a list of items, find the top k items that user might like*
  
- Summary compared to non-KG RecSys: **Big improvement in terms of Prec@k and diversity**

# Pros
- Able to incorporate Knowledge Graph as another source of information with relatively simple approach (embedding)
- Higher Prec@K compared to KPRN and its derivative models.
- Require much less memory and disk space compared to KORN.
- Doesn't require too much hand crafted features.
- During training, the model converge really fast (< 10 epochs)

# Cons
- Require relatively slow pre-preprocessing
- The train and test both takesMore diverse suggestion. a lot of time (almost 1 hr/epoch for training, and even longer for testing). 
- The model remember the user, the model need to be re-trained for every new user and  item addition.
- Loss function and metric used in training is not Prec@K, instead it uses AUC and accuracy.
- Output less diverse compared to KPRN

# Experiment notes
- Looking from the result of Ripplenet-1M, the usage of KG might turn out to be quite promising, especially to improve the diversity of suggestion.
- Use lower dimension of embedding (n = 16) result in much better performance and faster training result.

# Author
- Jessin Donnyson - jessinra@gmail.com

# Contributors
- Benedict Tobias Henokh Wahyudi - tobi8800@gmail.com
- Michael Julio - michael.julio@gdplabs.id
- Fallon Candra - fallon.candra@gdplabs.id