# ARYA

This is the source code for our arXiv and SDM 2021 submission:

1.Peiran Li, Fang Guo, and Jingbo Shang. 2020. User-Guided Aspect Classification for Domain-Specific Texts. arXiv preprint arXiv:2004.14555 (2020).

2.Peiran Li*, Fang Guo*, and Jingbo Shang. 2021. "Misc"-Aware Weakly Supervised Aspect Classification. In SIAM International Conference on Data Mining (SDM21).

The preprint version can be accessed at: [https://arxiv.org/abs/2004.14555](https://arxiv.org/abs/2004.14555)

- [Model](#model)
- [Training](#training)
	- [Required Inputs](#required-inputs)
	- [Dependencies](#dependencies)
	- [Command](#command)
- [Citation](#citation)

## Model

![ARYA-Framework](docs/ARYA_SDM21_Framework.png)

## Training

### Required inputs

- **Raw Corpus TXT File for Training Embedding**
  - Example: ```restaurant/restaurant_corpus.linked.txt```
    - One review per line.
    - Must be named as ```dataset_corpus.linked.txt```.
- **Training CSV File**
  - Example: ```restaurant/restaurant_train.csv```
    - Only one column.
    - One review per line.
    - Must be named as ```dataset_train.csv```.
- **K Aspect Test CSV File**
  - Example: ```restaurant/restaurant_test.csv```
    - Two columns (segment, aspect) per line.
    - Aspects **NOT** including `miscellaneous` aspect.
    - Must be named as ```dataset_test.csv```.
- **K+1 Aspect Test CSV File**
  - Example: ```restaurant/restaurant_test_kplus.csv```
    - Two columns (segment, aspect) per line.
    - Aspects including `miscellaneous` aspect.
    - Must be named as ```dataset_test_kplus.csv```
- **Word Embedding File**
  - Example: ```restaurant/restaurant.200d.txt```
    - The word embedding files have been provided but you can also train your own.
    - Must be named as ```dataset.200d.csv```
- **Aspects TXT File**
  - Example: ```restaurant/restaurant_aspects.txt```
    - One aspect per line.
    - Must be named as ```dataset_aspects.txt```
- **Seed Words Dictionary**
  - Example: ```restaurant/restaurant_seeds.txt```
    - Two columns (seedword, aspect) per line.
    - Separated by tab.
    - Must be named as ```dataset_seeds.txt```

### Dependencies

This project is based on ```python==3.6.9```. The dependencies are as follow:
```
pytorch==1.3.1
torchtext==0.4.0
scikit-learn==0.22
spacy==2.2.3
scipy=1.4.1
gensim==3.8.1
numpy==1.17.4
```

### Command

To train the word2vec embedding on the specific corpus, please run
`python embedding.py --dataset dataset_name`.

To train ARYA, please run
`python train.py --dataset dataset_name --quantile quantile_to_decide_hnorm_threshold --score_threshold score_threshold_for_seed_selection --seedword_limit max_num_seedwords --no_filtering --no_tuning`

The `--quantile` is the for deciding hnorm threshold for computing Pmisc.

The `--score_threshold` is the threshold for seed word selection in the seed expansion process.

The `--no_filtering` and `--no_tuning` are two options for ablation studies.

## Citation

Please cite our paper if you are using our framework. Thank you!

```
@article{li2020user,
  title={User-Guided Aspect Classification for Domain-Specific Texts},
  author={Li, Peiran and Guo, Fang and Shang, Jingbo},
  journal={arXiv preprint arXiv:2004.14555},
  year={2020}
}
```
