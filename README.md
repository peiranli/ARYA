This is the source code for our arXiv User-Guided Aspect Classification for
Domain-Specific Texts. <br>
<br>
Train the word2vec embedding on the specific corpus by running `python embedding.py --dir dir --input corpus_file --output output_file`. <br>
<br>
As long as the csv files and word2vec embedding file have been prepared, run the training file by running `python train.py --dataset dataset_name --datadir dir --aspects aspect_txt --train train_csv --test test_csv --test_kplus test_kplus_csv --seedwords seedwords_file --embedding embedding_file --quantile quantile_to_decide_hnorm_threshold --score_threshold score_threshold_for_seed_selection --seedword_limit max_num_seedwords --no_filtering --no_tuning`. <br>
<br>
The `--no_filtering` and `--no_tuning` are two options.
