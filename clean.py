import pandas as pd
import argparse

def main(args):
    texts = []
    with open(args.dir+args.corpus) as f:
        for line in f:
            texts.append(line.strip())

    df = pd.DataFrame(texts, columns=['text'])
    df.to_csv(args.dir+args.dataset+"_train.csv",index=False,header=None)
    aspects = []
    with open(args.dir+args.aspects) as f:
        for line in f:
            lst = line.split()
            idx = lst[0]
            aspect = lst[1].lower()
            aspects.append(aspect)

    data = []
    with open(args.dir+args.test) as f:
        for line in f:
            lst = line.split('\t')
            #print(lst)
            aspect = lst[1]
            text = lst[3].strip()
            #print(text)
            data.append([text,aspects[int(aspect)-1]])

    df = pd.DataFrame(data, columns = ['text','label'])
    df.to_csv(args.dir+args.dataset+"_test.csv",index=False,header=None)

    data = []
    with open(args.dir+args.test_kplus) as f:
        for line in f:
            lst = line.split('\t')
            #print(lst)
            aspect = lst[2]
            text = lst[3].strip()
            data.append([text,aspect])

    df = pd.DataFrame(data, columns = ['text','label'])
    df.to_csv(args.dir+args.dataset+"_test_kplus.csv",index=False,header=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--dir', type=str, default='yelp/')
    parser.add_argument('--aspects', type=str, default='yelp_aspects.txt')
    parser.add_argument('--corpus', type=str, default='yelp_corpus.linked.txt')
    parser.add_argument('--train', type=str, default='yelp_train.txt')
    parser.add_argument('--test', type=str, default='yelp_test.txt')
    parser.add_argument('--test_kplus', type=str, default='yelp_test_kplus.txt')
    args = parser.parse_args()
    main(args)
    