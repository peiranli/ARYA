import gensim
import argparse

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        print("reading file {0}...this may take a while".format(self.dataset+"/"+self.dataset+"_corpus.linked.txt"))
        with open(self.dataset+"/"+self.dataset+"_corpus.linked.txt", 'rb') as f:
            for i, line in enumerate(f):

                if (i % 10000 == 0):
                    print("read {0} reviews".format(i))
                # do some pre-processing and return list of words for each review
                # text
                yield gensim.utils.simple_preprocess(line)

def main(args):
    documents = MyCorpus(args.dataset)
    model = gensim.models.Word2Vec(
            documents,
            size=200,
            min_count=2,
            iter=20)
    model.wv.save_word2vec_format(args.dataset+"/"+args.dataset+".200d.txt", binary=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='restaurant')
    args = parser.parse_args()
    main(args)
    