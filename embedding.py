import gzip
import gensim 
import logging
import argparse

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, input_file):
        self.input_file = input_file

    def __iter__(self):
        logging.info("reading file {0}...this may take a while".format(self.input_file))
        with open(self.input_file, 'rb') as f:
            for i, line in enumerate(f):

                if (i % 10000 == 0):
                    logging.info("read {0} reviews".format(i))
                # do some pre-processing and return list of words for each review
                # text
                yield gensim.utils.simple_preprocess(line)

def main(args):
    documents = MyCorpus(args.dir+args.input)
    model = gensim.models.Word2Vec(
            documents,
            size=200,
            min_count=2,
            iter=20)
    model.wv.save_word2vec_format(args.dir+args.output, binary=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='yelp/')
    parser.add_argument('--input', type=str, default='yelp_corpus.linked.txt')
    parser.add_argument('--output', type=str, default='yelp.200d.txt')
    args = parser.parse_args()
    main(args)
    