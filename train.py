import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext
from torchtext import data
from torchtext import datasets
import collections
import math
import argparse
from util import *
from model.pseudolabel import *
from model.cnn import *
from scipy import stats
import numpy as np
from sklearn.metrics import confusion_matrix

def main(args):

    aspects = []
    with open(args.dir+args.aspects) as f:
        for line in f:
            lst = line.split()
            idx = lst[0]
            aspect = lst[1].lower()
            aspects.append(aspect)
    print(aspects)
    
    TEXT = data.Field(tokenize=tokenizer)
    train_data = data.TabularDataset(path=args.dir+args.train, format='csv',fields=[('text', TEXT)])
    LABEL = data.LabelField()
    test_data = data.TabularDataset(path=args.dir+args.test, format='csv',fields=[('text', TEXT), ('label', LABEL)])
    embedding = torchtext.vocab.Vectors(args.dir+args.embedding)

    MAX_VOCAB_SIZE = 10000
    TEXT.build_vocab(train_data, 
                     max_size = MAX_VOCAB_SIZE, 
                     vectors = embedding, 
                     unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(test_data)
    print(LABEL.vocab.stoi)
    print(LABEL.vocab.itos)
    BATCH_SIZE = 256

    if torch.cuda.is_available():
        torch.cuda.set_device(6)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    train_iterator = data.BucketIterator(
        train_data, 
        batch_size = BATCH_SIZE, 
        device = device,
        sort=False)
    test_iterator = data.BucketIterator(
        test_data, 
        batch_size = len(test_data), 
        device = device,
        sort=False)

    LABEL_KPLUS = data.LabelField()
    test_kplus_data = data.TabularDataset(path=args.dir+args.test_kplus, format='csv',fields=[('text', TEXT), ('label', LABEL_KPLUS)])
    test_kplus_iterator = data.BucketIterator(
        test_kplus_data, 
        batch_size = len(test_kplus_data), 
        device = device,
        sort=False)
    LABEL_KPLUS.build_vocab(test_kplus_data)
    print(LABEL_KPLUS.vocab.stoi)

    from sklearn import metrics

    def train_metric(preds, label):
        max_preds = preds.argmax(dim=1)
        max_label = label.argmax(dim=1)
        acc = metrics.accuracy_score(max_label.cpu().numpy(), max_preds.cpu().numpy())
        return acc

    def train(model, pseudolabel, iterator, optimizer):
        criterion = nn.KLDivLoss()
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        pseudolabel.eval()

        for batch in iterator:
            optimizer.zero_grad()
            probs, _ = model(batch.text)  #[batch size, output dim] 
            p, q = pseudolabel(batch.text)
            loss = criterion(torch.log(probs), p.detach())
            acc = train_metric(probs, p.detach())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(model, eval_data, LABEL):
        preds = []
        labels = []
        for e in eval_data.examples:
            pred = predict(model, e.text)
            preds.append(pred)
            labels.append(LABEL.vocab.stoi[e.label])

        f1 = metrics.f1_score(labels, preds, average='weighted')
        acc = metrics.accuracy_score(labels, preds)
        return acc, f1

    def predict_class(model, sentence, min_len = 5):
        model.eval()
        tokenized = [tok for tok in tokenizer(sentence)]
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        preds, _ = model(tensor)
        max_preds = preds.argmax(dim = 1)
        return max_preds.item()

    def predict(model, sentence, min_len = 5):
        model.eval()
        if len(sentence) < min_len:
            sentence += ['<pad>'] * (min_len - len(sentence))
        indexed = [TEXT.vocab.stoi[t] for t in sentence]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        preds, _ = model(tensor)
        max_preds = preds.argmax(dim = 1)
        return max_preds.item()

    def predict_pseudolabel(model, sentence, min_len = 5):
        model.eval()
        if len(sentence) < min_len:
            sentence += ['<pad>'] * (min_len - len(sentence))
        indexed = [TEXT.vocab.stoi[t] for t in sentence]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        _, preds = model(tensor)
        max_preds = preds.argmax(dim = 1)
        return max_preds.item()

    def get_qs(model, sentence, min_len = 5):
        model.eval()
        if len(sentence) < min_len:
            sentence += ['<pad>'] * (min_len - len(sentence))
        indexed = [TEXT.vocab.stoi[t] for t in sentence]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        p, q = model(tensor)
        max_q = torch.max(q, 1)[1]
        return q, max_q

    def get_p(model, sentence, min_len = 5):
        model.eval()
        if len(sentence) < min_len:
            sentence += ['<pad>'] * (min_len - len(sentence))
        indexed = [TEXT.vocab.stoi[t] for t in sentence]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        preds, classes = model(tensor)
        return preds, classes

    import datetime
    time = int(datetime.datetime.now().timestamp())

    import logging
    logging.basicConfig(filename='outputs/'+str(time)+'train-yelp.log',level=logging.DEBUG)

    import collections
    seed_words_d = collections.defaultdict(set)
    with open(args.dir+args.seedwords) as f:
        for line in f:
            lst = line.split()
            w1 = lst[0].lower()
            w2 = lst[1].lower()
            seed_words_d[w2].add(w1)

    seed_words = sorted(seed_words_d.items(), key=lambda x:LABEL.vocab.stoi[x[0]])
    print(seed_words)

    def get_seed_embedding(seed_words):
        SEED_WORDS = []
        for w, lst in seed_words:
            temp = []
            for e in lst:
                temp.append(TEXT.vocab.vectors[TEXT.vocab.stoi[e]].unsqueeze(0))
            embeds = torch.cat(temp)
            embed = torch.mean(embeds,dim=0)
            SEED_WORDS.append(embed.unsqueeze(0))
        SEED_WORDS = torch.cat(SEED_WORDS)
        SEED_WORDS = SEED_WORDS.unsqueeze(1)
        SEED_WORDS = SEED_WORDS.unsqueeze(1)
        return SEED_WORDS
    SEED_WORDS = get_seed_embedding(seed_words)
    print(SEED_WORDS.shape)
    
    def init_kmodel(SEED_WORDS):
        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 200
        N_FILTERS = 100
        FILTER_SIZES = [2,3,4]
        KOUTPUT_DIM = len(LABEL.vocab)
        DROPOUT = 0.5
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        k_model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, KOUTPUT_DIM, DROPOUT, PAD_IDX)
        k_model = k_model.to(device)
        #k_model.load_state_dict(torch.load('k-model.pt'))

        k_pseudolabel = PseudoLabel(INPUT_DIM, EMBEDDING_DIM, KOUTPUT_DIM, KOUTPUT_DIM, PAD_IDX, SEED_WORDS)
        k_pseudolabel.eval()
        k_pseudolabel = k_pseudolabel.to(device)

        pretrained_embeddings = TEXT.vocab.vectors

        k_model.embedding.weight.data.copy_(pretrained_embeddings)
        k_pseudolabel.embedding.weight.data.copy_(pretrained_embeddings)

        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        k_model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        k_model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        k_pseudolabel.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        k_pseudolabel.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
        return k_model, k_pseudolabel
    
    k_model, k_pseudolabel = init_kmodel(SEED_WORDS)
    k_model_optimizer = optim.Adam(filter(lambda p: p.requires_grad, k_model.parameters()))

    N_EPOCHS = 5
    for epoch in range(N_EPOCHS):
        print("epoch: ",epoch+1)
        train_loss, train_acc = train(k_model, k_pseudolabel, train_iterator, k_model_optimizer)
        print("training loss: ",train_loss)
        print("training accuracy: ",train_acc)
        valid_acc, valid_f1 = evaluate(k_model, test_data, LABEL)
        print("validation accuracy: ",valid_acc)
        print('validation F1:',valid_f1)
    torch.cuda.empty_cache()

    preds = []
    labels = []
    for e in test_data.examples:
        pred = predict(k_model, e.text)
        preds.append(pred)
        labels.append(LABEL.vocab.stoi[e.label])
        
    def log_info(labels, preds):
        print(metrics.accuracy_score(labels, preds))
        logging.debug(metrics.accuracy_score(labels, preds))
        print(metrics.precision_score(labels, preds, average='weighted'))
        logging.debug(metrics.precision_score(labels, preds, average='weighted'))
        print(metrics.recall_score(labels, preds, average='weighted'))
        logging.debug(metrics.recall_score(labels, preds, average='weighted'))
        print(metrics.f1_score(labels, preds, average='weighted'))
        logging.debug(metrics.f1_score(labels, preds, average='weighted'))
        m = confusion_matrix(labels, preds)
        print(m)
        logging.debug(m)

    log_info(labels, preds)

    logging.debug("k pseudolabel")
    preds = []
    labels = []
    for e in test_data.examples:
        pred = predict_pseudolabel(k_pseudolabel, e.text)
        preds.append(pred)
        labels.append(LABEL.vocab.stoi[e.label])
    log_info(labels, preds)

    lst1 = []
    lst2 = []
    for e in test_kplus_data.examples:
        qs,_ = get_qs(k_pseudolabel, e.text)
        preds,_ = get_p(k_model, e.text)
        vs = [v.item() for v in preds.squeeze(0) if v.item() != 0]
        h_norm = (-1/math.log(preds.shape[1]))*sum([v*math.log(v)for v in vs])
        if e.label == 'miscellaneous':
            lst1.append(int(h_norm*100))
        else:
            lst2.append(int(h_norm*100))

    a = np.array(lst2)
    threshold = np.quantile(a, args.quantile)/100
    print("threshold:", threshold)
    logging.debug("threshold:"+str(threshold))

    def init_kplusmodel(SEED_WORDS):
        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 200
        N_FILTERS = 100
        FILTER_SIZES = [2,3,4]
        OUTPUT_DIM = len(LABEL_KPLUS.vocab)
        DROPOUT = 0.5
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        kplus_model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
        kplus_model = kplus_model.to(device)

        kplus_pseudolabel = PseudoLabelPlus(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM-1, OUTPUT_DIM-1, PAD_IDX, SEED_WORDS, k_model, threshold, device)
        kplus_pseudolabel = kplus_pseudolabel.to(device)
        kplus_pseudolabel.eval()

        pretrained_embeddings = TEXT.vocab.vectors

        kplus_model.embedding.weight.data.copy_(pretrained_embeddings)
        kplus_pseudolabel.embedding.weight.data.copy_(pretrained_embeddings)

        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        kplus_model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        kplus_model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        kplus_pseudolabel.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        kplus_pseudolabel.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
        return kplus_model, kplus_pseudolabel

    kplus_model, kplus_pseudolabel = init_kplusmodel(SEED_WORDS)
    kplus_model_optimizer = optim.Adam(filter(lambda p: p.requires_grad, kplus_model.parameters()))

    N_EPOCHS = 5
    for epoch in range(N_EPOCHS):
        print("epoch: ",epoch+1)
        train_loss, train_acc = train(kplus_model, kplus_pseudolabel, train_iterator, kplus_model_optimizer)
        print("training loss: ",train_loss)
        print("training accuracy: ",train_acc)
        valid_acc, valid_f1 = evaluate(kplus_model, test_kplus_data, LABEL_KPLUS)
        print("validation accuracy: ",valid_acc)
        print('validation F1:',valid_f1)
    torch.cuda.empty_cache()

    preds = []
    labels = []
    for e in test_kplus_data.examples:
        pred = predict(kplus_model, e.text)
        preds.append(pred)
        labels.append(LABEL_KPLUS.vocab.stoi[e.label])
    log_info(labels, preds)

    logging.debug("kplus pseudolabel")
    preds = []
    labels = []
    for e in test_kplus_data.examples:
        pred = predict_pseudolabel(kplus_pseudolabel, e.text)
        preds.append(pred)
        labels.append(LABEL_KPLUS.vocab.stoi[e.label])
    log_info(labels, preds)

    import nltk
    import string
    from nltk.corpus import stopwords
    stop_words_en = list(set(stopwords.words('english')))
    stop_words_fr = list(set(stopwords.words('french')))
    stop_words_sp = list(set(stopwords.words('spanish')))
    stop_words = set(stop_words_en+stop_words_fr+stop_words_sp)
    
    def update_seeds(seed_words_d, no_filtering, no_tuning):
        tf1 = collections.defaultdict(dict)
        pool1 = collections.defaultdict(dict)
        kl = nn.KLDivLoss()
        for e in test_data.examples:
            #p_orig, label = get_qs(k_pseudolabel, e.text)
            p_orig, label = get_p(k_model, e.text)
            kls = []
            words = []
            #label = LABEL.vocab.itos[label.item()]
            label = e.label
            for i in range(len(e.text)):
                tmp = e.text[i]
                if tmp not in tf1[label]:
                    tf1[label][tmp] = 0
                tf1[label][tmp] += 1
                if tmp in stop_words or tmp in string.punctuation or tmp ==  '<unk>' or tmp == '<pad>':
                    continue
                e.text[i] = '<unk>'
                #p_new, _ = get_qs(k_pseudolabel, e.text)
                p_new, _ = get_p(k_model, e.text)
                loss = kl(torch.log(p_orig.detach()), p_new.detach())
                kls.append(loss.item())
                words.append(tmp)
                e.text[i] = tmp
            lst = list(zip(words, kls))
            lst.sort(key=lambda x: x[1], reverse=True)
            
            #print(lst[:len_])
            if not no_tuning:
                for i in range(len(lst)//4):
                    threshold = 5e-2
                    if lst[i][1] > threshold:
                        if lst[i][0] not in pool1[label]:
                            pool1[label][lst[i][0]] = 0
                        pool1[label][lst[i][0]] += lst[i][1]
                        
        pops1 = collections.defaultdict(dict)
        aspects1 = list(tf1.keys())
        for i in range(len(aspects1)):
            for word in tf1[aspects1[i]]:
                sum_ = 0
                for j in range(len(aspects1)):
                    if word in tf1[aspects1[j]]:
                        sum_ += tf1[aspects1[j]][word]
                pops1[aspects1[i]][word] = tf1[aspects1[i]][word] / sum_

        dists1 = collections.defaultdict(dict)
        for i in range(len(aspects1)):
            for word in tf1[aspects1[i]]:
                max_ = 0
                for j in range(len(aspects1)):
                    if word in tf1[aspects1[j]]:
                        max_ = max(max_, tf1[aspects1[j]][word])
                dists1[aspects1[i]][word] = tf1[aspects1[i]][word] / max_
        
        scores1 = collections.defaultdict(dict)
        for i in range(len(aspects1)):
            if no_tuning:
                for word in tf1[aspects1[i]]:
                    scores1[aspects1[i]][word] = pops1[aspects1[i]][word]*dists1[aspects1[i]][word]
            else:
                for word in pool1[aspects1[i]]:
                    scores1[aspects1[i]][word] = pops1[aspects1[i]][word]*dists1[aspects1[i]][word]

        candidates1 = collections.defaultdict(list)
        for aspect in aspects1:
            candidates1[aspect] = sorted(scores1[aspect].items(), key=lambda x: x[1], reverse=True)
            
        commons1 = set()
        aspects1 = list(candidates1.keys())
        for i in range(len(aspects1)-1):
            for j in range(i+1, len(aspects1)):
                lst1, _ = zip(*candidates1[aspects1[i]])
                lst2, _ = zip(*candidates1[aspects1[j]])
                common = set.intersection(set(lst1), set(lst2))
                for c in common:
                    commons1.add(c)

        miscs = set()
        if not no_filtering:
            tf2 = collections.defaultdict(dict)
            pool2 = collections.defaultdict(dict)
            kl = nn.KLDivLoss()
            for e in test_kplus_data.examples:
                #p_orig, label = get_qs(kplus_pseudolabel, e.text)
                p_orig, label = get_p(kplus_model, e.text)
                kls = []
                words = []
                #label = LABEL.vocab.itos[label.item()]
                label = e.label
                for i in range(len(e.text)):
                    tmp = e.text[i]
                    if tmp not in tf2[label]:
                        tf2[label][tmp] = 0
                    tf2[label][tmp] += 1
                    if tmp in stop_words or tmp in string.punctuation or tmp ==  '<unk>' or tmp == '<pad>':
                        continue
                    e.text[i] = '<unk>'
                    #p_new, _ = get_qs(kplus_pseudolabel, e.text)
                    p_new, _ = get_p(kplus_model, e.text)
                    loss = kl(torch.log(p_orig.detach()), p_new.detach())
                    kls.append(loss.item())
                    words.append(tmp)
                    e.text[i] = tmp
                lst = list(zip(words, kls))
                lst.sort(key=lambda x: x[1], reverse=True)
                
                #print(lst[:len_])
                
                for i in range(len(lst)//4):
                    threshold = 1e-2
                    if lst[i][1] > threshold:
                        if lst[i][0] not in pool2[label]:
                            pool2[label][lst[i][0]] = 0
                        pool2[label][lst[i][0]] += lst[i][1]
                
            pops2 = collections.defaultdict(dict)
            aspects2 = list(tf2.keys())
            for i in range(len(aspects2)):
                for word in tf2[aspects2[i]]:
                    sum_ = 0
                    for j in range(len(aspects2)):
                        if word in tf2[aspects2[j]]:
                            sum_ += tf2[aspects2[j]][word]
                    pops2[aspects2[i]][word] = tf2[aspects2[i]][word] / sum_
                    
            dists2 = collections.defaultdict(dict)
            for i in range(len(aspects2)):
                for word in tf2[aspects2[i]]:
                    max_ = 0
                    for j in range(len(aspects2)):
                        if word in tf2[aspects2[j]]:
                            max_ = max(max_, tf2[aspects2[j]][word])
                    dists2[aspects2[i]][word] = tf2[aspects2[i]][word] / max_
                    
            scores2 = collections.defaultdict(dict)
            for i in range(len(aspects2)):
                for word in pool2[aspects2[i]]:
                    scores2[aspects2[i]][word] = pops2[aspects2[i]][word]*dists2[aspects2[i]][word]

            candidates2 = collections.defaultdict(list)

            for aspect in aspects2:
                candidates2[aspect] = sorted(scores2[aspect].items(), key=lambda x: x[1], reverse=True)
                
            print(candidates2['miscellaneous'])

            for i in range(len(candidates2['miscellaneous'])):
                word, score = candidates2['miscellaneous'][i]
                if score > 1e-2:
                    miscs.add(word)

        for aspect in aspects:
            if not no_filtering:
                for word in miscs:
                    if word in seed_words_d[aspect]:
                        seed_words_d[aspect].remove(word)
            i = 0
            while len(seed_words_d[aspect]) < args.seedword_limit and i < len(candidates1[aspect]):
                word, score = candidates1[aspect][i]
                if word not in seed_words_d[aspect] and word not in commons1 and word not in miscs and score >= args.score_threshold:
                    seed_words_d[aspect].add(word)
                i+=1

        commons2 = set()
        aspects2 = list(seed_words_d.keys())
        for i in range(len(aspects2)-1):
            for j in range(i+1, len(aspects2)):
                lst1 = seed_words_d[aspects2[i]]
                lst2 = seed_words_d[aspects2[j]]
                common = set.intersection(set(lst1), set(lst2))
                for c in common:
                    commons2.add(c)

        for aspect in aspects2:
            for c in commons2:
                if c in seed_words_d[aspect]:
                    seed_words_d[aspect].remove(c)
        
            
    update_seeds(seed_words_d, args.no_filtering, args.no_tuning)
    print(seed_words_d)
    
    for k in seed_words_d:
        seed_words_d[k] = list(seed_words_d[k])
    for k in seed_words_d:
        seed_words_d[k] = set(seed_words_d[k])

    seed_words = sorted(seed_words_d.items(), key=lambda x:LABEL.vocab.stoi[x[0]])
    print(seed_words)
    logging.debug(seed_words)

    get_seed_embedding(seed_words)
    k_model, k_pseudolabel = init_k_model(SEED_WORDS)
    k_model_optimizer = optim.Adam(filter(lambda p: p.requires_grad, k_model.parameters()))

    N_EPOCHS = 5
    for epoch in range(N_EPOCHS):
        print("epoch: ",epoch+1)
        train_loss, train_acc = train(k_model, k_pseudolabel, train_iterator, k_model_optimizer)
        print("training loss: ",train_loss)
        print("training accuracy: ",train_acc)
        valid_acc, valid_f1 = evaluate(k_model, test_data, LABEL)
        print("validation accuracy: ",valid_acc)
        print('validation F1:',valid_f1)
    torch.cuda.empty_cache()

    preds = []
    labels = []
    for e in test_data.examples:
        pred = predict(k_model, e.text)
        preds.append(pred)
        labels.append(LABEL.vocab.stoi[e.label])
    log_info(labels, preds)
    
    logging.debug("k pseudolabel")
    preds = []
    labels = []
    for e in test_data.examples:
        pred = predict_pseudolabel(k_pseudolabel, e.text)
        preds.append(pred)
        labels.append(LABEL.vocab.stoi[e.label])
    log_info(labels, preds)
    
    lst1 = []
    lst2 = []
    for e in test_kplus_data.examples:
        qs,_ = get_qs(k_pseudolabel, e.text)
        preds,_ = get_p(k_model, e.text)
        vs = [v.item() for v in preds.squeeze(0) if v.item() != 0]
        h_norm = (-1/math.log(preds.shape[1]))*sum([v*math.log(v)for v in vs])
        if e.label == 'miscellaneous':
            lst1.append(int(h_norm*100))
        else:
            lst2.append(int(h_norm*100))

    a = np.array(lst2)
    threshold = np.quantile(a, args.quantile)/100
    print("threshold:", threshold)
    logging.debug("threshold:"+str(threshold))

    kplus_model, kplus_pseudolabel = init_kplus_model(SEED_WORDS)
    kplus_model_optimizer = optim.Adam(filter(lambda p: p.requires_grad, kplus_model.parameters()))

    N_EPOCHS = 5
    for epoch in range(N_EPOCHS):
        print("epoch: ",epoch+1)
        train_loss, train_acc = train(kplus_model, kplus_pseudolabel, train_iterator, kplus_model_optimizer)
        print("training loss: ",train_loss)
        print("training accuracy: ",train_acc)
        valid_acc, valid_f1 = evaluate(kplus_model, test_kplus_data, LABEL_KPLUS)
        print("validation accuracy: ",valid_acc)
        print('validation F1:',valid_f1)
    torch.cuda.empty_cache()

    preds = []
    labels = []
    for e in test_kplus_data.examples:
        pred = predict(kplus_model, e.text)
        preds.append(pred)
        labels.append(LABEL_KPLUS.vocab.stoi[e.label])
    log_info(labels, preds)

    logging.debug("kplus pseudolabel")
    preds = []
    labels = []
    for e in test_kplus_data.examples:
        pred = predict_pseudolabel(kplus_pseudolabel, e.text)
        preds.append(pred)
        labels.append(LABEL_KPLUS.vocab.stoi[e.label])
    log_info(labels, preds)

    for i in range(3):
        logging.debug("iteration: " + str(i+1))

        import copy
        seed_words_d_copy = copy.deepcopy(seed_words_d)
        
        update_seeds(seed_words_d, args.no_filtering, args.no_tuning)
        print(seed_words_d)
        
        seed_words = sorted(seed_words_d.items(), key=lambda x:LABEL.vocab.stoi[x[0]])
        print(seed_words)
        logging.debug(seed_words)
        
        if seed_words_d == seed_words_d_copy:
            break

        get_seed_embedding(seed_words)
        k_model, k_pseudolabel = init_k_model(SEED_WORDS)
        k_model_optimizer = optim.Adam(filter(lambda p: p.requires_grad, k_model.parameters()))

        N_EPOCHS = 5
        for epoch in range(N_EPOCHS):
            print("epoch: ",epoch+1)
            train_loss, train_acc = train(k_model, k_pseudolabel, train_iterator, k_model_optimizer)
            print("training loss: ",train_loss)
            print("training accuracy: ",train_acc)
            valid_acc, valid_f1 = evaluate(k_model, test_data, LABEL)
            print("validation accuracy: ",valid_acc)
            print('validation F1:',valid_f1)
        torch.cuda.empty_cache()

        preds = []
        labels = []
        for e in test_data.examples:
            pred = predict(k_model, e.text)
            preds.append(pred)
            labels.append(LABEL.vocab.stoi[e.label])
        log_info(labels, preds)

        logging.debug("k pseudolabel")
        preds = []
        labels = []
        for e in test_data.examples:
            pred = predict_pseudolabel(k_pseudolabel, e.text)
            preds.append(pred)
            labels.append(LABEL.vocab.stoi[e.label])
        log_info(labels, preds)

        lst1 = []
        lst2 = []
        for e in test_kplus_data.examples:
            qs,_ = get_qs(k_pseudolabel, e.text)
            preds,_ = get_p(k_model, e.text)
            vs = [v.item() for v in preds.squeeze(0) if v.item() != 0]
            h_norm = (-1/math.log(preds.shape[1]))*sum([v*math.log(v)for v in vs])
            if e.label == 'miscellaneous':
                lst1.append(int(h_norm*100))
            else:
                lst2.append(int(h_norm*100))

        a = np.array(lst2)
        threshold = np.quantile(a, args.quantile)/100
        print("threshold:", threshold)
        logging.debug("threshold:"+str(threshold))
        
        kplus_model, kplus_pseudolabel = init_kplus_model(SEED_WORDS)
        kplus_model_optimizer = optim.Adam(filter(lambda p: p.requires_grad, kplus_model.parameters()))

        N_EPOCHS_2 = 5
        for epoch in range(N_EPOCHS_2):
            print("epoch: ",epoch+1)
            train_loss, train_acc = train(kplus_model, kplus_pseudolabel, train_iterator, kplus_model_optimizer)
            print("training loss: ",train_loss)
            print("training accuracy: ",train_acc)
            valid_acc, valid_f1 = evaluate(kplus_model, test_kplus_data, LABEL_KPLUS)
            print("validation accuracy: ",valid_acc)
            print('validation F1:',valid_f1)
        torch.cuda.empty_cache()

        preds = []
        labels = []
        for e in test_kplus_data.examples:
            pred = predict(kplus_model, e.text)
            preds.append(pred)
            labels.append(LABEL_KPLUS.vocab.stoi[e.label])
        log_info(labels, preds)
        
        logging.debug("kplus pseudolabel")
        preds = []
        labels = []
        for e in test_kplus_data.examples:
            pred = predict_pseudolabel(kplus_pseudolabel, e.text)
            preds.append(pred)
            labels.append(LABEL_KPLUS.vocab.stoi[e.label])
        log_info(labels, preds)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--dir', type=str, default='yelp/')
    parser.add_argument('--aspects', type=str, default='yelp_aspects.txt')
    parser.add_argument('--train', type=str, default='yelp_train.csv')
    parser.add_argument('--test', type=str, default='yelp_test.csv')
    parser.add_argument('--test_kplus', type=str, default='yelp_test_kplus.csv')
    parser.add_argument('--seedwords', type=str, default='yelp_seed_aspect_words_wovalue.txt')
    parser.add_argument('--embedding', type=str, default='yelp.200d.txt')
    parser.add_argument('--quantile', type=float, default=0.65)
    parser.add_argument('--score_threshold', type=float, default=0.6)
    parser.add_argument('--seedword_limit', type=int, default=10)
    parser.add_argument('--no_filtering', action='store_true')
    parser.add_argument('--no_tuning', action='store_true')
    args = parser.parse_args()
    main(args)