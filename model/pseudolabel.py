import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

def get_q_null(x, threshold, upperbound):
    if x >= threshold:
        return ((x-threshold)/(upperbound-threshold)).item()
    else:
        return 0

def get_hnorm(qs):
    tmp = []
    for q in qs:
        vs = [v.item() for v in q if v.item() != 0]
        h_norm = (-1/math.log(qs.shape[1]))*sum([v*math.log(v)for v in vs])
        tmp.append(torch.tensor(h_norm).unsqueeze(0))
    h_norms = torch.cat(tmp)
    return h_norms

class PseudoLabel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, output_dim, pad_idx, seed_words):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        for param in self.embedding.parameters():
            param.requires_grad = False
        
        #self.conv = nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = (1, embedding_dim))
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = 1, 
                                              kernel_size = (1, embedding_dim)) 
                                    for _ in range(n_filters)
                                    ])
        
        for i in range(len(self.convs)):
            self.convs[i].weight = torch.nn.Parameter(seed_words[i].unsqueeze(0))
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        text = text.permute(1, 0)
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        
        #conved = [F.cosine_similarity(embedded, conv.weight, dim=3) for conv in self.convs] 
        
        #conv_n = [batch size, 1, sent len]
        
        conved = [conv.permute(0, 2, 1) for conv in conved]
            
        #conv_n = [batch size, sent len, 1]
        
        cat = torch.cat(conved,dim=2)
        
        #conv_n = [batch size, sent len, n_filters]
        
        weights = F.max_pool1d(cat, cat.shape[2])
        
        #weights = [batch size, sent len, 1]
        
        embedded = embedded.squeeze(1)
        
        scaled_text = torch.mul(embedded, weights)
        
        #scaled_text = [batch size, sent len, emb dim]
        
        sen_embedded = torch.mean(scaled_text, dim=1)
        
        # sen_embedded = [batch size, emb dim]
        
        sen_embedded = sen_embedded.unsqueeze(1)
        
        # sen_embedded = [batch size, 1, emb dim]
        
        sen_embedded = sen_embedded.unsqueeze(1)
        
        # sen_embedded = [batch size, 1, 1, emb dim]
        
        conved = [F.relu(conv(sen_embedded)).squeeze(3) for conv in self.convs]
        
        #conved = [F.cosine_similarity(sen_embedded, conv.weight, dim=3) for conv in self.convs] 
        
        #conv = [batch size, 1, 1]
        
        cat = torch.cat(conved,dim=2)
            
        #conv = [batch size, 1, n_filters]
        
        q = cat.squeeze(1)
        
        #conv = [batch size, n_filters]
        
        q = F.softmax(q,dim=1)
        
        #q = [batch size, output dim]
        
        """h_norm = get_hnorm(q)
        
        #h_norm = [batch size]
        
        q_null = F.sigmoid(h_norm).unsqueeze(1)
        
        #q_null = [batch size, 1]
        
        q_k = q*(1-q_null)
        
        q_kplus = torch.cat([q_k[:,:2], q_null, q_k[:, 2:]], dim=1)
        
        #q_kplus = [batch size, output dim+1]
        
        fs = torch.sum(q_kplus,dim=0) #[1, output dim+1]
        
        q2fs = torch.div(torch.mul(q_kplus, q_kplus), fs) #[batch size, output dim]"""
        
        fs = torch.sum(q,dim=0) #[1, output dim+1]
        
        q2fs = torch.div(torch.mul(q, q), fs)
        
        sum_ = torch.sum(q2fs,dim=1).unsqueeze(1) #[batch size, 1] 
        
        p = torch.div(q2fs, sum_)
            
        return p, q

class PseudoLabelPlus(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, output_dim, pad_idx, seed_words, k_model, threshold, upperbound, device, LABEL_stoi):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        for param in self.embedding.parameters():
            param.requires_grad = False
        
        #self.conv = nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = (1, embedding_dim))
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = 1, 
                                              kernel_size = (1, embedding_dim)) 
                                    for _ in range(n_filters)
                                    ])
        
        for i in range(len(self.convs)):
            self.convs[i].weight = torch.nn.Parameter(seed_words[i].unsqueeze(0))
            
        self.k_model = k_model
        self.k_model.eval()
        
        self.threshold = threshold
        self.upperbound = upperbound
        self.device = device
        self.LABEL_stoi = LABEL_stoi
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        text = text.permute(1, 0)
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        
        #conved = [F.cosine_similarity(embedded, conv.weight, dim=3) for conv in self.convs] 
        
        #conv_n = [batch size, 1, sent len]
        
        conved = [conv.permute(0, 2, 1) for conv in conved]
            
        #conv_n = [batch size, sent len, 1]
        
        cat = torch.cat(conved,dim=2)
        
        #conv_n = [batch size, sent len, n_filters]
        
        weights = F.max_pool1d(cat, cat.shape[2])
        
        #weights = [batch size, sent len, 1]
        
        embedded = embedded.squeeze(1)
        
        scaled_text = torch.mul(embedded, weights)
        
        #scaled_text = [batch size, sent len, emb dim]
        
        sen_embedded = torch.mean(scaled_text, dim=1)
        
        # sen_embedded = [batch size, emb dim]
        
        sen_embedded = sen_embedded.unsqueeze(1)
        
        # sen_embedded = [batch size, 1, emb dim]
        
        sen_embedded = sen_embedded.unsqueeze(1)
        
        # sen_embedded = [batch size, 1, 1, emb dim]
        
        conved = [F.relu(conv(sen_embedded)).squeeze(3) for conv in self.convs]
        
        #conved = [F.cosine_similarity(sen_embedded, conv.weight, dim=3) for conv in self.convs] 
        
        #conv = [batch size, 1, 1]
        
        cat = torch.cat(conved,dim=2)
            
        #conv = [batch size, 1, n_filters]
        
        q = cat.squeeze(1)
        
        #conv = [batch size, n_filters]
        
        q = F.softmax(q,dim=1)
        
        #q = [batch size, output dim]
        
        pred, _ = self.k_model(text.permute(1, 0))
        
        h_norm = get_hnorm(pred)
        
        #h_norm = [batch size]
        
        #q_null = F.sigmoid(h_norm).unsqueeze(1)
        
        q_null = torch.FloatTensor(list(map(lambda p: get_q_null(p, self.threshold, self.upperbound), h_norm))).unsqueeze(1).to(self.device)
        
        #q_null = [batch size, 1]
        
        q_k = q*(1-q_null)
        
        misc_idx = self.LABEL_stoi['miscellaneous']
        
        q_kplus = torch.cat([q_k[:,:misc_idx], q_null, q_k[:, misc_idx:]], dim=1)
        
        #q_kplus = [batch size, output dim+1]
        
        fs = torch.sum(q_kplus,dim=0) #[1, output dim+1]
        
        q2fs = torch.div(torch.mul(q_kplus, q_kplus), fs) #[batch size, output dim]
        
        sum_ = torch.sum(q2fs,dim=1).unsqueeze(1) #[batch size, 1] 
        
        p = torch.div(q2fs, sum_)
            
        return p, q_kplus