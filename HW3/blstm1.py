# Sohail Haresh Gidwani
# USC ID: 7321203258

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import argparse, os, random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---- data parsing ----

def load_sents(fpath, labeled=True):
    sents, cur = [], []
    with open(fpath) as f:
        for ln in f:
            ln = ln.rstrip('\n')
            if ln.strip() == '':
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            parts = ln.split()
            idx = int(parts[0])
            word = parts[1]
            tag = parts[2] if labeled and len(parts) >= 3 else None
            cur.append((idx, word, tag))
    if cur:
        sents.append(cur)
    return sents

def build_w2i(sents):
    wc = {}
    for s in sents:
        for _, w, _ in s:
            wc[w] = wc.get(w, 0) + 1
    w2i = {'<pad>': 0, '<unk>': 1}
    for w in sorted(wc):
        w2i[w] = len(w2i)
    return w2i

def build_t2i(sents):
    tags = set()
    for s in sents:
        for _, _, t in s:
            if t: tags.add(t)
    return {t: i for i, t in enumerate(sorted(tags))}


# ---- dataset ----

class NERSet(Dataset):
    def __init__(self, sents, w2i, t2i=None):
        self.items = []
        unk = w2i['<unk>']
        for s in sents:
            wids = [w2i.get(w, unk) for _, w, _ in s]
            tids = [t2i[t] for _, _, t in s] if (t2i and s[0][2] is not None) else None
            raw = [(i, w) for i, w, _ in s]
            self.items.append((wids, tids, raw))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def pad_batch(batch):
    wids_l, tids_l, raw_l = zip(*batch)
    lens = torch.tensor([len(w) for w in wids_l])
    wids = pad_sequence([torch.tensor(w, dtype=torch.long) for w in wids_l],
                        batch_first=True, padding_value=0)
    if tids_l[0] is not None:
        tids = pad_sequence([torch.tensor(t, dtype=torch.long) for t in tids_l],
                            batch_first=True, padding_value=-1)
    else:
        tids = None
    return wids, tids, lens, raw_l


# ---- model ----

class NERTagger(nn.Module):
    def __init__(self, vsz, n_tags, emb_dim=100, hid=256, fc_dim=128, drop=0.33):
        super().__init__()
        self.emb = nn.Embedding(vsz, emb_dim, padding_idx=0)
        nn.init.uniform_(self.emb.weight, -0.05, 0.05)
        self.emb.weight.data[0].fill_(0)

        self.drop = nn.Dropout(drop)
        self.lstm = nn.LSTM(emb_dim, hid, num_layers=1, batch_first=True,
                            bidirectional=True, dropout=drop)
        self.fc = nn.Linear(hid * 2, fc_dim)
        self.elu = nn.ELU()
        self.clf = nn.Linear(fc_dim, n_tags)

    def forward(self, x, lengths):
        e = self.drop(self.emb(x))
        pk = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(pk)
        h, _ = pad_packed_sequence(h, batch_first=True)
        h = self.drop(h)
        return self.clf(self.elu(self.fc(h)))


# ---- train / eval helpers ----

def run_epoch(model, loader, opt, crit, dev, clip=5.0):
    model.train()
    total_loss, nb = 0, 0
    for wids, tids, lens, _ in loader:
        wids, tids = wids.to(dev), tids.to(dev)
        opt.zero_grad()
        logits = model(wids, lens.to(dev))
        loss = crit(logits.view(-1, logits.size(-1)), tids.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total_loss += loss.item()
        nb += 1
    return total_loss / nb

@torch.no_grad()
def eval_loss(model, loader, crit, dev):
    model.eval()
    total, nb, correct, total_tok = 0, 0, 0, 0
    for wids, tids, lens, _ in loader:
        wids, tids = wids.to(dev), tids.to(dev)
        logits = model(wids, lens.to(dev))
        loss = crit(logits.view(-1, logits.size(-1)), tids.view(-1))
        total += loss.item()
        nb += 1
        preds = logits.argmax(-1)
        mask = tids != -1
        correct += ((preds == tids) & mask).sum().item()
        total_tok += mask.sum().item()
    return total / nb, correct / total_tok if total_tok > 0 else 0

@torch.no_grad()
def predict_all(model, loader, i2t, dev):
    model.eval()
    all_sents = []
    for wids, _, lens, raw_batch in loader:
        wids = wids.to(dev)
        logits = model(wids, lens.to(dev))
        preds = logits.argmax(-1).cpu()
        for b in range(len(lens)):
            slen = lens[b].item()
            sent = []
            for j in range(slen):
                idx, word = raw_batch[b][j]
                tag = i2t[preds[b][j].item()]
                sent.append((idx, word, tag))
            all_sents.append(sent)
    return all_sents

def write_output(pred_sents, fpath):
    with open(fpath, 'w') as f:
        for sent in pred_sents:
            for idx, word, tag in sent:
                f.write('{} {} {}\n'.format(idx, word, tag))
            f.write('\n')


# ---- main ----

def main():
    pa = argparse.ArgumentParser(description='Task 1: BiLSTM NER')
    pa.add_argument('--mode', choices=['train', 'predict'], default='train')
    pa.add_argument('--data', default='data')
    pa.add_argument('--model_file', default='blstm1.pt')
    pa.add_argument('--lr', type=float, default=0.15)
    pa.add_argument('--momentum', type=float, default=0.9)
    pa.add_argument('--epochs', type=int, default=30)
    pa.add_argument('--bs', type=int, default=32)
    pa.add_argument('--clip', type=float, default=5.0)
    pa.add_argument('--device', default=None)
    args = pa.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('device:', device)

    # vocab + tags always built from train
    train_s = load_sents(os.path.join(args.data, 'train'))
    w2i = build_w2i(train_s)
    t2i = build_t2i(train_s)
    i2t = {v: k for k, v in t2i.items()}
    print('vocab:', len(w2i), '| tags:', len(t2i))

    if args.mode == 'train':
        dev_s = load_sents(os.path.join(args.data, 'dev'))
        train_dl = DataLoader(NERSet(train_s, w2i, t2i), batch_size=args.bs,
                              shuffle=True, collate_fn=pad_batch)
        dev_dl = DataLoader(NERSet(dev_s, w2i, t2i), batch_size=args.bs,
                            shuffle=False, collate_fn=pad_batch)

        model = NERTagger(len(w2i), len(t2i)).to(device)
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        sched = optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.5)
        crit = nn.CrossEntropyLoss(ignore_index=-1)

        best = 1e9
        for ep in range(1, args.epochs + 1):
            tr = run_epoch(model, train_dl, opt, crit, device, args.clip)
            dv, acc = eval_loss(model, dev_dl, crit, device)
            lr_now = sched.get_last_lr()[0]
            print('ep %02d | train %.4f | dev %.4f | acc %.4f | lr %.5f' % (ep, tr, dv, acc, lr_now))
            sched.step()

            if dv < best:
                best = dv
                torch.save({'model': model.state_dict(), 'w2i': w2i, 't2i': t2i},
                           args.model_file)
                print('  -> saved')

        # reload best and write predictions
        ckpt = torch.load(args.model_file, weights_only=False, map_location=device)
        model.load_state_dict(ckpt['model'])

        preds = predict_all(model, dev_dl, i2t, device)
        write_output(preds, 'dev1.out')
        print('wrote dev1.out')

        test_s = load_sents(os.path.join(args.data, 'test'), labeled=False)
        test_dl = DataLoader(NERSet(test_s, w2i), batch_size=args.bs,
                             shuffle=False, collate_fn=pad_batch)
        preds = predict_all(model, test_dl, i2t, device)
        write_output(preds, 'test1.out')
        print('wrote test1.out')

    elif args.mode == 'predict':
        ckpt = torch.load(args.model_file, weights_only=False, map_location=device)
        w2i = ckpt['w2i']
        t2i = ckpt['t2i']
        i2t = {v: k for k, v in t2i.items()}

        model = NERTagger(len(w2i), len(t2i)).to(device)
        model.load_state_dict(ckpt['model'])

        dev_s = load_sents(os.path.join(args.data, 'dev'))
        dev_dl = DataLoader(NERSet(dev_s, w2i, t2i), batch_size=args.bs,
                            shuffle=False, collate_fn=pad_batch)
        preds = predict_all(model, dev_dl, i2t, device)
        write_output(preds, 'dev1.out')

        test_s = load_sents(os.path.join(args.data, 'test'), labeled=False)
        test_dl = DataLoader(NERSet(test_s, w2i), batch_size=args.bs,
                             shuffle=False, collate_fn=pad_batch)
        preds = predict_all(model, test_dl, i2t, device)
        write_output(preds, 'test1.out')
        print('wrote dev1.out, test1.out')


if __name__ == '__main__':
    main()
