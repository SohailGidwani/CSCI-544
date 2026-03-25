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

N_CASES = 6
CASE_DIM = 10


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

def build_c2i(sents):
    chars = set()
    for s in sents:
        for _, w, _ in s:
            for c in w:
                chars.add(c)
    c2i = {'<pad>': 0, '<unk>': 1}
    for c in sorted(chars):
        c2i[c] = len(c2i)
    return c2i


# ---- GloVe ----

def load_glove(fpath):
    print('loading glove from', fpath)
    vecs = {}
    with open(fpath, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            vecs[word] = vec
    print('glove loaded: %d vectors, dim=%d' % (len(vecs), len(next(iter(vecs.values())))))
    return vecs

def make_emb_matrix(w2i, glove, emb_dim=100):
    n = len(w2i)
    mat = np.random.uniform(-0.05, 0.05, (n, emb_dim)).astype(np.float32)
    mat[0] = 0
    found = 0
    for w, idx in w2i.items():
        if idx < 2:
            continue
        wl = w.lower()
        if wl in glove:
            mat[idx] = glove[wl]
            found += 1
        elif w in glove:
            mat[idx] = glove[w]
            found += 1
    print('glove coverage: %d / %d (%.1f%%)' % (found, n, 100 * found / n))
    return mat


# ---- case features ----

def get_case(w):
    if any(c.isdigit() for c in w):
        return 4
    if w.islower():
        return 1
    if w.isupper():
        return 2
    if w[0].isupper():
        return 3
    return 5


# ---- dataset ----

class NERSetCNN(Dataset):
    def __init__(self, sents, w2i, c2i, t2i=None):
        self.items = []
        unk_w = w2i['<unk>']
        unk_c = c2i['<unk>']
        for s in sents:
            wids = [w2i.get(w, unk_w) for _, w, _ in s]
            cids = [get_case(w) for _, w, _ in s]
            chars = [[c2i.get(c, unk_c) for c in w] for _, w, _ in s]
            tids = [t2i[t] for _, _, t in s] if (t2i and s[0][2] is not None) else None
            raw = [(i, w) for i, w, _ in s]
            self.items.append((wids, cids, chars, tids, raw))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def pad_batch_cnn(batch):
    wids_l, cids_l, chars_l, tids_l, raw_l = zip(*batch)
    lens = torch.tensor([len(w) for w in wids_l])
    max_slen = max(lens).item()

    wids = pad_sequence([torch.tensor(w, dtype=torch.long) for w in wids_l],
                        batch_first=True, padding_value=0)
    cids = pad_sequence([torch.tensor(c, dtype=torch.long) for c in cids_l],
                        batch_first=True, padding_value=0)

    max_wlen = max(len(c) for chars in chars_l for c in chars)
    char_pad = torch.zeros(len(batch), max_slen, max_wlen, dtype=torch.long)
    for b, chars in enumerate(chars_l):
        for w, cseq in enumerate(chars):
            char_pad[b, w, :len(cseq)] = torch.tensor(cseq, dtype=torch.long)

    if tids_l[0] is not None:
        tids = pad_sequence([torch.tensor(t, dtype=torch.long) for t in tids_l],
                            batch_first=True, padding_value=-1)
    else:
        tids = None
    return wids, cids, char_pad, tids, lens, raw_l


# ---- model ----

class CharCNN(nn.Module):
    def __init__(self, n_chars, char_emb=30, n_filters=50, kernel=3):
        super().__init__()
        self.cemb = nn.Embedding(n_chars, char_emb, padding_idx=0)
        self.conv = nn.Conv1d(char_emb, n_filters, kernel, padding=kernel // 2)
        self.n_filters = n_filters

    def forward(self, char_ids):
        e = self.cemb(char_ids)
        e = e.transpose(1, 2)
        c = self.conv(e)
        c = c.max(dim=2)[0]
        return c


class NERTaggerCNN(nn.Module):
    def __init__(self, vsz, n_tags, emb_matrix, n_chars,
                 n_cases=N_CASES, case_dim=CASE_DIM,
                 char_emb=30, n_filters=50, kernel=3,
                 hid=256, fc_dim=128, drop=0.33):
        super().__init__()
        emb_dim = emb_matrix.shape[1]
        self.word_emb = nn.Embedding(vsz, emb_dim, padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.case_emb = nn.Embedding(n_cases, case_dim, padding_idx=0)
        self.char_cnn = CharCNN(n_chars, char_emb, n_filters, kernel)

        lstm_in = emb_dim + case_dim + n_filters
        self.drop = nn.Dropout(drop)
        self.lstm = nn.LSTM(lstm_in, hid, num_layers=1, batch_first=True,
                            bidirectional=True, dropout=drop)
        self.fc = nn.Linear(hid * 2, fc_dim)
        self.elu = nn.ELU()
        self.clf = nn.Linear(fc_dim, n_tags)

    def forward(self, wids, cids, char_ids, lengths):
        B, S = wids.shape
        we = self.word_emb(wids)
        ce = self.case_emb(cids)

        char_flat = char_ids.view(B * S, -1)
        char_feat = self.char_cnn(char_flat)
        char_feat = char_feat.view(B, S, -1)

        x = torch.cat([we, ce, char_feat], dim=-1)
        x = self.drop(x)
        pk = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(pk)
        h, _ = pad_packed_sequence(h, batch_first=True)
        h = self.drop(h)
        return self.clf(self.elu(self.fc(h)))


# ---- train / eval ----

def run_epoch(model, loader, opt, crit, device, clip=5.0):
    model.train()
    total_loss, nb = 0, 0
    for wids, cids, char_ids, tids, lens, _ in loader:
        wids = wids.to(device)
        cids = cids.to(device)
        char_ids = char_ids.to(device)
        tids = tids.to(device)
        opt.zero_grad()
        logits = model(wids, cids, char_ids, lens.to(device))
        loss = crit(logits.view(-1, logits.size(-1)), tids.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total_loss += loss.item()
        nb += 1
    return total_loss / nb

@torch.no_grad()
def eval_loss(model, loader, crit, device):
    model.eval()
    total, nb, correct, total_tok = 0, 0, 0, 0
    for wids, cids, char_ids, tids, lens, _ in loader:
        wids = wids.to(device)
        cids = cids.to(device)
        char_ids = char_ids.to(device)
        tids = tids.to(device)
        logits = model(wids, cids, char_ids, lens.to(device))
        loss = crit(logits.view(-1, logits.size(-1)), tids.view(-1))
        total += loss.item()
        nb += 1
        preds = logits.argmax(-1)
        mask = tids != -1
        correct += ((preds == tids) & mask).sum().item()
        total_tok += mask.sum().item()
    return total / nb, correct / total_tok if total_tok > 0 else 0

@torch.no_grad()
def predict_all(model, loader, i2t, device):
    model.eval()
    all_sents = []
    for wids, cids, char_ids, _, lens, raw_batch in loader:
        wids = wids.to(device)
        cids = cids.to(device)
        char_ids = char_ids.to(device)
        logits = model(wids, cids, char_ids, lens.to(device))
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
    pa = argparse.ArgumentParser(description='Bonus: BiLSTM-CNN with char features')
    pa.add_argument('--mode', choices=['train', 'predict'], default='train')
    pa.add_argument('--data', default='data')
    pa.add_argument('--glove', default='glove.6B.100d')
    pa.add_argument('--model_file', default='blstm_cnn.pt')
    pa.add_argument('--emb_lr', type=float, default=0.001)
    pa.add_argument('--lr', type=float, default=0.1)
    pa.add_argument('--momentum', type=float, default=0.9)
    pa.add_argument('--epochs', type=int, default=50)
    pa.add_argument('--bs', type=int, default=32)
    pa.add_argument('--clip', type=float, default=5.0)
    pa.add_argument('--char_emb', type=int, default=30)
    pa.add_argument('--n_filters', type=int, default=50)
    pa.add_argument('--kernel', type=int, default=3)
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

    train_s = load_sents(os.path.join(args.data, 'train'))
    w2i = build_w2i(train_s)
    t2i = build_t2i(train_s)
    i2t = {v: k for k, v in t2i.items()}
    c2i = build_c2i(train_s)
    print('vocab:', len(w2i), '| tags:', len(t2i), '| chars:', len(c2i))

    if args.mode == 'train':
        glove = load_glove(args.glove)

        dev_s = load_sents(os.path.join(args.data, 'dev'))
        test_s = load_sents(os.path.join(args.data, 'test'), labeled=False)
        extra = 0
        for sents in [dev_s, test_s]:
            for s in sents:
                for _, w, _ in s:
                    if w not in w2i and w.lower() in glove:
                        w2i[w] = len(w2i)
                        extra += 1
        i2t = {v: k for k, v in t2i.items()}
        print('expanded vocab by %d -> %d total' % (extra, len(w2i)))

        emb_mat = make_emb_matrix(w2i, glove, emb_dim=100)
        del glove

        train_dl = DataLoader(NERSetCNN(train_s, w2i, c2i, t2i), batch_size=args.bs,
                              shuffle=True, collate_fn=pad_batch_cnn)
        dev_dl = DataLoader(NERSetCNN(dev_s, w2i, c2i, t2i), batch_size=args.bs,
                            shuffle=False, collate_fn=pad_batch_cnn)

        model = NERTaggerCNN(len(w2i), len(t2i), emb_mat, len(c2i),
                             char_emb=args.char_emb, n_filters=args.n_filters,
                             kernel=args.kernel).to(device)

        emb_params = list(model.word_emb.parameters())
        emb_ids = set(id(p) for p in emb_params)
        other_params = [p for p in model.parameters() if id(p) not in emb_ids]
        opt = optim.SGD([
            {'params': emb_params, 'lr': args.emb_lr},
            {'params': other_params, 'lr': args.lr}
        ], momentum=args.momentum)
        sched = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
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
                torch.save({'model': model.state_dict(), 'w2i': w2i, 't2i': t2i,
                            'c2i': c2i},
                           args.model_file)
                print('  -> saved')

        ckpt = torch.load(args.model_file, weights_only=False, map_location=device)
        model.load_state_dict(ckpt['model'])

        dev_preds = predict_all(model, dev_dl, i2t, device)
        write_output(dev_preds, 'dev_cnn.out')
        print('wrote dev_cnn.out')

        test_dl = DataLoader(NERSetCNN(test_s, w2i, c2i), batch_size=args.bs,
                             shuffle=False, collate_fn=pad_batch_cnn)
        test_preds = predict_all(model, test_dl, i2t, device)
        write_output(test_preds, 'pred')
        print('wrote pred')

    elif args.mode == 'predict':
        ckpt = torch.load(args.model_file, weights_only=False, map_location=device)
        w2i = ckpt['w2i']
        t2i = ckpt['t2i']
        c2i = ckpt['c2i']
        i2t = {v: k for k, v in t2i.items()}

        dummy_mat = np.zeros((len(w2i), 100), dtype=np.float32)
        model = NERTaggerCNN(len(w2i), len(t2i), dummy_mat, len(c2i)).to(device)
        model.load_state_dict(ckpt['model'])

        dev_s = load_sents(os.path.join(args.data, 'dev'))
        dev_dl = DataLoader(NERSetCNN(dev_s, w2i, c2i, t2i), batch_size=args.bs,
                            shuffle=False, collate_fn=pad_batch_cnn)
        dev_preds = predict_all(model, dev_dl, i2t, device)
        write_output(dev_preds, 'dev_cnn.out')

        test_s = load_sents(os.path.join(args.data, 'test'), labeled=False)
        test_dl = DataLoader(NERSetCNN(test_s, w2i, c2i), batch_size=args.bs,
                             shuffle=False, collate_fn=pad_batch_cnn)
        test_preds = predict_all(model, test_dl, i2t, device)
        write_output(test_preds, 'pred')
        print('wrote dev_cnn.out, pred')


if __name__ == '__main__':
    main()
