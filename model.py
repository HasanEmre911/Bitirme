from google.colab import drive
import zipfile, os, re
from tqdm import tqdm

drive.mount('/content/drive', force_remount=True)

zip_path = "/content/drive/MyDrive/cv.zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content')

root = '/content/resumes_corpus'
extract_path = None
for dirpath, dirnames, filenames in os.walk(root):
    if any(f.endswith('.txt') for f in filenames):
        extract_path = dirpath
        break
if extract_path is None:
    raise FileNotFoundError("Couldn't locate directory with .txt files under /content/resume_corpus")
print("✅ Veriler şu dizinde hazır:", extract_path)

import pandas as pd, numpy as np, torch
from collections import Counter
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizerFast, LongformerModel
from torch import nn
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ CUDA kullanılacak:" if device.type=="cuda" else "⚠️ GPU kullanılmıyor.")

texts, raw_labels, fields, edus, exps, certs, imps = [], [], [], [], [], [], []
for fname in sorted(os.listdir(extract_path)):
    if not fname.endswith(".txt"): continue
    base   = fname[:-4]
    txt_fp = os.path.join(extract_path, base + ".txt")
    lab_fp = os.path.join(extract_path, base + ".lab")
    if not os.path.exists(lab_fp): continue

    txt  = open(txt_fp, "r", encoding="utf-8", errors="ignore").read()
    labs = [l.strip() for l in open(lab_fp, "r", encoding="utf-8", errors="ignore") if l.strip()]

    first_line = txt.splitlines()[0]
    m = re.match(r'^(.*?)\s*-\s*', first_line)
    field = m.group(1).strip() if m else ""

    e = re.search(r'(\d+)\s+Years\s+of.*?experience', txt, re.IGNORECASE)
    exp = float(e.group(1)) if e else 0.0

    edu_block = re.search(r'Education\s+(.+?)\s+Skills', txt, re.IGNORECASE|re.DOTALL)
    edu_str   = edu_block.group(1).strip() if edu_block else ""
    if re.search(r'PhD|Doctor', edu_str, re.IGNORECASE): edu="PhD"
    elif re.search(r'Master', edu_str, re.IGNORECASE):    edu="Master"
    elif re.search(r'Bachelor', edu_str, re.IGNORECASE):  edu="Bachelor"
    else:                                                  edu="Other"

    cb = re.search(r'TRAININGS ATTENDED\s*[:\-]?\s*(.+)', txt, re.IGNORECASE|re.DOTALL)
    cert_block = cb.group(1).split("\n\n")[0] if cb else ""
    cert_list = re.findall(r'[•\-\?]\s*([^•\-\?]+)', cert_block) or ([cert_block.strip()] if cert_block.strip() else [])

    texts.append(txt)
    raw_labels.append(labs)
    fields.append(field)
    edus.append(edu)
    exps.append(exp)
    certs.append(cert_list)
    imps.append(labs)

df = pd.DataFrame({
    'text': texts,
    'raw_labels': raw_labels,
    'field': fields,
    'education_level': edus,
    'experience_years': exps,
    'certifications': certs,
    'improvement_labels': imps
})

df['clean_text'] = df['text'].astype(str).apply(lambda x: re.sub(r'<[^>]+>', ' ', x))
df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'\s+', ' ', x)).str.lower()

all_lbl = [" ".join(x.split()[:3]).lower() for sub in df['raw_labels'] for x in sub]
cnts    = Counter(all_lbl)
min_cnt = 5 if any(c>=5 for c in cnts.values()) else 1
freq    = {l for l,c in cnts.items() if c>=min_cnt}
df['filtered_labels'] = df['raw_labels'].apply(lambda lab: [
    " ".join(x.split()[:3]).lower() for x in lab if " ".join(x.split()[:3]).lower() in freq
])
df = df[df['filtered_labels'].map(len)>0].reset_index(drop=True)

ohe       = OneHotEncoder(sparse_output=False).fit(df[['field','education_level']])
cat_feat  = ohe.transform(df[['field','education_level']])
exp_feat  = df[['experience_years']].values.astype(float)
mlb_cert  = MultiLabelBinarizer().fit(df['certifications'])
cert_feat = mlb_cert.transform(df['certifications'])
meta_feat = np.hstack([cat_feat, exp_feat, cert_feat])

mlb_imp = MultiLabelBinarizer().fit(df['improvement_labels'])
Y       = mlb_imp.transform(df['improvement_labels'])
X_tr, X_te, m_tr, m_te, y_tr, y_te = train_test_split(df['clean_text'].tolist(), meta_feat, Y, test_size=0.2, random_state=42)
tok = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')

class LazyEncodingDataset(Dataset):
    def __init__(self, texts, meta, labels, tokenizer, max_len=1024):
        self.texts     = texts
        self.meta_feat = torch.tensor(meta, dtype=torch.float32)
        self.labels    = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len   = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], truncation=True, padding='max_length',
                             max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'metadata':       self.meta_feat[i],
            'labels':         self.labels[i]
        }

train_ds = LazyEncodingDataset(X_tr, m_tr, y_tr, tok)
test_ds  = LazyEncodingDataset(X_te, m_te, y_te, tok)

class CombinedModel(nn.Module):
    def __init__(self, meta_dim, num_labels):
        super().__init__()
        self.text_encoder = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        hidden_size       = self.text_encoder.config.hidden_size
        self.meta_net     = nn.Sequential(nn.Linear(meta_dim,128), nn.ReLU())
        self.classifier   = nn.Linear(hidden_size+128, num_labels)
    def forward(self, input_ids, attention_mask, metadata):
        txt_out  = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0][:,0]
        meta_out = self.meta_net(metadata)
        return self.classifier(torch.cat([txt_out, meta_out], dim=1))

model   = CombinedModel(meta_feat.shape[1], Y.shape[1]).to(device)
opt     = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
loss_fn = BCEWithLogitsLoss()

from tqdm import tqdm
for epoch in range(5):
    model.train()
    loop = tqdm(DataLoader(train_ds, batch_size=16, shuffle=True), leave=True)
    for b in loop:
        opt.zero_grad()
        logits = model(b['input_ids'].to(device),
                       b['attention_mask'].to(device),
                       b['metadata'].to(device))
        loss = loss_fn(logits, b['labels'].to(device))
        loss.backward()
        opt.step()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in DataLoader(test_ds, batch_size=32):
            out = model(b['input_ids'].to(device),
                        b['attention_mask'].to(device),
                        b['metadata'].to(device))
            preds += (torch.sigmoid(out) > 0.5).int().cpu().tolist()
            trues += b['labels'].int().tolist()

    print(f"Epoch {epoch+1} Micro F1:", f1_score(trues, preds, average='micro', zero_division=0))

    ckpt_path = f"/content/drive/MyDrive/epoch_{epoch+1}_checkpoint"
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_path, "model_weights.pt"))
    tok.save_pretrained(ckpt_path)

final_path = "/content/drive/MyDrive/combined_model_final"
os.makedirs(final_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(final_path, "model_weights.pt"))
tok.save_pretrained(final_path)
import pickle

with open(f"{final_path}/ohe.pkl", "wb") as f:
    pickle.dump(ohe, f)

with open(f"{final_path}/mlb_cert.pkl", "wb") as f:
    pickle.dump(mlb_cert, f)

with open(f"{final_path}/mlb_imp.pkl", "wb") as f:
    pickle.dump(mlb_imp, f)

df.to_csv(f"{final_path}/train_data.csv", index=False)