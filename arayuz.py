import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import torch
import numpy as np
import pickle
import os
import pandas as pd
import re
import difflib
import unidecode
import subprocess

from transformers import LongformerTokenizerFast, LongformerModel
from torch import nn

try:
    from googletrans import Translator
    translator = Translator()
except:
    translator = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CERT_STOPWORDS = {
    'professional field', 'professionals', 'professionalism', 'professional manner', 
    'professional services', 'professional experience', 'professional development',
    'professional growth', 'professional environment', 'certifications', 'certificates',
    'certifications/licenses driver\'s license', 'certification and accreditation',
    'professional publications', 'certification and production', 'professional web applications using html5',
    'certification and production', 'certificates/licenses driver\'s license', 'certified',
    'license', 'licenses', 'driver\'s license'
}

def normalize_label(label):
    if not label: return ""
    label = label.lower().replace("_", " ")
    label = unidecode.unidecode(label)
    label = re.sub(r'[^a-z0-9 ]', '', label)
    label = ' '.join(label.split())
    return label

def normalize_labels_list(labels):
    return [normalize_label(l) for l in labels if l]

def load_model_and_tools():
    tokenizer = LongformerTokenizerFast.from_pretrained(BASE_DIR)
    ohe = pickle.load(open(os.path.join(BASE_DIR, "ohe.pkl"), "rb"))
    mlb_cert = pickle.load(open(os.path.join(BASE_DIR, "mlb_cert.pkl"), "rb"))
    mlb_imp = pickle.load(open(os.path.join(BASE_DIR, "mlb_imp.pkl"), "rb"))
    meta_dim = ohe.transform(pd.DataFrame([[ohe.categories_[0][0], ohe.categories_[1][0]]], columns=ohe.feature_names_in_)).shape[1] + 1 + mlb_cert.transform([[]]).shape[1]
    num_labels = len(mlb_imp.classes_)
    model = CombinedModel(meta_dim, num_labels)
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "model_weights.pt"), map_location="cpu"))
    model.eval()
    return model, tokenizer, ohe, mlb_cert, mlb_imp, meta_dim

class CombinedModel(nn.Module):
    def __init__(self, meta_dim, num_labels):
        super().__init__()
        self.text_encoder = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        hidden_size = self.text_encoder.config.hidden_size
        self.meta_net = nn.Sequential(nn.Linear(meta_dim,128), nn.ReLU())
        self.classifier = nn.Linear(hidden_size+128, num_labels)
    def forward(self, input_ids, attention_mask, metadata):
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0][:,0]
        meta_out = self.meta_net(metadata)
        return self.classifier(torch.cat([txt_out, meta_out], dim=1))

def pdf_to_text(pdf_path):
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang="eng") + "\n"
    return text

def fuzzy_match(user_val, model_vals):
    matches = difflib.get_close_matches(user_val.lower(), [v.lower() for v in model_vals], n=1, cutoff=0.4)
    if matches:
        for v in model_vals:
            if v.lower() == matches[0]:
                return v
    return model_vals[0]

def normalize_cert_name(name):
    name = name.lower()
    name = unidecode.unidecode(name)
    name = re.sub(r'[^a-z0-9 ]', ' ', name)
    name = re.sub(r'\b(certificate|certification|course|of|in|for|the|and|training|professional|associate|specialization|license|licensed|completed|program|award|badge|exam|examined|passed|licenses)\b', '', name)
    name = ' '.join(name.split())
    return name.strip()

def extract_certifications(text):
    cert_keywords = [
        "certificate", "certification", "licensed", "diploma", "course", "training", "badge",
        "accreditation", "udemy", "coursera", "linkedin learning", "google cloud", "aws", "azure", "oracle",
        "ibm", "expert", "specialization", "khan academy", "data camp", "bootcamp", "harvardx", 
        "edx", "stanford online", "nanodegree", "codeacademy", "machine learning", "deep learning", "python",
        "data science", "sql", "cloud", "devops", "microsoft", "scrum", "cisco", "security", "project management", "google certified"
    ]
    lines = text.splitlines()
    certs = []
    for line in lines:
        line_lower = line.strip().lower()
        if any(kw in line_lower for kw in cert_keywords):
            clean = line.strip("•-–: .,\t").strip()
            clean_norm = normalize_cert_name(clean)
            if len(clean_norm) > 3 and clean_norm not in CERT_STOPWORDS and not clean_norm.isdigit():
                certs.append(clean_norm)
    return list(dict.fromkeys(certs))

def auto_extract_metadata(text, ohe, mlb_cert):
    field = ""
    all_fields = list(ohe.categories_[0])
    field_keywords = [
        "data scientist", "software engineer", "developer", "front end", "backend", "full stack",
        "java developer", "python developer", "security analyst", "network administrator", "database administrator",
        "web developer", "project manager", "systems administrator", "mobile developer", "ml engineer",
        "ai engineer", "cloud engineer", "devops engineer", "test engineer", "it specialist", "data analyst"
    ] + [f.lower() for f in all_fields]
    text_lower = text.lower()
    for f in field_keywords:
        if f in text_lower:
            field = f
            break
    if not field:
        lines = text.splitlines()
        for l in lines[:5]:
            for f in field_keywords:
                if f in l.lower():
                    field = f
                    break
            if field: break
    field = fuzzy_match(field if field else "", all_fields)

    edu = ""
    all_edus = list(ohe.categories_[1])
    edu_keywords = [
        ("PhD", ["phd", "doctor", "doctorate"]),
        ("Master", ["master", "msc", "m.sc", "grad school"]),
        ("Bachelor", ["bachelor", "undergraduate", "bsc", "b.sc", "b.eng"]),
        ("Associate", ["associate"]),
        ("High School", ["high school", "secondary school", "diploma", "matura"])
    ]
    found_edu = False
    for level, kws in edu_keywords:
        for k in kws:
            if k in text_lower:
                edu = level
                found_edu = True
                break
        if found_edu: break
    if not edu:
        for e in all_edus:
            if e.lower() in text_lower:
                edu = e
                break
    edu = fuzzy_match(edu if edu else "", all_edus)

    exp = 0.0
    exp_patterns = [
        r'(\d{1,2})\s*(\+)?\s*(years|year|yr|yrs)',
        r'(\d{1,2})\s*experience'
    ]
    for pat in exp_patterns:
        match = re.search(pat, text_lower)
        if match:
            exp = float(match.group(1))
            break

    certs = extract_certifications(text)
    all_certs = [normalize_cert_name(c) for c in mlb_cert.classes_]
    certs_final = []
    for c in certs:
        for ac in all_certs:
            similarity = difflib.SequenceMatcher(None, c, ac).ratio()
            if similarity > 0.75 or ac in c or c in ac:
                certs_final.append(ac)
    if not certs_final:
        certs_final = certs
    certs_final = list(dict.fromkeys(certs_final))
    return field, edu, exp, certs_final

def prepare_metadata(field, edu, exp, certs, ohe, mlb_cert):
    meta_cat = ohe.transform(pd.DataFrame([[field, edu]], columns=ohe.feature_names_in_))
    exp = np.array([[exp]])
    cert_vec = mlb_cert.transform([certs])
    meta = np.hstack([meta_cat, exp, cert_vec])
    return meta

def predict_with_model(text, model, tokenizer, meta):
    meta_tensor = torch.tensor(meta, dtype=torch.float32)
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=1024, return_tensors="pt")
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            metadata=meta_tensor
        )
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        return probs

def load_stats_df():
    df = pd.read_csv(os.path.join(BASE_DIR, "train_data.csv"))
    def parse_labels(x):
        try:
            if isinstance(x, str):
                vals = eval(x)
                return normalize_labels_list(vals)
            return []
        except:
            return []
    df["filtered_labels_normalized"] = df["filtered_labels"].apply(parse_labels)
    return df

def get_stats_for_field(df, label_name, cert_classes):
    norm_label = normalize_label(label_name)
    filtered = df[df["filtered_labels_normalized"].apply(lambda labellist: norm_label in labellist)]
    filtered = filtered[filtered["education_level"].str.lower() != "other"]
    n = len(filtered)
    if n == 0:
        return 0, [], 0.0, []
    avg_exp = filtered["experience_years"].mean()
    edu_counts = filtered["education_level"].value_counts().to_dict()
    cert_list = []
    for cl in filtered["certifications"]:
        try:
            cert_list.extend([normalize_cert_name(c) for c in eval(str(cl))])
        except:
            continue
    cert_counts = pd.Series(cert_list).value_counts()
    cert_counts = cert_counts[[c for c in cert_counts.index if c not in CERT_STOPWORDS and len(c)>3]]
    common_certs = list(cert_counts.index[:3])
    return n, common_certs, avg_exp, edu_counts

def match_common_certifications(user_certs, common_certs, threshold=0.75):
    matched = []
    missing = []
    user_certs_lower = [normalize_cert_name(uc) for uc in user_certs]
    for c in common_certs:
        found = False
        for uc in user_certs_lower:
            similarity = difflib.SequenceMatcher(None, c, uc).ratio()
            if similarity > threshold or c in uc or uc in c:
                found = True
                break
        if found:
            matched.append(c)
        else:
            missing.append(c)
    return matched, missing

def build_recommendation(user_certs, user_exp, user_edu, label_name, df, cert_classes):
    n, common_certs, avg_exp, edu_counts = get_stats_for_field(df, label_name, cert_classes)
    matched_certs, missing_certs = match_common_certifications(user_certs, common_certs, threshold=0.75)
    edu_percent = 0
    edu_msg = ""
    if user_edu.lower() != "other" and user_edu in edu_counts and n > 0:
        edu_percent = 100 * edu_counts[user_edu] / n
        edu_msg = f"Senin gibi {label_name}’lerin %{edu_percent:.1f}’i '{user_edu}' eğitim seviyesine sahip."
    elif user_edu.lower() != "other":
        edu_msg = f"Eğitim seviyesi istatistiği bulunamadı."
    else:
        edu_msg = ""
    exp_msg = f"Ortalama tecrübe yılı: {avg_exp:.1f}. Senin tecrüben: {user_exp} yıl."
    if user_exp >= avg_exp:
        exp_comp = "Tecrüben, alan ortalamasının üzerinde veya eşit."
    else:
        exp_comp = "Tecrüben, alan ortalamasının altında. Deneyimini artırman faydalı olabilir."
    if not user_certs:
        cert_msg = "Alanında en sık görülen sertifikalar: " + ", ".join(common_certs) + "."
        if common_certs:
            cert_msg += "\nSenin eksik olduğun önemli sertifikalar: " + ", ".join(common_certs)
        else:
            cert_msg += "\nBu alanda yaygın sertifika bulunamadı."
    elif missing_certs and common_certs:
        cert_msg = "Alanında en sık görülen sertifikalar: " + ", ".join(common_certs) + "."
        cert_msg += "\nSenin eksik olduğun önemli sertifikalar: " + ", ".join(missing_certs)
    elif not missing_certs and common_certs:
        cert_msg = "Alanında en sık görülen sertifikalar: " + ", ".join(common_certs) + "."
        cert_msg += "\nTebrikler! En yaygın sertifikalara sahipsin."
    else:
        cert_msg = "Bu alanda yaygın sertifika bulunamadı."
    msg = f"\n\nKariyer Yol Haritası ve Kıyaslama ({label_name}):\n"
    msg += exp_msg + "\n" + exp_comp + "\n"
    msg += edu_msg + "\n"
    msg += cert_msg + "\n"
    return msg

def clean_llama_output(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    items = []
    for l in lines:
        if l.startswith("-") or re.match(r"^\d+\.", l):
            items.append(l)
        if len(items) == 3:
            break
    if not items:
        items = lines[:3]
    return "\n".join(items)

def get_llama_local_recommendation(user_summary):
    llama_bin = os.path.join(BASE_DIR, "llama.cpp", "build", "bin", "llama-cli")
    llama_model = os.path.join(BASE_DIR, "phi-2.Q4_K_M.gguf")
    prompt = (
    "Below is a summary of a candidate's CV:\n"
    f"{user_summary}\n"
    "Please provide 3 short, actionable, and personalized career suggestions for this person. "
    "Use clear, simple English. Start each suggestion with a dash (-), and write only 3 items."
)
    try:
        proc = subprocess.run(
            [
                llama_bin,
                "-m", llama_model,
                "--n-predict", "300",
                "--temp", "0.7",
                "-p", prompt
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        output = proc.stdout
        out_split = output.split("[/INST]")
        if len(out_split) > 1:
            output = out_split[-1].strip()
        cleaned = clean_llama_output(output) 
        return cleaned if cleaned else "LLM'den çıktı alınamadı."
    except Exception as e:
        return f"LLM önerisi alınırken hata oluştu: {e}"

class CVAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CV Yol Haritası Asistanı")
        self.root.geometry("900x950")
        self.model, self.tokenizer, self.ohe, self.mlb_cert, self.mlb_imp, self.meta_dim = load_model_and_tools()
        self.cv_text = ""
        self.metadata = None
        self.df = load_stats_df()
        self.label = tk.Label(root, text="PDF formatında CV'nizi yükleyin", font=("Arial", 14))
        self.label.pack(pady=10)
        self.upload_button = tk.Button(root, text="CV Yükle (PDF)", command=self.upload_cv, width=30, bg="#2196f3", fg="white", font=("Arial", 12))
        self.upload_button.pack(pady=10)
        self.text_area = scrolledtext.ScrolledText(root, width=90, height=15, font=("Arial", 10))
        self.text_area.pack(pady=10)
        self.text_area.config(state='disabled')
        self.analyze_button = tk.Button(root, text="Yol Haritası Çiz", command=self.analyze_cv, width=30, bg="#4caf50", fg="white", font=("Arial", 12))
        self.analyze_button.pack(pady=10)
        self.analyze_button.config(state='disabled')
        self.eng_button = tk.Button(root, text="İngilizceye Çevir", command=self.translate_to_eng, width=30, bg="#ff9800", fg="white", font=("Arial", 12))
        self.eng_button.pack(pady=10)
        self.eng_button.config(state='disabled')
        self.ai_button = tk.Button(root, text="AI Destekli Kişisel Öneri Al", command=self.get_ai_advice, width=30, bg="#7e57c2", fg="white", font=("Arial", 12))
        self.ai_button.pack(pady=10)
        self.ai_button.config(state='disabled')
        self.result_area = scrolledtext.ScrolledText(root, width=90, height=28, font=("Arial", 11))
        self.result_area.pack(pady=10)
        self.result_area.config(state='disabled')
        self.last_summary = ""

    def upload_cv(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            try:
                self.cv_text = pdf_to_text(file_path)
                self.text_area.config(state='normal')
                self.text_area.delete(1.0, tk.END)
                self.text_area.insert(tk.END, self.cv_text)
                self.text_area.config(state='disabled')
                self.analyze_button.config(state='normal')
                self.eng_button.config(state='normal')
                self.ai_button.config(state='disabled')
            except Exception as e:
                messagebox.showerror("Hata", f"OCR sırasında hata oluştu: {e}")

    def analyze_cv(self):
        if not self.cv_text.strip():
            messagebox.showwarning("Uyarı", "Önce CV yüklemelisiniz.")
            return
        field, edu, exp, certs = auto_extract_metadata(self.cv_text, self.ohe, self.mlb_cert)
        meta = prepare_metadata(field, edu, exp, certs, self.ohe, self.mlb_cert)
        probs = predict_with_model(self.cv_text, self.model, self.tokenizer, meta)
        labels = [self.mlb_imp.classes_[i] for i, v in enumerate(probs) if v > 0.5]
        if labels and probs[np.argmax(probs)] > 0.8:
            main_label = self.mlb_imp.classes_[np.argmax(probs)]
        else:
            if not field or probs[np.argmax(probs)] < 0.5:
                all_labels = list(self.mlb_imp.classes_)
                fuzzy_label = fuzzy_match(field if field else "", all_labels)
                main_label = fuzzy_label if fuzzy_label else self.mlb_imp.classes_[np.argmax(probs)]
            else:
                main_label = field

        result_str = "Otomatik Çıkarılan Bilgiler:\n"
        result_str += f"Alan: {main_label}\n"
        result_str += f"Eğitim: {edu}\n"
        result_str += f"Tecrübe (yıl): {exp}\n"
        result_str += f"Sertifikalar: {', '.join(certs) if certs else 'Bulunamadı'}\n\n"
        result_str += "Tahmin Edilen Etiketler:\n"
        result_str += "\n".join(f"- {lbl} (skor: {probs[i]:.2f})" for i, lbl in enumerate(self.mlb_imp.classes_) if probs[i] > 0.5)
        if not labels:
            result_str += "\n(Uygun etiket bulunamadı.)"
        result_str += "\n\nTüm skorlar:\n"
        result_str += ", ".join([f"{lbl}: {probs[i]:.2f}" for i, lbl in enumerate(self.mlb_imp.classes_)])

        if main_label:
            cert_classes = list(self.mlb_cert.classes_)
            rec_msg = build_recommendation(
                user_certs=certs, user_exp=exp, user_edu=edu,
                label_name=main_label, df=self.df, cert_classes=cert_classes
            )
            result_str += rec_msg
        self.last_summary = result_str
        self.result_area.config(state='normal')
        self.result_area.delete(1.0, tk.END)
        self.result_area.insert(tk.END, result_str)
        self.result_area.config(state='disabled')
        self.ai_button.config(state='normal')

    def get_ai_advice(self):
        if not self.last_summary.strip():
            messagebox.showwarning("Uyarı", "Önce analiz yapmalısınız.")
            return
        messagebox.showinfo("Bilgi", "LLM önerisi hazırlanıyor, lütfen bekleyin...")
        ai_msg = get_llama_local_recommendation(self.last_summary)
        advice_win = tk.Toplevel()
        advice_win.title("LLM Destekli Kişisel Kariyer Önerisi")
        st = scrolledtext.ScrolledText(advice_win, width=90, height=8)
        st.pack(padx=10, pady=10)
        st.insert(tk.END, ai_msg)

    def translate_to_eng(self):
        if not self.cv_text.strip():
            messagebox.showwarning("Uyarı", "Önce CV yüklemelisiniz.")
            return
        if translator is None:
            messagebox.showwarning("Çeviri Kütüphanesi Yok", "googletrans yüklü değil. Kurmak için: pip install googletrans==4.0.0-rc1")
            return
        try:
            tr_text = self.cv_text[:4500]
            eng_text = translator.translate(tr_text, src='tr', dest='en').text
            trans_win = tk.Toplevel()
            trans_win.title("CV İngilizce Çeviri (ilk 4500 karakter)")
            st = scrolledtext.ScrolledText(trans_win, width=90, height=30)
            st.pack(padx=10, pady=10)
            st.insert(tk.END, eng_text)
        except Exception as e:
            messagebox.showerror("Çeviri Hatası", f"Çeviri başarısız: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CVAnalyzerApp(root)
    root.mainloop()