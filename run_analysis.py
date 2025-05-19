# === Install required packages (for reference only, don't run in script)
# !pip install spacy metaphone jellyfish rapidfuzz sentence-transformers xlsxwriter
# !python -m spacy download en_core_web_trf
# !pip install -U spacy[transformers]

import pandas as pd
import spacy
import unicodedata
from metaphone import doublemetaphone
import jellyfish
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import requests
import io
from itertools import chain
from datetime import datetime, timedelta

# === Dynamic date range for the previous month ===
today = datetime.today()
first_day_prev_month = today.replace(day=1) - timedelta(days=1)
start_date_str = first_day_prev_month.replace(day=1).strftime('%Y-%m-%d')
end_date_str = first_day_prev_month.strftime('%Y-%m-%d')

# === Download dataset from Apify ===
API_TOKEN = "apify_api_R0fcsZbJ52Z2Ea2lQmab1jyQbZVaA14jGi0i"
DATASET_ID = "fMj8MMlcjWp9NkIh3"
URL = f"https://api.apify.com/v2/datasets/{DATASET_ID}/items?clean=true&format=csv"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
df = pd.read_csv(io.StringIO(requests.get(URL, headers=headers).text)).rename(columns={'text': 'review_text'})

# === Filter by publishedDate range ===
df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce')
mask = (df['publishedDate'] >= start_date_str) & (df['publishedDate'] <= end_date_str)
df = df.loc[mask].copy()

# === Load models ===
nlp = spacy.load("en_core_web_trf")
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Known and irrelevant names ===
known_names = ['Seánna', 'Aine', 'Anna', 'Brooklyn', 'Cain', 'Corey',
               'Dáire', 'Dan', 'Declan', 'Holly', 'Lyndsey',
               'Rachel', 'Rebecca', 'Rhianna', 'Sam', 'Sarah', 'Stuart', 'Alex']

irrelevant_names = {'titanic', 'jack the ripper', 'ripper', 'room', 'harry', 'harry potter',
                   'jack', 'jack the ripped', 'timescape', 'escape', 'team', 'crew', 'game',
                   'experience', 'staff', 'guide', 'host'}

def normalize(text):
    if not isinstance(text, str):
        return ''
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c)).lower().strip()

# === Build phonetic index ===
phonetic_index = {}
for name in known_names:
    norm = normalize(name)
    for code in doublemetaphone(norm):
        if code:
            phonetic_index.setdefault(code, set()).add(name)

known_name_vecs = model.encode(known_names, convert_to_tensor=True)

# === Matching functions ===
def fuzzy_phonetic_match(norm_name):
    dm1, dm2 = doublemetaphone(norm_name)
    candidates = phonetic_index.get(dm1, set()) | phonetic_index.get(dm2, set())
    for cand in candidates:
        cand_norm = normalize(cand)
        max_len = max(len(norm_name), len(cand_norm))
        if (jellyfish.jaro_winkler_similarity(norm_name, cand_norm) >= (0.75 if max_len <= 4 else 0.85)
            or jellyfish.levenshtein_distance(norm_name, cand_norm) <= 2
            or fuzz.token_sort_ratio(norm_name, cand_norm) >= (70 if max_len <= 4 else 85)):
            return cand
    return None

def match_names(names):
    matched = set()
    for name in names or []:
        norm = normalize(name)
        cand = fuzzy_phonetic_match(norm)
        if cand:
            matched.add(cand)
            continue
        try:
            name_vec = model.encode(name, convert_to_tensor=True)
            sim_scores = util.cos_sim(name_vec, known_name_vecs)[0]
            best_idx = sim_scores.argmax().item()
            if sim_scores[best_idx].item() >= 0.7:
                matched.add(known_names[best_idx])
        except Exception:
            pass
    return sorted(matched)

# === Name extraction ===
def extract_names(text):
    if not isinstance(text, str) or not text.strip():
        return []

    doc = nlp(text)
    names = {ent.text for ent in doc.ents if ent.label_ == "PERSON"}
    names |= {token.text for token in doc if token.dep_ in ('nsubj', 'nsubjpass') and token.pos_ == "PROPN"}

    for token in doc:
        if token.text[0].isupper() and token.pos_ in {"PROPN", "NOUN"}:
            norm = normalize(token.text)
            if any(normalize(k) == norm for k in known_names):
                names.add(token.text)
            elif fuzzy_phonetic_match(norm):
                names.add(token.text)

    return list({n for n in names if normalize(n) not in irrelevant_names})

# === Apply logic ===
df['extracted_names'] = df['review_text'].apply(extract_names)
df['matched_exact_names'] = df['extracted_names'].apply(match_names)
df['matched_names_display'] = df['matched_exact_names'].apply(lambda x: ', '.join(x))

# === Count matches ===
all_matches = list(chain.from_iterable(df['matched_exact_names']))
name_counts = pd.Series(all_matches).value_counts().reset_index()
name_counts.columns = ['matched_exact_name', 'count']

# === Separate matched/unmatched ===
matched_reviews_df = df[df['matched_exact_names'].map(bool)].copy()
unmatched_df = df[(df['extracted_names'].map(len) == 1) & (df['matched_exact_names'].map(len) == 0)].copy()
unmatched_df['unmatched_name'] = unmatched_df['extracted_names'].str[0]
unmatched_counts = unmatched_df['unmatched_name'].value_counts().reset_index()
unmatched_counts.columns = ['unmatched_name', 'count']
unmatched_df = unmatched_df.merge(unmatched_counts, on='unmatched_name').sort_values('count', ascending=False)

# === Format dates ===
matched_reviews_df['publishedDate'] = matched_reviews_df['publishedDate'].dt.date
unmatched_df['publishedDate'] = unmatched_df['publishedDate'].dt.date

# === Output file setup ===
month_str = pd.to_datetime(start_date_str).strftime('%B')
year_str = pd.to_datetime(start_date_str).strftime('%Y')
matched_sheet = f"Matched-{year_str} {month_str}"
unmatched_sheet = f"Unmatched-{year_str} {month_str}"

# === Write to Excel ===
final_excel = 'final_output.xlsx'
with pd.ExcelWriter(final_excel, engine='xlsxwriter') as writer:
    name_counts.to_excel(writer, matched_sheet, index=False, startrow=0)
    matched_reviews_df[['publishedDate', 'matched_names_display', 'review_text']].rename(
        columns={'matched_names_display': 'matched_exact_name'}
    ).to_excel(writer, matched_sheet, index=False, startrow=len(name_counts) + 3)

    unmatched_counts.to_excel(writer, unmatched_sheet, index=False, startrow=0)
    unmatched_df[['publishedDate', 'unmatched_name', 'review_text']].to_excel(
        writer, unmatched_sheet, index=False, startrow=len(unmatched_counts) + 3)

    workbook = writer.book
    writer.sheets[matched_sheet].set_tab_color('#90ee90')  # Green
    writer.sheets[unmatched_sheet].set_tab_color('#ffcccb')  # Red
