import re
import nltk
import urllib.request
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter

# 1. Setup & Downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

# 2. Load the actual text of Moby Dick
def load_moby_dick():
    url = "https://www.gutenberg.org/files/2701/2701-0.txt"
    with urllib.request.urlopen(url) as response:
        return response.read().decode('utf-8')

# 3. Noise Removal
def clean_text(text):
    # Keep periods for sentence tokenization, remove other special chars/numbers
    cleaned_text = re.sub(r'[^a-zA-Z\s.]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# 4. Lemmatization Helper (Maps NLTK tags to WordNet tags)
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wordnet.ADJ
    elif treebank_tag.startswith('V'): return wordnet.VERB
    elif treebank_tag.startswith('N'): return wordnet.NOUN
    elif treebank_tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

# --- Execution Flow ---

raw_text = load_moby_dick()
# We'll take a large slice to keep it fast but representative
moby_dick_text = raw_text[5000:55000] 
cleaned = clean_text(moby_dick_text)

# Tokenization
sentences = nltk.sent_tokenize(cleaned)
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Stop Words & POS Tagging
stop_words = set(stopwords.words('english'))
tagged_sentences = []

for sent in tokenized_sentences:
    # Filter stop words and tag
    filtered = [w for w in sent if w.lower() not in stop_words]
    tagged_sentences.append(nltk.pos_tag(filtered))

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tagged = []
for sent in tagged_sentences:
    lem_sent = [(lemmatizer.lemmatize(w, get_wordnet_pos(t)), t) for w, t in sent]
    lemmatized_tagged.append(lem_sent)

# 5. Chunking
# Pattern updated: Adjective(s) followed by a Noun
chunking_patterns = "NP: {<JJ>*<NN.>+}" 
chunk_parser = nltk.RegexpParser(chunking_patterns)

# 6. Extraction & Counting
def get_top_phrases(tagged_data, label, limit=10):
    phrases = []
    for sent in tagged_data:
        tree = chunk_parser.parse(sent)
        for subtree in tree.subtrees(filter=lambda t: t.label() == label):
            phrase = " ".join([word for word, tag in subtree.leaves()])
            phrases.append(phrase.lower())
    return Counter(phrases).most_common(limit)

# Final Output
print("--- Top 10 Noun Phrases in Moby Dick ---")
top_nps = get_top_phrases(lemmatized_tagged, 'NP')
for phrase, count in top_nps:
    print(f"{phrase:20} | Count: {count}")