from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('tokenizers')
import warnings
warnings.filterwarnings(action='ignore')



import zipfile

with zipfile.ZipFile("./phase-1/feature-engineering/data/Gutenburg.zip", 'r') as zip_ref:
    file_name = zip_ref.namelist()[0]
    with zip_ref.open(file_name) as file:
        content = file.read().decode('utf-8', errors='ignore')
        cleaned_text = content.replace("\n", " ")
        print("File loaded")

data = []

for i in sent_tokenize(cleaned_text):
    temp = []

    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

model1 = Word2Vec(data, min_count=1, vector_size=100, window=9, sg=0)

model2 = Word2Vec(data, min_count=1, vector_size=100, window=10, sg=1)

print("Cosine similarity between 'alice' " +
      "and 'wonderland' - CBOW : ",
      model1.wv.similarity('alice', 'wonderland'))

print("Cosine similarity between 'alice' " +
      "and 'machines' - CBOW : ",
      model1.wv.similarity('alice', 'machines'))

