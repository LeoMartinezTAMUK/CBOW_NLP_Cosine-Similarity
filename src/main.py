# Word Embedding using Continous Bag of Words (CBOW) for Natural Language Processing (NLP)
# Created by: Leo Martinez III in Fall 2024

# Ensure that all of these python libraries are properly installed before runtime
import nltk # Natural Language Toolkit (NLTK)
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

def read_pdf(file_path): # method for reading the PDF file
    text = ""
    with open(file_path, 'rb') as file: # 'rb' means read binary
        reader = PyPDF2.PdfReader(file)
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

def tokenize_text(text): # method for converting text into tokens
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence.lower()) for sentence in sentences]
    return words

def find_cosine_similarity(model, word_pairs): # method for cosine similarity
    similarities = []
    for pair in word_pairs:
          similarity = cosine_similarity([model.wv[pair[0]]], [model.wv[pair[1]]])[0][0]
          similarities.append((pair, similarity))
    return similarities

def main(): # main method for invoking all previous methods
    nltk.download('punkt') # retrieve necessary resources

    # read text from PDF
    pdf_file_path = "research_paper.pdf" # about 6 compact pages of length (content is available on IEEE Xplore)
    text = read_pdf(pdf_file_path)

    # tokenize text
    tokenized_text = tokenize_text(text) # tokenize text into words/sentences

    # train Word2Vec model (parameter 'sg=0' means to utilize CBOW instead of Skip-Gram)
    model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=6, sg=0) # perform training on tokens

    # choose word pairs from article (10 word pairs)
    word_pairs = [('intrusion', 'detection'), ('machine', 'learning'), ('cyber', 'threats'),
              ('network', 'behavior'), ('deep', 'neural'), ('anomaly', 'detection'),
              ('signature-based', 'detection'), ('behavior-based', 'detection'),
              ('cyber', 'defenses'), ('ctgan', 'network')]

    # find cosine similarity between word pairs
    similarities = find_cosine_similarity(model, word_pairs)

    # print similarities
    print()
    for pair, similarity in similarities:
        print(f"Similarity between '{pair[0]}' and '{pair[1]}': {similarity}")

    # find pair with highest similarity
    max_similarity_pair = max(similarities, key=lambda pair: pair[1]) # compare each pair value with index [1] as the key
    print(f"\nPair with highest similarity: {max_similarity_pair[0]}, similarity: {max_similarity_pair[1]}")

main() # run the entire program