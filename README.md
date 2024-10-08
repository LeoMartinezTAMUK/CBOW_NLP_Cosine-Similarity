# Word Embedding using Continuous Bag of Words (CBOW) for Natural Language Processing (NLP)

**Author:** Leo Martinez III - [LinkedIn](https://www.linkedin.com/in/leo-martinez-iii/)

**Contact:** [leo.martinez@students.tamuk.edu](mailto:leo.martinez@students.tamuk.edu)

**Created:** Fall 2024

To clone this repository:

```
git clone https://github.com/LeoMartinezTAMUK/CBOW_NLP_Cosine-Similarity.git
```

---

This project implements Word Embedding using the Continuous Bag of Words (CBOW) method for natural language processing tasks. The program processes PDF files, tokenizes text, trains a Word2Vec model using CBOW, and evaluates the cosine similarity between selected word pairs from the document.

### Key Features:

- **Language**: Python 3.18
- **IDE**: Spyder
- **Libraries Used**:
  - `nltk`
  - `gensim`
  - `sklearn`
  - `PyPDF2`

### PDF File:

The input for this project is a research paper PDF file, approximately 6 pages long, which is processed to extract text. The content is available on IEEE Xplore.

### Implementation Details:

- **PDF Reading**: The `read_pdf()` method reads and extracts text from the specified PDF file.
- **Text Tokenization**: The `tokenize_text()` method tokenizes the extracted text into sentences and words, preparing it for Word2Vec model training.
- **Word2Vec Training**: The Word2Vec model is trained using CBOW (with `sg=0` parameter). It is configured with:
  - **Vector Size**: 100
  - **Window Size**: 5
  - **Minimum Word Count**: 1
  - **Workers**: 6 (to speed up training)
- **Cosine Similarity**: The `find_cosine_similarity()` method computes cosine similarities between pre-selected word pairs from the research paper.
  
### Word Pairs:

Here are the word pairs chosen from the research paper for cosine similarity analysis:

1. 'intrusion' and 'detection'
2. 'machine' and 'learning'
3. 'cyber' and 'threats'
4. 'network' and 'behavior'
5. 'deep' and 'neural'
6. 'anomaly' and 'detection'
7. 'signature-based' and 'detection'
8. 'behavior-based' and 'detection'
9. 'cyber' and 'defenses'
10. 'ctgan' and 'network'

### Evaluation:

The similarity between the word pairs is computed using cosine similarity, and the results are displayed. The pair with the highest similarity score is highlighted in the output. A screenshot of the output can be found labeled as "code_output.pmg"

### Running the Program:

1. Ensure that all required libraries are installed:
2. Download necessary NLTK resources:
nltk.download('punkt')
3. Specify the path to the research paper obtained from IEEE Xplore
pdf_file_path = "research_paper.pdf"

### Note:

- For further details and similar projects, please visit my [GitHub Page](https://github.com/LeoMartinezTAMUK).

Here is a brief explanation of the items:
- **src:** folder that contains the source code python script (.py)
- **README.md:** contains most basic information about the project
- **LICENSE:** contains license information in regards to the Github repository
- **code_output.png:** contains the output generated from the NLP model


