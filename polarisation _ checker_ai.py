import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')

# Load source documents
def load_sources(folder_path):
    texts = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
                filenames.append(filename)
    return texts, filenames

# Compare student submission
def check_plagiarism(student_text, source_texts, source_filenames):
    documents = source_texts + [student_text]
    tfidf = TfidfVectorizer().fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf[-1], tfidf[:-1])
    
    print("\n🔍 Plagiarism Report:")
    for idx, score in enumerate(cosine_similarities[0]):
        print(f"Compared with {source_filenames[idx]}: Similarity Score = {score:.2f}")
        if score > 0.5:
            print("⚠️ Potential plagiarism detected!\n")

# Main
if __name__ == "__main__":
    source_texts, source_filenames = load_sources("sources")

    student_input = input("Paste the student's submission here:\n")
    check_plagiarism(student_input, source_texts, source_filenames)
