from fastapi import FastAPI
import pickle
import nltk
from nltk.corpus import stopwords

stopwords = set(stopwords.words("english"))

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

app = FastAPI()

def nlp(text: str):
    text = text.strip().lower()
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word not in stopwords]
    return " ".join(filtered_words)

@app.post("/api")
def api(text: str):
    try:
        text_clean = nlp(text)
        text_vectorized = vectorizer.transform([text_clean])
        pred = model.predict(text_vectorized)
        label = le.inverse_transform(pred)[0]

        return {
            "prediction": label
        }
    except Exception as e:
        print("Error:", e)
        return {"error": "Internal server error"}