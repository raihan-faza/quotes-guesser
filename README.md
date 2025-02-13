# Quote Guesser

## 📌 Overview
This is a simple NLP-based text classification project that predicts whether a given quote is from a **Motivational Speaker** or a **Fictional Character**. The model is trained using Scikit-learn and exposed via a FastAPI endpoint.

## 🚀 Features
- Classifies quotes into "Motivational" or "Fictional"
- Uses **TF-IDF Vectorization** for text processing
- Implements a simple **Naive Bayes Classifier**
- Provides a **FastAPI** endpoint for real-time predictions

## 🛠️ Tech Stack
- **Python 3.x**
- **FastAPI** (for API)
- **Scikit-learn** (for ML model)
- **NLTK** (for text preprocessing)

## 📂 Dataset
This project uses a dataset from Kaggle:
[Quotes 500K Dataset](https://www.kaggle.com/datasets/manann/quotes-500k?select=quotes.csv)

## 📂 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/who-said-it.git
cd who-said-it
```

### 2️⃣ Create a Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```


## 📡 API Usage
### **POST /predict**
**Request:**
```json
{
    "quote": "Do or do not, there is no try."
}
```

**Response:**
```json
{
    "category": "Fictional"
}
```

## 📝 To-Do
- Improve dataset size
- Implement a more advanced ML model (e.g., Transformer-based)
- Deploy API using **Docker**

---
{project postponed}
