# Install missing packages (if needed)
!pip install -q wordcloud nltk scikit-learn pandas matplotlib seaborn 

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Suppress warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')

# Download necessary NLTK data only once
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
# Download the 'punkt_tab' resource
nltk.download('punkt_tab') # This line was added to download the punkt_tab data

# Load dataset (Update file path if necessary)
from google.colab import files
uploaded = files.upload()  # Upload your dataset (UpdatedResumeDataSet.csv)
df = pd.read_csv('UpdatedResumeDataSet.csv')

print(df.head())

# Data Preprocessing Function
def clean_text(text):
    text = re.sub('http\S+\s*', ' ', text)  # Remove URLs
    text = re.sub('RT|cc', ' ', text)  # Remove RT and cc
    text = re.sub('#\S+', '', text)  # Remove hashtags
    text = re.sub('@\S+', ' ', text)  # Remove mentions
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # Remove punctuation
    text = re.sub(r'[^\x00-\x7f]', r' ', text)  # Remove non-ASCII characters
    text = re.sub('\s+', ' ', text).strip()  # Remove extra whitespace
    return text.lower()

df['cleaned_text'] = df['Resume'].apply(clean_text)

# Tokenization, Stopword Removal, and Lemmatization
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

df['processed_text'] = df['cleaned_text'].apply(preprocess_text)

# Encode Categories
label_encoder = LabelEncoder()
df['Category_Label'] = label_encoder.fit_transform(df['Category'])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X = vectorizer.fit_transform(df['processed_text'])
y = df['Category_Label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Performance Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# WordCloud Visualization
text_data = ' '.join(df['processed_text'])
wordcloud = WordCloud(background_color='black', max_words=200, width=1400, height=1200).generate(text_data)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Resume WordCloud')
plt.show()