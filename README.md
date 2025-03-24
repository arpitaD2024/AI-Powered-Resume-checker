# AI-Powered-Resume-checker

## Overview
This project is an AI-powered Resume Classifier that categorizes resumes into different job sectors using Natural Language Processing (NLP) and Machine Learning (ML). It allows users to upload resumes (PDF/DOCX) and predicts the best job category for them.

## Features
- Supports PDF & DOCX Resume Upload  
- Preprocesses Resumes (Cleaning, Tokenization, Stopword Removal, Lemmatization)  
- Uses TF-IDF Vectorization for feature extraction  
- Trains Naïve Bayes, Random Forest, and SVM models  
- Evaluates Accuracy, Confusion Matrix, and Classification Report  
- Visualizes Data with WordCloud and Confusion Matrix  

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/resume-classification.git
cd resume-classification
pip install -r requirements.txt
```

## Usage
### Train the Model
Run the script to train and evaluate the model:
```bash
python main.py
```
### Upload and Classify a Resume
Place the resume file (PDF/DOCX) inside the `uploads/` folder and run:
```bash
python main.py --predict uploads/sample_resume.pdf
```

## Repository Structure
```
resume-classification/
│── data/                # Dataset folder
│   ├── UpdatedResumeDataSet.csv  # Training dataset
│── uploads/             # Folder for user-uploaded resumes
│── main.py              # Main script for training and prediction
│── utils.py             # Helper functions (text extraction, preprocessing)
│── requirements.txt     # Required dependencies
│── README.md            # Project documentation
```

## Dependencies
Ensure you have the following installed:
- Python 3.8+
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn, NLTK, WordCloud
- PyPDF2, python-docx

Install via:
```bash
pip install -r requirements.txt
```

## Future Enhancements
- Improve accuracy with Deep Learning (BERT, Transformers).
- Deploy a Web Application for user-friendly access.
- Implement Bias Mitigation for fair job recommendations.

## License
This project is open-source and available under the MIT License.

