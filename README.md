# Sentiment Analysis of Social Media Text about Mental Health

A Natural Language Processing (NLP) project comparing traditional machine-learning approaches with a fine-tuned Transformer model for binary sentiment classification.

**Academic context:** MSc in Business Intelligence & Analytics, ISCAP (2025/26)  
**Author:** João Coelho

## Project overview

The project investigates whether a pretrained language model can outperform classical text-classification methods when analysing social media posts related to mental health.

The task is framed as binary sentiment classification (positive vs. negative). The work combines exploratory analysis, text preparation, feature engineering, model comparison, and explainability.

## Methodology

1. **Exploratory analysis:** inspect class distribution, frequent terms, and patterns across selected metadata.
2. **Text preparation:** process HTML, URLs, chat expressions, emoticons, emojis, contractions, stopwords, and lemmatization.
3. **Classical representation:** TF-IDF with Chi-Square feature selection.
4. **Classical models:** Logistic Regression, Support Vector Machines, Naive Bayes, tree-based models, and ensemble methods.
5. **Transformer approach:** fine-tune DistilBERT using both original and preprocessed text.
6. **Explainability:** use SHAP-based analysis to investigate model behaviour and inspect predictions on new examples.

The train/test split is performed before fitting data-dependent preprocessing and resampling steps to reduce the risk of data leakage.

## Selected test results

| Model | Accuracy | Precision | Recall | F1 | Cohen's Kappa |
|---|---:|---:|---:|---:|---:|
| **DistilBERT — original text** | **0.900** | **0.964** | 0.831 | **0.893** | **0.800** |
| DistilBERT — preprocessed text | 0.871 | 0.888 | **0.849** | 0.868 | 0.741 |
| Logistic Regression — TF-IDF | 0.835 | 0.866 | 0.812 | 0.838 | 0.671 |

The best overall F1 and accuracy were achieved by DistilBERT trained on the original text. In this experiment, aggressive text normalization did not improve the Transformer results, suggesting that contextual and expressive cues in the original text can be useful.

## Technology stack

- Python, pandas, NumPy
- NLTK and spaCy
- scikit-learn and imbalanced-learn
- Hugging Face Transformers, DistilBERT, and PyTorch
- SHAP for explainability
- Google Colab

## Repository structure

```text
.
├── notebooks/
│   ├── 01_analise_sentimentos_final.ipynb
│   └── 02_classificacao_relatorio.ipynb
└── docs/
    └── project presentation
```

## Reproducing the project

The notebooks were developed in Google Colab and expect the dataset to be available separately.

1. Open the final notebook in Google Colab.
2. Upload or mount the dataset in your Google Drive.
3. Update the dataset path in the data-loading section.
4. Install the required Python packages if they are not already available.
5. Run the notebook from top to bottom.

## Scope and limitations

- The task is a binary classification exercise; it does not provide clinical assessment or mental-health diagnosis.
- Results depend on the dataset, label quality, class distribution, and train/test split used in the experiment.
- Performance on this dataset should not be assumed to generalize to other platforms, languages, or populations.
- Predictions about sensitive topics require careful interpretation and should not be used as a substitute for qualified support.
