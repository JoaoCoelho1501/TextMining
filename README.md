# Análise de Sentimentos em Textos sobre Saúde Mental

> Pipeline completo de Text Mining: pré-processamento avançado, TF-IDF, seleção de atributos, comparação de classificadores e fine-tuning do DistilBERT.

**Mestrado em Business Intelligence & Analytics (ISCAP)**, cadeira de Text Mining para Business Analytics, 2025/26

## O problema
Classificar o sentimento (positivo ou negativo) de publicações em redes sociais sobre saúde mental e perceber se um modelo de linguagem pré-treinado compensa face aos métodos clássicos.

## Abordagem
1. **EDA**: word clouds, palavras mais frequentes e sentimento por período do dia, idade e país.
2. **Pré-processamento**: NER, remoção de HTML e URLs, expansão de chatwords, emoticons, emojis e contrações, stopwords e lematização.
3. **Representação**: TF-IDF com seleção de atributos por Qui-Quadrado e oversampling. A divisão treino/teste é feita **antes** destes passos, para evitar *data leakage*.
4. **Modelos clássicos**: Logistic Regression, SVM, Naive Bayes, árvores e ensembles, com validação cruzada.
5. **LLM**: fine-tuning do **DistilBERT**, testado com e sem pré-processamento.
6. **Explicabilidade** com SHAP e aplicação a comentários novos.

## Resultados (conjunto de teste)
| Modelo | Accuracy | Precision | Recall | F1 | Kappa |
|---|---|---|---|---|---|
| **DistilBERT** (texto original) | **0,900** | **0,964** | 0,831 | **0,893** | **0,800** |
| DistilBERT (texto pré-processado) | 0,871 | 0,888 | **0,849** | 0,868 | 0,741 |
| Logistic Regression (TF-IDF) | 0,835 | 0,866 | 0,812 | 0,838 | 0,671 |

**Conclusão:** o DistilBERT é o melhor modelo e funciona melhor com o **texto original**. Um pré-processamento agressivo remove negações e intensidade emocional que o Transformer sabe aproveitar.

## Estrutura
```
notebooks/01_analise_sentimentos_final.ipynb   projeto completo, com resultados
notebooks/02_classificacao_relatorio.ipynb     versão anterior focada na classificação
```

## Stack
`pandas` · `NLTK` · `spaCy` · `scikit-learn` · `imbalanced-learn` · `Transformers (DistilBERT)` · `PyTorch` · `SHAP`

## Como correr
Os notebooks foram feitos no Google Colab, com os dados no Google Drive. Ajusta o caminho do dataset na secção de carregamento.

## Autor
João Coelho
