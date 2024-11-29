---
layout: post
title: PART 2 - Product Sales Trend Analysis for Black Friday week 
image: 
  path: /assets/img/blog/product clasification/markus-spiske-sales.jpg
  width: 800
  height: 600
description: >
  In this project, I build a interactive R Shiny Dashboard to present Sales Analysis Trends of Thanks giving week of 2021, Insights and Business Recommendation 
tags: [Dashboard ,Data Visualisation, R , Trend Analysis]
sitemap: true
hide_last_modified: true
---


**People don’t buy what you do; they buy why you do it.** ~ *Simon Sinek*

**R Shiny App Script** - [![](https://img.shields.io/badge/R_shiny-View_in_Github-blue?style=plastic&logo=R)](https://github.com/fariedd/Product-text-calssification-and-Sales-Analysis/blob/main/app.R){:target="_blank"}

**Data Analysis Python Notebook** - [![](https://img.shields.io/badge/Notebook-View_in_Github-blue?style=plastic&logo=Jupyter)](https://github.com/fariedd/Product-text-calssification-and-Sales-Analysis/blob/main/Sale%20Analysis%20notebook.ipynb){:target="_blank"}

### Introduction

**FLEC** is a partnership management company that leverages advanced tools to help businesses understand and capitalize on market trends.

In today's competitive market, data-driven decision-making is crucial for achieving sales success. Analyzing sales data, provides valuable insights that empower businesses to optimize their strategies and drive growth.

For this project, we focused on product sales data for Thanksgiving and Black Friday. By uncovering key trends, identifying areas for improvement, and making informed decisions, we aim to enhance overall performance.

To illustrate the power of **FLEC** sales data analysis, I am excited to present our latest dashboard, which highlights essential metrics and actionable insights.

**Complete Sales Presentation** - [![](https://img.shields.io/badge/View-Dashboard-FF6C00?style=flat-square)](https://farisayi-dakwa.shinyapps.io/Presentation-Product-Items-Sales-Analysis/){:target="_blank"}

## Methods 

A product text classification model was used to categorise product items, see more <a href = 'https://fariedakwa.netlify.app/blog/product-text-classification/' target ='_blank' > here</a>

```python
# Predict categories for unlabeled data
# Create a pipeline with TF-IDF vectorizer and Naive bayes hadnle textual data and imblanced datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)## no need to sub-sample teh data

pipeline_2 = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.50, 
                              min_df=1, max_features=500000, 
                              ngram_range=(1, 2), 
                              stop_words='english', 
                               use_idf=True, 
                              smooth_idf=True, 
                              sublinear_tf=True)),   # Adjust max_features based on your dataset size, ngrams captures bi-words
    #('svd', TruncatedSVD(n_components=1000)), # Adjust n_components as needed
   ('nb', MultinomialNB(alpha=0.001))  # Naive Bayes classifier
])

# Train the model
pipeline_2.fit(X_train, y_train)

product_sales['target_label'] = pipeline_2.predict(product_sales['item_name'])
```

