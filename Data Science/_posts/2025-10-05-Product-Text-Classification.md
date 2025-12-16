---
layout: post
title: PART 1 - Product text Classification (TF-IDF + Multinomial Naive Bayes)
image: 
  path: /assets/img/blog/product clasification/texto.jpg
  width: 800
  height: 600
description: >
  In this project, I  built a text classifcation model to predict categories of > 700 000 products item descriptions, leveraging the power of Multinomial Naive Bayes to handle large datasets. My model had an accuracy of 88%
tags: [Regression, Python, Machine Learning]
sitemap: true
hide_last_modified: true
---


**Failure is an option here. If things are not failing, you are not innovating enough** ~ *Elon Musk*.

**Complete Jupyter Notebook** - [![](https://img.shields.io/badge/GitHub-View_in_GitHub-blue?logo=GitHub)](https://github.com/fariedd/Product-text-calssification-and-Sales-Analysis/blob/main/Product%20Text%20Classification.ipynb){:target="_blank"}

* toc
{:toc}




## Introduction

In this project, I developed a deployable machine learning product classification model for a B2B e-commerce company struggling with incomplete category information in their product catalog. The objective was to enhance the accuracy and efficiency of sales product categorization, ultimately improving the user experience on their website. The task involved building a text classification model to predict missing category data based on product descriptions. A significant challenge was dealing with category data in multiple formats, some of which did not align with the <a href="https://support.google.com/merchants/answer/6324436?hl=en#Format&zippy=%2Capparel-products" target="_blank">Google product taxonomy format</a>. I started by analyzing the dataset to identify and extract rows that conformed to the Google taxonomy, applying natural language processing techniques to clean the text data and enrich the training set. The resulting model was trained to accurately classify new products into the correct categories, improving data consistency and streamlining product management.


### Data Overview

The key variables to categorise products will come from one table contain incosistantly categorised products that will be cleaned to created a final dataset for modelling.

<style>
  table {
    border-collapse: collapse;
    width: 100%; /* Ensures the table takes up the full width, avoiding extra space */
    font-size: 12px;
  }
  th, td {
    border: 1px solid black;
    padding: 5px;
    text-align: left;
  }
</style>

<table>
  <tr>
    <th><strong>Column Name</strong></th>
    <th><strong>Definition</strong></th>
  </tr>
  <tr>
    <td><strong>item_name</strong></td>
    <td><em>Product item description</em></td>
  </tr>
  <tr>
    <td><strong>original_format_category</strong></td>
    <td><em>Google Merchant Catalog value</em></td>
  </tr>
</table>

``` python
import pandas as pd
product_taxanomy = pd.read_csv('ProductDataChallenge_02_ProductTaxonomy.csv')

product_taxanomy.head(6)
```
<!-- -->

| item_name                                                             | original_format_category                                           |
|----------------------------------------------------------------------|--------------------------------------------------------------------|
| Women's Tune Squad Graphic Lounge Shorts - Yellow L                  | Apparel & Accessories > Clothing > Shorts                          |
| Nickelodeon Cra-Z-Sand Tri-Color Bucket of Sand                      | Toys & Games > Toys > Art & Drawing Toys > Play Dough & Putty      |
| Apollo Tools 53pc DT9773 Household Tool Kit with Tool Box Red        | Hardware > Tools > Tool Sets > Hand Tool Sets                      |
| Henckels Forged Classic Christopher Kimball 3pc Starter Knife Set    | Home & Garden > Kitchen & Dining > Kitchen Tools & Utensils > Kitchen Knives |
| BOBS from Skechers Blue Printed Dog Walking Kit Medium               | Animals & Pet Supplies > Pet Supplies > Dog Supplies               |
| ASICS Kid's PRE EXCITE 8 Pre-School Running Shoe, 1M, Pink           | Apparel & Accessories > Shoes                                      |


## Methods & Data Cleaning

The client wants to standadise product categorisation using google taxanomy format. Taxanomy data was downloaded <a href = 'https://support.google.com/merchants/answer/6324436?sjid=11384066497760467600-EU' target ='_blank' > here</a>
Firstly l filtered out products not in google taxanomy format and then remove duplicates.

``` python
# Clean product taxonomy file match it to google merchant catalogue 
google_catalogue = pd.read_csv('taxonomy-with-ids-template.csv')
# Combine columns into one with ">" separator, stopping at NaN
def combine_levels(row):
    combined = []
    for item in row[1:]:  # Skip the first column
        if pd.isna(item):
            break
        combined.append(str(item))
    return ' > '.join(combined)

google_catalogue['original_format_category'] = google_catalogue.apply(combine_levels, axis=1)
google_catalogue.rename(columns={"1": "id"}, inplace=True)

#combine with product_taxanomy and remove nans for mis classification
new_taxanomy =product_taxanomy.merge(google_catalogue[['id','original_format_category']], on='original_format_category', how='left')

#remove categories not in google catalogue format
new_taxanomy.dropna(subset=["id"],inplace=True)
```
l removed duplicate products because l assumed that each product belongs to one category, this could affect the prediction of unseen products

```python
# Check for dupllicates
duplicates_2 = new_taxanomy.groupby('item_name').size()
duplicates_2 =duplicates_2[duplicates_2 > 1].sort_values(ascending=False).reset_index()

##remove duplicated items
new_taxanomy = new_taxanomy[~new_taxanomy['item_name'].isin(duplicates_2['item_name'])]

#double check
zz =new_taxanomy.groupby('item_name').size()
zz[zz > 1].sort_values(ascending=False).reset_index()# double check
```
### **Text Normalisation**

l normalized text before inputting it into a predictive model to ensure consistency and remove noise. This helps the model focus on meaningful patterns, improving accuracy and performance. l preprocessed the data by removing non-alphanumeric characters, stripping whitespace, and converting text to lowercase. 

``` python
import requests
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove non-alphanumeric characters
    text =text.strip() #remove white space
    text = text.lower() # Convert text to lowercase
    return text


new_taxanomy['new_name'] = new_taxanomy['item_name'].apply(clean_text)
```

## Model Training & Pipeline

l created a two-level preprocessing and modeling pipeline, employing `TfidfVectorizer()` for transforming text data into term-frequency inverse document frequency (TF-IDF) vectors and `MultinomialNB()` for the Naive Bayes classification. Additionally, l utilized` GridSearchCV` to perform hyperparameter tuning for optimal model parameters. This pipeline ensured efficient text normalization and robust classification performance.

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a powerful technique in Natural Language Processing used to convert text data into a statistically interpratable numerical feature vector that reflects the significance of a word within a document relative to a collection of documents, known as a corpus. It assigns a numerical value to each word (*TF-IDF score*) that quantifies the importance of each term (*word*) in a document (*string of product item name description*) by considering its frequency in the document and its rarity across multiple documents. The will then be used by machine learning model to categorise the products and hten convert the numerica feature vector back into text 

### TF-IDF Score:
$$ \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t) $$

### Term Frequency (TF):

**Definition**: Measures the frequency of a term in a document.

**Formula**:

$$ \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d} $$

**Purpose**: Captures the importance of a term within a specific document.

### Inverse Document Frequency (IDF):

**Definition**: Measures the rarity of a term across all documents in the corpus.

**Formula**:

$$ \text{IDF}(t) = \log \left(\frac{\text{Total number of documents}}{\text{Number of documents containing term } t}\right) $$

**Purpose**: Assigns higher weights to terms that are rare across the corpus, reducing the impact of common terms.

Because of the size of the data...

``` python
#count number of unique values
print(f'Number of product categories: {new_taxanomy["target_label"].nunique()} ' ) 
print(f'Number of product items: {new_taxanomy["new_name"].nunique()} ' ) 

Number of product categories: 2871 
Number of product items: 577600 

```
l used a Multinomial Naive Bayes for product classification because it is both robust (it can handle multidimensional text data) and efficient (it can handle large data and is not computationally intensive). After applying TfidfVectorizer(), which converts text data into numerical TF-IDF vectors, the Multinomial Naive Bayes algorithm becomes more accurate in distinguishing between different product categories based on textual features.

### Model Implementation

``` python
#THIS CODE TOOK THREE HOURS TO RUN DO NOT RE-RUN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)## no need to sub-sample teh data
# Define the pipeline
pipeline_2 = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB(alpha=0.0001))
])

# Define the parameter grid for grid search
param_grid = {
    'tfidf__max_features': [200000, 350000, 500000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'nb__alpha': [0.0001, 0.001, 0.01, 0.1]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline_2, param_grid, cv=3, n_jobs=1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validated score: ", grid_search.best_score_)

# Evaluate on the test set
accuracy = grid_search.score(X_test, y_test)
print("Test set accuracy: ", accuracy)

```
<pre>
Best parameters found:  {'nb__alpha': 0.001, 'tfidf__max_features': 500000, 'tfidf__ngram_range': (1, 2)}
Best cross-validated score:  0.8641511636958045
Test set accuracy:  0.8747221573562354
</pre>

## Results
### Final Model

l employed the model with the highest cross-validation and test accuracy 

``` python
#Best model parameter
pipeline_2 = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.75, 
                              min_df=1, max_features=500000, 
                              ngram_range=(1, 2), stop_words='english', 
                               use_idf=True, 
                              smooth_idf=True, 
                              sublinear_tf=True)),
    ('nb', MultinomialNB(alpha=0.001))
])
```
### Evaluation and Prediction

To evaluate our model, l used precision, recall, F1-score, and accuracy. l also calculated macro and micro averages to better understand the model's performance with unbalanced data. **Macro averages** treat all classes equally, providing a balanced view across classes, while **micro averages** consider the frequency of each class, giving a more holistic measure of the overall performance.

``` python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Predict on the test set
y_pred_2 = pipeline_2.predict(X_test)

# Assuming y_test and y_pred_2 are already defined
accuracy = accuracy_score(y_test, y_pred_2)
precision_macro = precision_score(y_test, y_pred_2, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred_2, average='macro', zero_division=0)
f1_macro = f1_score(y_test, y_pred_2, average='macro', zero_division=0)
precision_micro = precision_score(y_test, y_pred_2, average='micro', zero_division=0)
recall_micro = recall_score(y_test, y_pred_2, average='micro', zero_division=0)
f1_micro = f1_score(y_test, y_pred_2, average='micro', zero_division=0)

# Create a dictionary of the metrics
metrics_data = {
    'Accuracy': [accuracy, accuracy],  # Accuracy is the same for both macro and micro
    'Precision': [precision_macro, precision_micro],
    'Recall': [recall_macro, recall_micro],
    'F1 Score': [f1_macro, f1_micro]
}

# Specify row labels
row_labels = ['Macro Avg', 'Micro Avg']

# Convert dictionary to a DataFrame
metrics_df = pd.DataFrame(metrics_data, index=row_labels)

# Display the table
print(metrics_df)

```
<pre>
           Accuracy  Precision  Recall   F1 Score
Macro Avg    0.88      0.62      0.54      0.56
Micro Avg    0.88      0.88      0.88      0.88
</pre>
 The low macro averages indicates that the model's performance is not consistent across all classes. It suggests that some classes may have lower precision, recall, or F1 scores, potentially due to class imbalance or the model struggling with specific categories.
 
 On the other end high micro average means that the model performs well overall when considering all classes and their frequencies. This reflects strong overall accuracy and suggests that the model correctly identifies a large proportion of instances, even if some individual classes are less well-predicted.



## PART 2

Due to high traffic during Black Friday week, the product classification model was used to categorize products. Sales data was analyzed to reveal valuable trends in buying patterns, such as popular products, peak shopping times, and discrepancies between regional sales.
``` python
import pandas as pd
product_sales = pd.read_csv('ProductDataChallenge_01_ProductSales_thanksgiving_week.csv')

# Predict categories for unlabeled data
product_sales['target_label'] = pipeline_2.predict(product_sales['item_name'])
```
An interaction Shiny dashboard was created showcasing Product sales KPIs, trends and business recommendations. For more information click  <a href = 'https://fariedakwa.netlify.app/blog/sales-trends-analysis/' target ='_blank' > here</a>