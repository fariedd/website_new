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

* toc
{:toc}

### Introduction

My client is a partnership B2B company looking to leverage data-driven decision-making tools to help businesses understand and capitalize on market trends and drive growth in product sales.
In this project, l analyzed sales for Thanksgiving and Black Friday week of 2021 from Monday 22 November –to Monday 29 November. 

The main objectives were:
1.	What product sales trends were observed during the week?
2.	What are the top-selling products and categories during Thanksgiving week?
3.	What are the highest commissioned products?
4.	Which products and product categories are most frequently purchased together?
5.	How were product sales trends similar or different between Black Friday and Cyber Monday?
6.	What business recommendations can you provide based on the above analysis?  
7.	How would the product classification model be deployed and productionalised to support client-facing product features?

To illustrate the power of sales data analysis l presented my findings in a live R shiny interactive dashboard highlighting the objectives on each tab.

**Click to view Sales Presentation** - [![](https://img.shields.io/badge/View-Dashboard-FF6C00?style=flat-square)](https://farisayi-dakwa.shinyapps.io/Presentation-Product-Items-Sales-Analysis/){:target="_blank"}

## Methods 

For this project, I utilized Python to summarize, wrangle data, and perform statistical analysis.**Click to view data wrangling and analysis in Python** - [![](https://img.shields.io/badge/Notebook-View_in_Github-blue?style=plastic&logo=Jupyter)](https://github.com/fariedd/Product-text-calssification-and-Sales-Analysis/blob/main/Sale%20Analysis%20notebook.ipynb){:target="_blank"}
l exported the data into R, I compiled and visualized the insights, presenting them as an organized interactive R shiny dashboard to enhance the user experience for stakeholders **Click to view R Shiny dashboard and visualisation script** - [![](https://img.shields.io/badge/R_shiny-View_in_Github-blue?style=plastic&logo=R)](https://github.com/fariedd/Product-text-calssification-and-Sales-Analysis/blob/main/app.R){:target="_blank"}

### Data Import

A Naïve Bayes text classifier was used to categorize product items using product descriptions, see more  <a href = 'https://fariedakwa.netlify.app/blog/product-text-classification/' target ='_blank' > here</a>

```python
#PYTHON CODE BLOCK
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

product_sales['target_label'] = pipeline_2.predict(product_sales['item_name'])# classify product sales using product description
```
### Data Overview

The key variables will come from one table containin product sales.

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
    <td><strong>action_id</strong></td>
    <td><em>Unique identifier for a transaction</em></td>
  </tr>
  <tr>
    <td><strong>sqldate</strong></td>
    <td><em>Date of transaction</em></td>
  </tr>
  <tr>
    <td><strong>item_name</strong></td>
    <td><em>Product item name</em></td>
  </tr>
  <tr>
    <td><strong>country</strong></td>
    <td><em>Country where the transaction took place</em></td>
  </tr>
  <tr>
    <td><strong>payout_type</strong></td>
    <td><em>Terms for commission</em></td>
  </tr>
  <tr>
    <td><strong>saleamt</strong></td>
    <td><em>Sale amount in USD</em></td>
  </tr>
  <tr>
    <td><strong>commission</strong></td>
    <td><em>Commission amount in USD</em></td>
  </tr>
   <tr>
    <td><strong>target_label</strong></td>
    <td><em>Predicted Google Merchant Catalog value</em></td>
  </tr>
</table>

``` python
#PYTHON CODE BLOCK
import pandas as pd
product_sales.head(6)
```
<!-- -->

<style>
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
    overflow-x: auto;
    display: block;
    white-space: nowrap; /* Prevents line breaks */
  }
  th, td {
    border: 1px solid black;
    padding: 5px;
    text-align: left;
  }
</style>

<table>
  <tr>
    <th>action_id</th>
    <th>sqldate</th>
    <th>item_name</th>
    <th>country</th>
    <th>payout_type</th>
    <th>saleamt</th>
    <th>commission</th>
    <th>target_label</th>
  </tr>
  <tr>
    <td>4270.5079.279362</td>
    <td>2021-11-27</td>
    <td>adidas Supernova Shoes Cloud White 11 Womens</td>
    <td>US</td>
    <td>PCT_OF_SALEAMOUNT</td>
    <td>65.0</td>
    <td>11.05</td>
    <td>Apparel & Accessories > Shoes</td>
  </tr>
  <tr>
    <td>4270.5080.265784</td>
    <td>2021-11-28</td>
    <td>adidas Adilette Comfort Slides Core Black 10 Mens</td>
    <td>US</td>
    <td>PCT_OF_SALEAMOUNT</td>
    <td>24.5</td>
    <td>1.47</td>
    <td>Apparel & Accessories > Shoes</td>
  </tr>
  <tr>
    <td>9800.5081.66990</td>
    <td>2021-11-29</td>
    <td>Pro Tools Ultimate Subscription ExpertPlus Support</td>
    <td>FR</td>
    <td>PCT_OF_SALEAMOUNT</td>
    <td>0.0</td>
    <td>0.00</td>
    <td>Software > Computer Software > Multimedia & Design Software > Music Composition Software</td>
  </tr>
  <tr>
    <td>10310.5077.1677940</td>
    <td>2021-11-26</td>
    <td>Loose Built-In Flex Jeans For Men</td>
    <td>CA</td>
    <td>PCT_OF_SALEAMOUNT</td>
    <td>35.4</td>
    <td>1.06</td>
    <td>Apparel & Accessories > Clothing > Pants</td>
  </tr>
  <tr>
    <td>4270.5081.1564705</td>
    <td>2021-11-29</td>
    <td>adidas Adissage Slides Core Black M 12 / W 13 Unisex</td>
    <td>US</td>
    <td>PCT_OF_SALEAMOUNT</td>
    <td>21.0</td>
    <td>2.52</td>
    <td>Apparel & Accessories > Shoes</td>
  </tr>
</table>

### Objective 1; Day to day sales trends

```python
#PYTHON CODE BLOCK
# since we are looking at a week of data the lowet granular value is day and for sales intepretation we need the name of day
#convert date time column
from datetime import datetime
product_sales_new['sqldate'] =pd.to_datetime(product_sales_new['sqldate'], format='%Y-%m-%d')
product_sales_new['day'] = product_sales_new['sqldate'].dt.strftime('%d')
product_sales_new['day_name'] = product_sales_new['sqldate'].dt.strftime("%A")#day name

pd.set_option('display.float_format', lambda x: '%.0f' % x)#supress scientific notation

day_to_day = product_sales_new.groupby(['sqldate','day','day_name'])['saleamt'].sum().reset_index()# day to day data

```
Visualise wrangled in R

```r
# R CODE BLOCK
plot_ly(day_to_day, 
        x = ~sqldate, 
        y = ~saleamt, 
        type = 'scatter', 
        mode = 'lines+markers', 
        hoverinfo = 'text',
        hovertemplate = '<b>%{x}</b>: %{y:.2s} USD<extra></extra>') %>%
  layout(
    hovermode = "x",
    xaxis = list(
      title = list(text = 'Thanksgiving Week November 2021', standoff = 20),
      tickformat = '%A %d %b',
      dtick = 'D1',
      showline = TRUE,
      linecolor = 'black'
    ),
    yaxis = list(
      title = list(text = 'Daily Sales Revenue (USD)', standoff = 20),
      showline = TRUE,
      linecolor = 'black'
    )
  )
```


<iframe src="\Data Science\plot.html" width="100%" height="400"></iframe>

**View more in dashboard** -  [![](https://img.shields.io/badge/View-Dashboard-FF6C00?style=flat-square)](https://farisayi-dakwa.shinyapps.io/Presentation-Product-Items-Sales-Analysis/){:target="_blank"}

**Objective 2, 3, and 5** required dats wrangling using Python and visulalisation and integration into siny dashboard in R as illustrated above.


### Objective 4; Product association

To determine frequently purchased products, I used a market basket association rule technique with support, confidence, and lift as metrics to quantify the strength of relationships between item sets.

- **Support**: The support of an item set is the probability of its occurrence out of the total number of transactions. It quantifies the share of transactions containing an item set. For item set A:
  

$$
  \text{support}(A) = \frac{n(A)}{n(\text{transactions})}
$$



- **Confidence**: For association \( A \rightarrow B \), confidence is the probability of A and B occurring together out of all transactions where A has already occurred:
  

$$
  \text{confidence} = \frac{n(A \cap B)}{n(A)} = P(B|A)
$$

Given the dataset's size (200,000 transactions and items), I created a sparse matrix to speed up processing and scaled the data to use the Apriori and FP-Growth algorithms, but the matrix size was too large. I opted to manually calculate support, confidence, and lift for item sets within the data. **View more in dashboard** -  [![](https://img.shields.io/badge/View-Dashboard-FF6C00?style=flat-square)](https://farisayi-dakwa.shinyapps.io/Presentation-Product-Items-Sales-Analysis/){:target="_blank"}

### Objective 6; Difference between Black Friday and Cyber Monday sales

Looking at graphs;
Cyber Monday had more commissioned items than Black Friday,
**Conclusion:** Cyber Monday is focused on online sales, which reach a larger audience with extended shopping time; where affiliates can be used to boost online traffic compared to Black Friday which focusses more on in-store shopping 

Sales almost doubled for both Black Friday and Cyber Monday, especially in the the United States but Cyber Monday had more sales,
**Conclusion 1:** Retailers often use Black Friday to attract a large crowd with in-store promotions. These promotions could be extended to Cyber Monday,offer exclusive online deals, sometimes with additional discounts which drive up sales
**Conclusion 2:** Canada conversely had less sales on Cyber monday than Black Friday, which could mean that customers still prefer traditional instore deals than online shopping

### Business Recommendations

**Black Friday and Cyber Monday Sales Spike:**

- Increase marketing and promotional sales on these days.
- Ensure that inventory levels are sufficient to meet increased demand, especially for US sales where customer spending almost doubles.
- For Cyber Monday, invest in enhancing the online shopping experience. Ensure that websites can handle increased traffic smoothly, which improves user experience and can increase the conversion rate.

**Top Selling Products and Categories:**

- **Theragun Elite Black, gaming consoles, and Jordan Shoes** are top sellers. To attract customers, push to feature these prominently in marketing campaigns.
- **Bowflex C6 Bike** and **Bowflex Cardio Machine Mat** are often bought together. We could offer complementary bundle discounts for these products and other similar products.

**Data Driven Decision:**

- Models are only as good as the data used. In this case, the `product_taxanomy` still needs more cleaning on the correct cataloguing of items before they can be trained on a model. This will improve the implementation of recommendation tools and product association rules technique to identify items frequently bought-together items and create bundle deals or cross-selling opportunities.


