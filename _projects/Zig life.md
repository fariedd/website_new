---
layout: post
title: Zig life
caption: This is what shows outside
description: >
  this is what shows inside
date: 5 June 2024
image: 
  path: /assets/img/blog/predict-cust-loyalty/regression-title-img.png
sitemap: true
---

[![](https://img.shields.io/badge/Read_Full_Analysis-blue?)](/blog/predicting-customer-loyalty/)
[![](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](/project-files/enhance-target-accuracy-ml.html)
[![](https://img.shields.io/badge/GitHub-View_in_GitHub-blue?logo=GitHub)](https://github.com/ibiene-ds/enhancing-targeting-accuracy)

A grocery retailer hired a  market research company to append **customer loyalty**information to their customer database. Unfortunately, only about half of the database could be tagged, thus the other half did not have this information present.  The goal of this project then is to build a model that  **predicts the loyalty score of the other half the customers**. 

The raw data had less than 1% of missing values so instead of imputation, missing values were excluded. Other preprocessing steps used include excluding outliers using the IQR method, and one-hot encoding categorical variables. Instead of using all the inputs in the model, *Recursive Feature Elimination using Cross Validation (RFECV)* was applied to infer the best set of input variables to use in the regression models. Three regression models were tested: *Random Forest, Decision Tree and Linear Regression*. Of all three models, **the Random Forest model gave the best R-squared and Adjusted R-squared values of 0.955 and 0.925 respectively**.

