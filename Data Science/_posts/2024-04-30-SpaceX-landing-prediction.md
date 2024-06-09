---
layout: post
title: Predict SpaceX rocket landing success using Machine Learning
image: 
  path: /assets/img/blog/predict-spaceX-landind/spacex_p.jpg
  width: 800
  height: 600
description: >
  In this project l built and trained four machine learning models which can predict if the first stage landing of SpaceX's falcon 9 rockets are successful with a combine accuracy of ~80%.
tags: [Python, Machine Learning, Web Scraping,API]
sitemap: true
hide_last_modified: true
---

[![](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/ibiene-ds/image-search-engine/blob/master/101_cnn_image_search_engine.ipynb)
[![](https://img.shields.io/badge/GitHub-View_in_GitHub-blue?logo=GitHub)](https://github.com/ibiene-ds/image-search-engine)

**Failure is an option here. If things are not failing, you are not innovating enough** ~ *Elon Musk*.
* toc
{:toc}


## Project Overview
### Problem Statement 

Going to space has always been a very expensive venture and SpaceX advertises Falcon 9 rocket launches on its website with a cost of 62 million dollars while other providers cost upward of 165 million dollars each, much of the savings is because since 2014 Space X has manage to successfully recover first stage rocket launched vehicles. This has made Space X rockets comparatively cheaper because their rockets are reusable. Although the landing success rate has improved over the years, not all rocket have landed successfully. Space X has been collecting data on its landing activities over the years, therefore if we can determine if the first stage will land, we can determine the cost of a launch. This information can be used if an alternate company wants to bid against SpaceX for a rocket launch.
 
### Methods
We will be predicting the landing success of Falcon 9 rocket launches and Falcon heavy launches from 2010 to 2021.
Data was collected by webscraping (*Beautifulsoup*) tables of SpaceX historical launch records from a [Wikipedia](https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922) page. [![](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/fariedd/IBM-DataScience-SpaceX-Capestone/blob/main/jupyter-labs-webscraping%20(1).ipynb).

More data on rocket landing outcomes were extracted using SpaceX resting APIs [![](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/fariedd/IBM-DataScience-SpaceX-Capestone/blob/main/jupyter-labs-spacex-data-collection-api_.ipynb).

Exploratory data analysis and preparatory feature engineering were implemented on the collected data to also visiualise landing outcome of rocket launches using Pandas and Matplotlib.Machine learning models typically cannot process string variables, so when there are text data within a data set, we would encode them (change them to numbers, typically binary) so that the algorithm can assess the relationship between the variable and the output. We used One Hot Encoding (OHE) from the sklearn.preprocessing library.
 [![](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/fariedd/IBM-DataScience-SpaceX-Capestone/blob/main/jupyter-labs-eda-dataviz%20(1).ipynb).

The detailed landing outcomes of the rockets were reclassified to 1 (landed) and 0 (did not land)
 for analysis, missing values and outliers were removed {add data wrangling notebook}.

We then implemented four classification machine learning models, Logistic regression, Support vector machine, Decision tree and K-nearest neighbor. Before modelling all variables were standardised using a StandardScaler so that variables measured at different scales may be reduced to the same scale and contribute equally to model fitting, with landing outcome as the dependant variable. 

The hyperparameters for each model were optimised using GridSearchCV, it evaluates the model for each combination of hyperparameters using the cross-validation method (cv =10) and chooses the combination with the best performance for each model before fitting.

The data was split into training and testing data, and a confusion matrix was created to show how the trained models can accurately predict the testing data (show false positve vs false negative).  [![](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/fariedd/IBM-DataScience-SpaceX-Capestone/blob/main/SpaceX_Machine_Learning_Prediction_Part_5.jupyterlite%20(1).ipynb).

#### Model Assesment

Model perfomance and comparison were determined using accuracy score, precision score, recall score and F1 score.

* **Accuracy**: This metric meassures how a machine learning model correctly predicts the outcome values in the test data (new data) by dividing the number of correct predictions by the total number of predictions. The accuracy score is measured on a scale of 0 to 1 or as a percentage. The higher the accuracy, the better.

* **Precision**: This metric measures how a machine learning model correctly predicts the positive outcome values (true postives) in a test data (in this case if the rocket landed successfully).  Answers the question: of all observations predicted as positive, what proportion was actually positive?. The accuracy score is measured on a scale of 0 to 1 or as a percentage. The higher the accuracy, the better.

* **Recall**: This metric measures how often a machine learning model correctly identifies positive instances (true positives) from all the actual positive samples in a test data (in this case if the rocket landed successfully). Answers the question: of all positive observations, how many did we predict as positive? The accuracy score is measured on a scale of 0 to 1 or as a percentage. The higher the accuracy, the better.

* **F1 Score**: This is the harmonic mean of Precision and Recall. A good F1-score comes when there is a balance between precision and recall.


### Results 

The goals of the project was:

* Build a model that would accurately predict the landing outcome of rockets (success/fail) . If we can determine if the first stage  will land, we can determine the cost of a launch.
* This information can be used if an alternate company wants to bid against SpaceX for a rocket launch.

The landing success of Space X rockets has continued to increase since 2013.
<br>
![](/assets/img/blog/predict-spaceX-landind/image.png)

Trend in landing success rate of Space X Falcon 9 rockets since 2010, success rate has had a positive upward trend since 2013.
{:.figcaption}
<br>

#### Models Perfomance

Based upon these,all models cna predict the landing success with an accuracy of ~80 acros all matrics excet for Decision Trees which perfomed lower but not singifcantly so


| Model                  | Accuracy   | Precision  | Recall       |  F1-Score  | 
|:------                 |:-----------|:-----------|:-----------  |:-----------|
| Logistic Regression (Log_Reg)   |  0.83      |   0.80	   |   1.0        |  0.89      | 
| Support Vector Machine (SVM) |  0.83      |   0.80	   |   1.0        |  0.89      |
| Decision tree           |  0.78      |  0.75	     |   1.0        |  0.85      |
| K-Nearest Neighbor (KNN)    |  0.83      |   0.80	   |   1.0        |  0.89      |

<br>
![](/assets/img/blog/predict-spaceX-landind/accuracy.png)

Model prediction accuracy for all built classification models, in a bar chart.
{:.figcaption}
<br>

**Confusion Matrix**



### Discussion, Growth & Next Steps

The way we have coded this up is very much for the "proof of concept".  In practice, we would definitely have to use a large data set and incooperate more variables into the model that could improve the predictions. Payloadmass of the rockets was an influential contributer to hte predictions of this model, (in practice we will have to remove and add some variables to see if htey affect the data). In this case removing payloadmass had a significant effect on the predictions, which hints at other factors being added to hte rocket influencing landing success of rockets. Alose inplace of Decision trees we can incooperate a random forest which offer a more complex visualisation option (nodes of classification).
 the last section of the code (where we submit a search) isolated, and running from all of the saved objects that we need - we wouldn't include it in a single script like we have here.

 This was tested only in two  categories (success or fail), we would want to test on a broader array of categories - most likely having a saved network for more specific conditions that describe  and categorise failure and successful landing of rockets in more detail. 

