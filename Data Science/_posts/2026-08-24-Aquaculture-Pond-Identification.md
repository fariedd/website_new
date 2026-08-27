---
layout: post
title: GeoAI Aquaculture Pond Identification using Satellite data (Sentinel-1 + Sentinel-2 bands)
image:
  path: /assets/img/blog/ponds.jpg
  width: 800
  height: 600
description: >
  A machine learning solution for the Zindi GeoAI Aquaculture Pond Identification Challenge, detecting aquaculture ponds from Sentinel-1 radar and Sentinel-2 optical satellite data. My solution ranked 253/1314, generalising cleanly from a clean training period to a blind, cloud-affected test period.
tags: [Classification, Remote Sensing, Python, Machine Learning]
sitemap: true
hide_last_modified: true
---

**Failure is an option here. If things are not failing, you are not innovating enough** ~ *Elon Musk*.

**Complete Jupyter Notebook** - [![](https://img.shields.io/badge/GitHub-View_in_GitHub-blue?logo=GitHub)](https://github.com/fariedd/GeoAI-Aquaculture-Pond-Identification-Zindi-Competition/blob/main/spectra_bands_prediction_clean.ipynb){:target="_blank"}

* toc
{:toc}


## Introduction

Aquaculture ponds are used to farm fish and shrimp in controlled water environments. They are easier to manage and more productive than open-water fishing, but to monitor them effectively we first need to know where they are. Satellite imagery offers a reliable way to map these ponds across large areas, supporting water management, environmental monitoring, and policy decisions, especially in regions where ground-level data is hard to come by.

This project was my entry to the [Zindi GeoAI Aquaculture Pond Identification Challenge](https://zindi.world/competitions/geoai-aquaculture-pond-identification-challenge/leaderboard). The goal was to build a machine learning model that predicts whether a given location is an aquaculture pond or another type of land cover, using features derived from satellite data. Each location represents a **10m × 10m patch of ground**, so the model needs to be precise.

The challenge came with a deliberate challenge: the model is **trained on data from one time period and tested on data from a different one**. The solution has to stay accurate as conditions change across seasons and years, not just perform well on familiar data, but perform well on data with missing values. These constraint shaped every decision I made.


### How it was evaluated

The leaderboard used a weighted, multi-metric score:

- **F1-Score (60%)** — a balanced measure of precision and recall, important because aquaculture ponds are a small fraction of all locations.
- **ROC-AUC (40%)** — how well the model ranks pond locations above non-pond locations across all thresholds, giving a stable read regardless of class imbalance.

The model had to output two things: a **binary target** (pond or other) and a **probability** of being a pond. 
 **Setting a custom probability threshold was forbidden** — the binary target had to come from the default 0.5 cutoff. That rule matters: it means the model can't be tuned to game F1 by shifting the decision boundary; the probabilities themselves have to be well-placed around 0.5.


### The problem statement

Two properties of the data defined the whole problem:

- The **test set is blind** — no labels, and I only see my score through the leaderboard.
- The **training data is complete** — no missing values in the bands.
- The **test data has missing values**, especially in the Sentinel-2 optical bands, which are blocked by cloud cover. The Sentinel-1 radar bands are far less affected, because radar penetrates cloud.

Training on clean, complete 12-month time series, but I predict on messy data from a different time period. 


## Data & Approach

Each pixel is described by a full year of monthly observations across twelve bands — two radar (**VH, VV** from Sentinel-1) and ten optical (**blue, green, nir, nira, re1, re2, re3, red, swir1, swir2** from Sentinel-2). In the raw file, those months live side by side as wide columns (`VH_01`, `VH_02`, …, `VH_12`).

The Data:

<div style="overflow-x: auto; width: 100%; margin: 1em 0;">
<table style="border-collapse: collapse; font-size: 11px; white-space: nowrap;">
  <tr>
    <th></th><th>ID</th><th>label</th>
    <th>VH_01</th><th>VV_01</th><th>blue_01</th><th>green_01</th><th>nir_01</th><th>nira_01</th><th>re1_01</th><th>re2_01</th>
    <th>...</th>
    <th>blue_12</th><th>green_12</th><th>nir_12</th><th>nira_12</th><th>re1_12</th><th>re2_12</th><th>re3_12</th><th>red_12</th><th>swir1_12</th><th>swir2_12</th>
  </tr>
  <tr>
    <td>0</td><td>ID_TR_NEW_XVGKFMLNRJ</td><td>0</td>
    <td>-29.099645</td><td>-22.471573</td><td>1665</td><td>1719</td><td>1367</td><td>1270</td><td>1689</td><td>1416</td>
    <td>...</td>
    <td>1639</td><td>1826</td><td>1395</td><td>1384</td><td>1736</td><td>1449</td><td>1436</td><td>1626</td><td>1307</td><td>1216</td>
  </tr>
  <tr>
    <td>1</td><td>ID_TR_NEW_GP8KNSWVP6</td><td>0</td>
    <td>-19.470574</td><td>-10.752340</td><td>1579</td><td>1740</td><td>2245</td><td>2231</td><td>2060</td><td>2147</td>
    <td>...</td>
    <td>1490</td><td>1622</td><td>2090</td><td>2109</td><td>1924</td><td>1941</td><td>2020</td><td>1779</td><td>2298</td><td>2131</td>
  </tr>
  <tr>
    <td>2</td><td>ID_TR_NEW_87X3957MVS</td><td>1</td>
    <td>-20.964854</td><td>-8.792675</td><td>1850</td><td>2345</td><td>3664</td><td>3470</td><td>2935</td><td>3384</td>
    <td>...</td>
    <td>1438</td><td>1673</td><td>1335</td><td>1289</td><td>1588</td><td>1377</td><td>1307</td><td>1503</td><td>1237</td><td>1168</td>
  </tr>
</table>
<p style="font-size: 11px; color: #666; margin-top: 4px;">5 rows × 146 columns</p>
</div>


### Reshaping to long format

My first decision was to **collapse the months into a long format**, one row per pixel-month instead of one row per pixel. This does two things at once: it folds the seasonal signal into the training data (the model sees each month as its own observation), and it gives me a natural way to deal with missing months later — I can simply drop the rows that are missing, rather than imputing 240+ columns.

``` python
bands = ["VH", "VV", "blue", "green", "nir", "nira",
         "re1", "re2", "re3", "red", "swir1", "swir2"]

all_data = origin_data.reset_index()          # a unique id to pivot on
long_data = pd.wide_to_long(all_data, stubnames=bands,
                            i="index", j="month", sep="_", suffix=r"\d+")
long_data = long_data.reset_index()
```

The optical reflectance bands are stored as integers scaled by 10,000, I rescale them back to the 0–1 reflectance range before computing any index that is sensitive to scale.

``` python
optical = ["blue", "green", "nir", "nira", "re1", "re2", "re3", "red", "swir1", "swir2"]
long_data[optical] = long_data[optical] / 10000
```


### Feature engineering — teaching the model what water looks like

A pond is, reflective body of water. I engineered features that make that signature explicit rather than expecting the model to detect water from raw bands.

- **NDWI** *(green − nir)/(green + nir)* — the classic water index; water reflects green and absorbs NIR.
- **MNDWI** *(green − swir1)/(green + swir1)* — a modified version using SWIR, which separates water from built-up land more cleanly.
- **VH − VV** — the radar backscatter difference. Because these are in decibels (log space), the ratio becomes a **subtraction**, not a division. Smooth water surfaces scatter radar differently than rough land.
- **AWEI (two variants)** — the Automated Water Extraction Index, in no-shadow and shadow-suppressing forms, designed to push the water/land contrast even harder than NDWI.
- **EVI** — the Enhanced Vegetation Index, computed *after* rescaling to reflectance because its `+1` term assumes a 0–1 range.
- **Otsu water flags** — for each month I compute an Otsu threshold on the radar bands and flag pixels below it as "water". Otsu finds the natural split between two populations, and these radar histograms are bimodal (water vs land), so the threshold cleanly separates them.

``` python
long_data['ndwi']  = (long_data["green"] - long_data["nir"])   / (long_data["green"] + long_data["nir"])
long_data['mndwi'] = (long_data["green"] - long_data["swir1"]) / (long_data["green"] + long_data["swir1"])
long_data['vh_vv'] = long_data['VH'] - long_data['VV']          # dB → subtraction, not ratio

long_data["awei_nsh"] = 4*(long_data["green"] - long_data["swir1"]) - (0.25*long_data["nir"] + 2.75*long_data["swir2"])
long_data["awei_sh"]  = long_data["blue"] + 2.5*long_data["green"] - 1.5*(long_data["nir"] + long_data["swir1"]) - 0.25*long_data["swir2"]
long_data["evi"]      = 2.5 * (long_data["nir"] - long_data["red"]) / (long_data["nir"] + 6*long_data["red"] - 7.5*long_data["blue"] + 1)

# per-month Otsu threshold on radar → binary water flag
from skimage.filters import threshold_otsu
thr = long_data.groupby('month')['VH'].apply(lambda s: threshold_otsu(s.dropna().values))
long_data['vh_water'] = (long_data['VH'] < long_data['month'].astype(int).map(thr)).astype(int)
```


### Handling the missing months honestly

This is where the long format pays off. In the blind test set, cloudy months arrive as fully-NaN rows. Rather than **imputing** those months — fabricating optical values the satellite never saw, I simply **drop** them and predict only on the months that have real data.

``` python
cols = [c for c in test_final_data.columns if c not in ('index', 'month', 'ID')]
model_input = test_final_data.dropna(how="all", subset=cols)   # drop fully-cloudy months
```

The justification is that my model learned "what a pond looks like" from twelve-month data, and filling gaps with borrowed values produces test pixels whose feature distribution differs from anything in training. Dropping-and-aggregating keeps every prediction grounded in something the satellite actually measured.. 


### A radar-only model for cloud-obscured months

Some missing rows are a special case: the optical bands are missing but the radar bands are present, because radar sees through cloud. Throwing these away wastes real signal and precious data. So alongside the full model I trained a **radar-only model** on just the always-present features (`VH`, `VV`, `vh_vv`, and the radar water flags), and routed each row to the right model.

``` python
radar_cols = ['month', 'VH', 'VV', 'vh_vv', 'vv_water', 'vh_water']

# a row is "radar-only" when all optical bands are NaN but radar is present
is_radar_only = X[optical_cols].isna().all(axis=1) & X[radar_cols].notna().all(axis=1)

proba.loc[full.index] = model.predict_proba(full)[:, 1]                   # full model
proba.loc[ro.index]   = radar_model.predict_proba(ro[radar_cols])[:, 1]   # radar-only model
```

This means a cloudy month still contributes a prediction based on what radar *can* see, instead of being silently dropped.


### Preventing leakage in the split

Because in the long format a single pixel spans twelve rows(months), an ordinary random tain_test_split would scatter one pixel's months across both train and test and since a pixel's months are highly correlated, the model would effectively see the test pixels during training. I used a **grouped, stratified split** so every pixel lands wholly on one side.

``` python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(X_10, long_y, groups=long_data['ID']))

# verify no pixel appears on both sides
assert len(set(long_data.iloc[train_idx]['ID']) & set(long_data.iloc[test_idx]['ID'])) == 0
```


### Aggregating months to a per-pixel decision

The label is per pixel, but the model predicts per month. To collapse a pixel's monthly predictions into one, I take the **maximum** probability across its  available months (in blind test pixels  had -7 months available). 
The reasoning for taking max month probability is l ask "does this pixel look strongly like a pond in its best available month?" and it works whether a pixel has five months or twelve.

``` python
pixel = X_new.groupby('ID').agg(proba=('proba', 'max'), label=('label', 'first'))
pred  = (pixel['proba'] >= 0.5).astype(int)
```


## Models

I trained four classifiers, all evaluated per-pixel after max aggregation, and all on the same leakage-safe grouped split:

- **XGBoost** — gradient-boosted trees; scale-invariant, native NaN handling, my primary workhorse.
- **Gradient Boosting** (scikit-learn) — a regularised second boosting model for diversity.
- **SVM (SVC)** — inside a pipeline with `StandardScaler` and a cyclical sin/cos encoding of month, since a distance-based model needs scaled features and a circular notion of the calendar.
- **TabPFN** — a pretrained tabular transformer that needs no tuning and is well-suited to small tabular data like this.

The SVM pipeline is worth showing, because it's the one model where preprocessing genuinely matters:

``` python
def month_cyclical(x):
    x = x.astype(float)
    radians = 2 * np.pi * x / 12
    return np.hstack([np.sin(radians), np.cos(radians)])   # Dec and Jan sit next to each other

preprocessor = ColumnTransformer(
    transformers=[('scale', StandardScaler(), scale_cols),
                  ('month', FunctionTransformer(month_cyclical), ['month'])],
    remainder='passthrough'
)
model = make_pipeline(preprocessor, SVC(random_state=42, probability=True))
model.fit(X_train, y_train)
```


## Results

### Validation (per-pixel, on the held-out grouped split)

On clean twelve-month validation data, every model was excellent:

| Model | ROC-AUC | F1 |
|---|---|---|
| XGBoost | 0.9966 | 0.9416 |
| Gradient Boosting | 0.9945 | 0.9355 |
| SVC | 0.9971 | 0.9571 |
| TabPFN | 0.9995 | 0.9932 |

### Blind competition test (ragged, missing months data)

The blind test told the real story and the gap between validation and test is the whole lesson of this project:

| Model | ROC-AUC | F1 |
|---|---|---|
| XGBoost | 0.8725 | 0.8489 |
| Gradient Boosting | 0.8486 | 0.8418 |
| SVC | 0.8924 | 0.8454 |
| TabPFN | 0.8581 | 0.8537 |
| **Competition Winner (1st place)** | **0.9689** | **0.9301** |

My best models generalised from ~0.99 validation to ~0.87–0.89 on the blind set, because the grouped split and drop-don't-impute strategy meant my validation was never inflated by leakage or fabricated data. 
**My Final rank: 253 / 1314.**

### What I confirmed along the way

I also ran **wide-format models with imputed months** (including MICE / predictive-mean-matching) and they consistently scored *below* the long-format drop-and-aggregate approach. Those models dropped my accuracy from ~0.85 to ~0.75 because imputed months diluted every pixel with predicted-pond signals. That negative result is what gave me confidence the long+max design was the right call.


## Conclusion

The models generalised **well but not spectacularly**, and the reason is instructive. My feature engineering captured *where* water is (spectral water indices, radar backscatter, Otsu flags) far better than *when and how it cycle, the seasonality of ponds*. The  pond's **temporal signature**, the fill-and-drain rhythm across the year is exactly what a clean twelve-month training set expresses richly and a test set with missing months doesnt. Also the max approach is likely to be affected by alot of False positives, coz it pick one month which said water and without seasonality it affect hte overall accuracy.

Two directions would close the gap to the leaderboard leaders:

1. **Model the temporal signature in a form that survives missing months** — persistence (fraction of months a pixel reads as water), cycle amplitude, or a fitted seasonal harmonic — rather than fragile summary statistics that mean different things when computed from five months versus twelve.
2. **Model the missing values in the test set explicitly** as part of the signal, rather than only routing around them.

The competition rewarded honesty about the train/test mismatch, and my solution's clean generalisation reflects that. The remaining distance to the top is a temporal-modelling problem, not a preprocessing one a satisfying place to have landed, because it points clearly at what to build next.
