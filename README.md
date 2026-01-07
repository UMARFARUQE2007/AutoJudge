## AutoJudge

**AutoJudge** is a machine learning tool designed to estimate the difficulty of competitive programming problems based on their textual descriptions. By analyzing problem statements from platforms like **Kattis**, this model predicts a numerical difficulty rating - (0-100), helping problem setters benchmark their questions and allowing users to find problems that match their skill level.
<br>
<br>
Unlike other methods which rely on user submission data (acceptance rates), this project leverages **Natural Language Processing (NLP)** and advanced feature engineering to predict difficulty *before* a problem is even solved by a large audience.
Expand
message.txt
5 KB
﻿
## AutoJudge

**AutoJudge** is a machine learning tool designed to estimate the difficulty of competitive programming problems based on their textual descriptions. By analyzing problem statements from platforms like **Kattis**, this model predicts a numerical difficulty rating - (0-100), helping problem setters benchmark their questions and allowing users to find problems that match their skill level.
<br>
<br>
Unlike other methods which rely on user submission data (acceptance rates), this project leverages **Natural Language Processing (NLP)** and advanced feature engineering to predict difficulty *before* a problem is even solved by a large audience.

## Dataset Used

### Why Quality Data Matters
In Machine Learning, the principle of *"Garbage In, Garbage Out"* is paramount. For a nuanced task like estimating competitive programming difficulty, raw text alone is often insufficient. A high-quality dataset is required to capture:
* **Mathematical Symbolism:** Metadata like constraints (e.g., $N \le 10^5$ vs $N \le 18$) and mathematical density provide strong hints about the algorithmic complexity required.
* **Rating Consistency:** Reliable ratings assigned by experts in this field (e.g., Kattis ratings) are essential for training a regression model to be accurate.

### Dataset Composition
The model was trained on a comprehensive dataset scraped from **Kattis**, sourced from the [TaskComplexityEval-24 Repository](https://github.com/AREEG94FAHAD/TaskComplexityEval-24).

## Approach & Models

This project tackles two distinct challenges in competitive programming analysis: **Classification** (predicting the problem type) and **Regression** (predicting the precise difficulty score).

### Feature Engineering
Both models utilize a hybrid feature extraction strategy. I found that raw text alone was insufficient, so we engineered specific signals to capture the essence of a problem:
* **TF-IDF Vectors:** Captures the importance of important keywords as if they appear less more weightage to it is given.
* **Manual Features:**
    * `math_density` & `math_count`: features for formal mathematical complexity.
    * `text_len`: Measures the verbosity of the problem statement.
    * `hard_topic_count`: flags the presence of advanced terminology in competitive programming (e.g., *FFT, Mobius,Segment Tree, Polygon, Convex Hull*).
    * `high_difficulty_signal_count`: using probability for filter out words appearing more frequently in hard problems
    * `medium_signal_count`:  using probability for filter out words appearing more frequently in medium problems
    * `is_short_statement`: this feature might be debatable as there are many hard problems with short statement, but on testing inclusion of it showed a accuracy boost

---

### Classification Task (Predicting Difficulty Tiers)
**Aim:** Classify problems into broad difficulty classes: **Easy, Medium, and Hard**.
* **The Experiment:** I initially employed a standard **Grid Search + TruncatedSVD** pipeline to automate hyperparameter tuning.
* **The Final Model: Manual Tuning** 
    * Surprisingly, manual experimentation outperformed the automated grid search. By intuitively adjusting parameters and focusing heavily on feature selection, the manually tuned model achieved higher accuracy.
    * The manual approach allowed for better handling of the subtle boundaries between "Medium" and "Hard" tiers, which the automated grid search struggled to distinguish effectively.

---

### Regression Task (Predicting Difficulty Score)
**Goal:** Predict the exact numerical difficulty rating on a scale of 1-100.

* **The Experiment:** We tested Random Forest baselines and standard Gradient Boosting.
* **The Final Model: LightGBM + Optuna** 
    * For the regression task, precision was key. I utilized **LightGBM** for its speed and efficiency with high-dimensional data.
    * Instead of manual tuning, I deployed **Optuna**—an advanced hyperparameter optimization framework. Optuna successfully squeezed out the best performance by fine-tuning sensitive parameters like `learning_rate`, `num_leaves`, and `lambda_l1/l2` regularization, achieving the lowest RMSE and relatively higher R2 score.
#  Model Evaluation Report

## 1. Classification Models (Difficulty Labeling)
We compared a manual Random Forest approach against a Grid Search tuned model.
| Model Strategy | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :--- | :--- | :--- | :--- |
| **Manual Parameters** | **0.53** | **0.51** | **0.53** | **0.51** |
| **Grid Search (Best)**| 0.50 | 0.47 | 0.50 | 0.45 |

<br>

## 2. Regression Models (Difficulty Score 0-100)
We evaluated three different regression strategies. The **Gradient Boosting + Optuna** approach yielded the best performance, achieving the lowest error (RMSE) and highest explained variance (R²).

| Model Architecture | RMSE (Root Mean Sq. Error) | MAE (Mean Abs. Error) | R² Score |
| :--- | :--- | :--- | :--- |
| **Regression Tree (GridSearch)** | 20.05 | 16.99 | 0.1310 |
| **Gradient Boosting (Standard)** | 20.15 | 16.55 | 0.1227 |
| **Gradient Boosting + Optuna** | **19.66** | **16.34** | **0.1641** |

### Key Observations
* **Optuna Improvement:** Tuning with Optuna improved the R² score from **0.12** (Standard) to **0.16**, capturing more complexity in the problem statements.
* **Error Reduction:** The Optuna model reduced the RMSE to **19.66**, making its predictions closer to the actual difficulty score than the other models.


## How to Run Locally?

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites
Ensure you have the following installed:
* **Python 3.9+** (Python 3.13 recommended)
* **pip** (Python package installer)

### 2. Clone the Repository
Open your terminal (or Command Prompt) and run:

```bash
git clone [https://github.com/UMARFARUQE2007/AutoJudge.git](https://github.com/UMARFARUQE2007/AutoJudge.git)
cd AutoJudge
```
Recommend to run in virtual env
<br>
Run following to get all the modules installed
``` bash
pip install -r need.txt
```
Note for Mac (M1/M2/M3) Users: If you encounter errors with LightGBM, you may need to install OpenMP via Homebrew: brew install libomp

###3. Start the Flash server
```bash
python app.py
```
You should see something like this:
* Serving Flask app 'app'
* Debug mode: on
* Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)

Open your web browser and navigate to: http://127.0.0.1:5000
<br>
Enter a problem statement, input description, and output description to see the predicted difficulty and score. Enjoy :)


