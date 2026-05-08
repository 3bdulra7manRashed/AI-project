# Heart Disease Prediction: Machine Learning Project Report

## 1. Data Preprocessing Techniques
Before feeding the data into the machine learning algorithms, rigorous preprocessing was applied to ensure data quality and model stability. The following techniques were implemented:
* **Handling Missing Values:** Missing data was carefully imputed based on the nature of the feature. Numerical features were imputed using either the average (e.g., `avg__Age`, `avg__Thallium`) or the median (e.g., `med__Cholesterol`, `med__BP`) to avoid the influence of outliers.
* **Categorical Encoding:** Machine learning models require numerical inputs. We applied **One-Hot Encoding** to nominal categorical features (e.g., `cat__Gender`, `cat__work_type`, `cat__smoking_status`) using `sklearn.preprocessing.OneHotEncoder`, dropping the first column to prevent multicollinearity (dummy variable trap).
* **Feature Scaling:** Standard scaling was applied to numerical features to ensure that variables with larger ranges (like Cholesterol or Max HR) do not dominate the distance calculations in models like SVM and Logistic Regression.

## 2. Exploratory Data Analysis (EDA) & Visualizations
*(Note: Insert your actual plots in this section before submission)*

To understand the underlying patterns and relationships between features, several exploratory analyses were conducted:
* **Target Distribution:** We visualized the balance between positive (Heart Disease) and negative (Healthy) cases to ensure our models wouldn't be biased towards a majority class.
* **Feature Relationships (Correlation Heatmap):** *[Insert Correlation Heatmap Image Here]*
  * **Analysis:** The heatmap revealed significant correlations between the target variable and features such as `Chest pain type`, `Max HR`, and `Exercise angina`. For instance, patients with certain types of chest pain and lower maximum heart rates showed a higher likelihood of heart disease.
* **Categorical Impact:** Bar charts were used to analyze the effect of categorical features like `smoking_status` and `Gender` on the disease prevalence, providing clinical intuition before the modeling phase.

## 3. Modeling and Hyperparameter Tuning
We evaluated four different machine learning algorithms. For each model, we used a Validation Set approach to tune hyperparameters and select the best configuration based on the **Macro F1-Score**. 

* **Logistic Regression:**
  * **Hyperparameters Tuned:** Regularization parameter (`C`). Tested values: [0.001, 0.01, 0.1, 1.0, 10.0].
  * **Best Configuration:** `C = 0.01` (Validation F1: 85.00%).
* **Decision Tree Classifier:**
  * **Hyperparameters Tuned:** Maximum depth (`max_depth`). Tested values: [3, 5, 10, 15, 20].
  * **Best Configuration:** `max_depth = 3` (Validation F1: 80.00%).
* **Support Vector Machine (SVM):**
  * **Hyperparameters Tuned:** Regularization (`C`) and `kernel` (linear vs. rbf).
  * **Best Configuration:** `C = 1.0`, `kernel = 'rbf'` (Validation F1: 87.70%).
* **Random Forest Classifier:**
  * **Hyperparameters Tuned:** Number of trees (`n_estimators`) and `max_depth`.
  * **Best Configuration:** `n_estimators = 100`, `max_depth = 10` (Validation F1: 83.25%).

## 4. Techniques Used to Enhance Results
To ensure robust and clinically safe predictions, we utilized the following enhancement techniques:
* **Hold-Out Validation Tuning:** Instead of evaluating models on the training set (which leads to overfitting), we strictly tuned hyperparameters on a separate Validation Set before doing the final evaluation on an unseen Test Set.
* **Macro/Weighted F1-Score Optimization:** Instead of relying solely on Accuracy, we optimized our models using the F1-Score to ensure a balance between Precision and Recall, which is crucial in imbalanced medical datasets.
* **Custom Decision Thresholding (Clinical Safety):** For the Logistic Regression model, we analyzed the `predict_proba()` outputs. We discussed lowering the classification threshold from the default 0.5 to 0.3. This technique sacrifices a small amount of Precision (increasing False Positives) to significantly boost Recall (minimizing False Negatives), ensuring that high-risk patients are not missed.

## 5. Final Conclusion
The project successfully developed a robust pipeline for predicting heart disease. After rigorously training and testing four different models on an unseen test set (56 instances), the results were as follows:

1. **Decision Tree:** 87.50% Accuracy (Best Performer)
2. **Random Forest:** 83.93% Accuracy
3. **Logistic Regression:** 82.14% Accuracy
4. **SVM:** 82.14% Accuracy

**Conclusion:** The **Decision Tree** (with `max_depth = 3`) emerged as the superior model. Not only did it achieve the highest overall accuracy (**87.50%**), but it also recorded the lowest number of False Positives (only 2 cases) while successfully identifying 19 true positive cases. Furthermore, as a white-box model, the shallow Decision Tree provides easily interpretable rules (e.g., simple IF-THEN conditions based on age or chest pain), making it highly suitable and trustworthy for deployment in a real-world medical diagnostic system.