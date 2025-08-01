# Crop-Recommendation-System-Using-ML

📌 Project Overview
This project predicts the most suitable crop to grow based on soil and environmental parameters such as Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall. The goal is to assist farmers and agricultural professionals in making data-driven decisions to improve crop productivity and land usage efficiency.

🔄 Pipeline of the Project
1. Data Loading
Dataset: Crop_recommendation Dataset.csv (loaded using pandas)

Checked shape, data types, null values, duplicate entries

2. Exploratory Data Analysis (EDA)
Used Seaborn and Matplotlib to visualize:

Correlation heatmap

Distributions of features like N, P, K

3. Label Encoding
Manually mapped 22 crops (string labels) to numeric values using a custom dictionary.

4. Feature-Target Separation
Features (X): N, P, K, temperature, humidity, pH, rainfall

Target (y): Encoded crop label

5. Train-Test Split
Used train_test_split from Scikit-learn (80% train, 20% test)

6. Feature Scaling
MinMaxScaler followed by StandardScaler was applied sequentially to normalize the features.

7. Model Training and Evaluation
You trained 10 different ML classifiers:

Model
Logistic Regression
Gaussian Naive Bayes
Support Vector Classifier
K-Nearest Neighbors
Decision Tree Classifier
Extra Tree Classifier
Random Forest Classifier
Bagging Classifier
Gradient Boosting Classifier
AdaBoost Classifier

Each model was trained and tested on the dataset.

Accuracy was calculated for each.

8. Performance Metrics
For the selected best model (e.g., GaussianNB):

Calculated: accuracy, precision, recall, f1_score, and confusion_matrix

📚 Methodology Used
✅ Supervised Learning – Classification
Input: Numerical features (N, P, K, temp, humidity, etc.)

Output: Multiclass crop labels

✅ Model Comparison
Used loop to train and evaluate each classifier

Chose the best based on accuracy

✅ Scaling Technique
Dual scaling approach:

MinMaxScaler for normalization

StandardScaler for standardization

Helps improve model convergence and performance

🛠️ Tools & Libraries Used
Category	Tools Used
Programming	Python
Libraries	Pandas, NumPy, Matplotlib, Seaborn
ML Frameworks	Scikit-learn
Visualization	Seaborn, Matplotlib
Models Used	LogisticRegression, SVM, KNN, NaiveBayes, etc.
IDE	Jupyter Notebook

🌍 Real-World Impact
✅ Practical Use Case
Helps farmers or agricultural advisors make informed crop choices

Reduces dependence on traditional, less-efficient decision-making

✅ Efficiency & Resource Optimization
Suggests crops based on soil chemistry and climate conditions

Improves yield, reduces waste, and enhances sustainability

✅ Scalability
Can be embedded into:

Mobile apps for farmers

Government advisory systems

IoT-enabled precision agriculture platforms

🧠 What’s Unique in Your Project
✅ Comparison of 10 different models, not just one or two

✅ Sequential dual-scaling (MinMax + Standard) for better performance

✅ Complete pipeline from data analysis to prediction

✅ Clean, manual encoding of labels for controlled mapping

✅ Strong focus on model evaluation and metrics
