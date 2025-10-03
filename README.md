# Contents:
  * [Click Fraud Detection](#click-fraud-detection)
  * [Iris Flower Segmentation](#iris-flower-segmentation)
  * [Movie Recommender System](#movie-recommender-system)

## Click Fraud Detection:
This project uses a large clickstream dataset (train.csv) where each record represents a click event with attributes such as IP, app, device, and timestamp. The main goal is to predict whether a click leads to an app download (is_attributed), which is a highly imbalanced binary classification problem.

I performed several feature engineering steps, including:

- Daily and hourly click counts per IP and per IP–App pair

- Distinct number of devices per IP (daily and hourly)

- Time till the next click per IP and per IP–App pair

After building these features, I trained and evaluated multiple machine learning models using pipelines with imputation and under-sampling to handle missing values and class imbalance. The models include:

- Decision Tree (best depth = 5)

- Multi-Layer Perceptron (MLP)

- Support Vector Machine (SVM) with RBF kernel

- K-Nearest Neighbors (KNN) (best k = 9)

Each model was tuned with GridSearchCV and evaluated using confusion matrices, precision, recall, and F1-score.

You can access the `.ipynb` file by downloading `Click Fraud Detection.ipynb`.

## Iris Flower Segmentation:
There is famous dataset that includes sepal length, sepal width, petal length and petal width of 150 iris flower that categorized in three different class:
  - Setosa
  - Versicolor
  - Virginica   
     
I implemented K-Means algorithm from scrach using python and used the algorithm to group the records to three clusters. You can access the `.ipynb` file by downloading `Iris Flower Segmentation.ipynb`
  
## Movie Recommender System:
Generally there are 2 methods of recommender system:
  - Content Based
  - User Based (collaborative)   
     
I decided to create a collaborative movie recommender system and downloaded [MovieLenz](https://grouplens.org/datasets/movielens/100k/) dataset and implement an algorithm to suggest 5 movies to a user. 
For doing this Project I must answer to these questions:
  - How much are the users near to our target user?
  - How much would target user rate if he watched each movie?   
     
You can access the `.ipynb` file by downloading `Movie Recommender.ipynb`     
