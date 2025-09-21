
# Hands-On Machine Learning Journey

[](https://www.google.com/search?q=https://colab.research.google.com/github/YOUR_USERNAME/handson_ml)

This repository chronicles my personal journey through the foundational concepts of traditional machine learning. It blends theoretical understanding and meticulous, from-scratch implementation, in order to build a deep, intuitive grasp of how these powerful algorithms work.

The project is structured to separate conceptual exploration from raw implementation, allowing for a focused approach to learning.

-----

## 💡Core Concepts Explored

This project covers a wide array of classical machine learning algorithms and techniques, including:

  * **Linear Regression:** Understanding linear relationships and predictive modeling.
  * **Logistic & Softmax Regression:** Grasping the fundamentals of classification.
  * **Support Vector Machines (SVMs):** Exploring maximum margin classifiers.
  * **Decision Trees & Random Forests:** Diving into the fascinating world of tree-based models.
  * **Ensemble Methods:** Bagging, Pasting, Boosting (AdaBoost, Gradient Boosting), and Stacking.
  * **Dimensionality Reduction:** Techniques like PCA, kernel PCA and LLE for feature extraction.
  * **Unsupervised Learning:** Methods for clustering, anomaly detection, and density estimation.

-----

## 📁 Project Structure

The repository is organized into two primary directories:

  * `notebooks/`: A collection of Jupyter notebooks that serve as my study log. They contain detailed notes, performance insights, and hands-on explorations of ML concepts using the excellent `scikit-learn` library.
  * `implementations/`: A Python package containing my own implementations of several machine learning models. The goal was to build them from the ground up, using mainly **NumPy**, while adhering to the elegant `scikit-learn` API conventions (`.fit()`, `.predict()`, etc.).

-----

## 🔧 From-Scratch Implementations

The `implementations` package currently includes the following models, with more to come:

  * `SoftmaxRegressor`
  * `MinibatchGDRegressor`
  * `MyTreeClassifier`
  * `MyTreeRegressor`
  * `MyStackingClasssifier`

These classes are designed for learning and demonstration, with a clear view into the inner workings of each algorithm.

-----

## 🚀 Getting Started

You can explore this project in two ways: online through Google Colab or locally on your machine.

### 1\. Option A: Explore NOTEBOOKS in Google Colab

Navigate into the desired notebook, hover thy cursor over the ![Run in Colab](https://colab.research.google.com/assets/colab-badge.svg)   button at the top and simply click. Tada!


### 2\. Option B: Explore the entire project on your local machine.

First, clone this repository to your local machine.

```bash
git clone https://github.com/bobmh43/handson_ml.git
cd handson_ml
```
You can then explore the python scripts under `implementations` after installing the dependencies using
```bash
pip install -r requirements-dev.txt
``` 
or the notebooks under `notebooks` using `jupyterlab`.

```bash
# install the jupyter notebook environment
pip install jupyterlab

# Start Jupyter Lab to explore the notebooks
jupyter lab
```



-----
## 🛠️ More to Come

Stay tuned for more implementations of traditional machine learning methods!


## 🙏 Source and Acknowledgements

This entire learning project was profoundly inspired by Aurélien Géron's masterful book, **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow"**. Its clear explanations and detailed code examples makes it an invaluable trove of ML knowledge. Many thanks to this tome for its companionship on countless long summer nights.