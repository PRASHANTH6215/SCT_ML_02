# SCT_ML_02
# 🧠 Customer Segmentation using Clustering Algorithms

This project performs **customer segmentation** using multiple clustering algorithms such as **K-Means**, **Hierarchical Clustering**, and **DBSCAN** on the Mall Customers dataset.

---

## 📌 Project Overview

Customer segmentation helps businesses understand different groups of customers based on their behavior and characteristics.

In this project:

* Data is preprocessed and scaled
* A synthetic feature **Purchase Frequency** is added
* Multiple clustering techniques are applied
* Results are visualized in a **3D scatter plot**

---

## 📊 Features Used

* Age
* Annual Income
* Spending Score
* Purchase Frequency *(randomly generated)*

---

## ⚙️ Algorithms Implemented

1. **K-Means Clustering**
2. **Agglomerative (Hierarchical) Clustering**
3. **DBSCAN (Density-Based Clustering)**

---

## 📈 Visualization

* 3D scatter plot using **Matplotlib**
* Displays clustering results from K-Means

---

## 📂 Project Structure

```
SCT_TASK_02/
│
├── SCT_TASK_02.py        # Main Python script
├── Mall_Customers.csv    # Dataset
├── README.md             # Project documentation
├── requirements.txt      # Dependencies
```

---

## 🚀 How to Run

1. Clone the repository or download the files

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the script:

```
python SCT_TASK_02.py
```

---

## 🧪 Output

* 3D visualization of customer clusters
* Cluster labels added to dataset:

  * KMeans
  * Hierarchical
  * DBSCAN

---

## 💡 Use Cases

* Marketing strategy optimization
* Customer targeting
* Business analytics

---

## ⚠️ Notes

* DBSCAN results may include noise points labeled as `-1`
* Purchase Frequency is randomly generated for demonstration

---

## 📌 Future Improvements

* Add Streamlit dashboard
* Use real behavioral datasets
* Hyperparameter tuning for better clustering

---

## 👨‍💻 Author

Prashanth B
