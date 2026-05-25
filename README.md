# 🛒 E-Commerce Sales Analysis & Customer Review Prediction

## 📌 Project Overview

This project analyzes a real-world Brazilian e-commerce dataset from Olist containing over 100,000 orders. The goal was to perform end-to-end data analysis, extract business insights, build SQL-based analytics, and develop a machine learning model to predict low customer review scores.

The project demonstrates practical skills in:
- Python data analysis
- SQL querying
- Data visualization
- Machine learning
- Business insight generation

---

# 📂 Project Structure

```bash
ecommerce-sales-analysis/
├── data/
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   └── ...
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_SQL_Analysis.ipynb
│   └── 03_ML_Model.ipynb
│
├── dashboard/
│   └── sales_dashboard.pbix
│
├── README.md
└── requirements.txt


📊 Dataset

Dataset: Brazilian E-Commerce Public Dataset by Olist

The dataset includes:

Orders
Customers
Products
Payments
Reviews
Sellers
Geolocation data

Size:

100K+ orders
Multiple relational tables
Real-world transactional e-commerce data
🧪 Exploratory Data Analysis (EDA)

The EDA notebook includes:

Data cleaning and preprocessing
Missing value analysis
Duplicate checks
Revenue trend analysis
Customer shopping behavior analysis
Product category analysis
Review score distribution
Correlation analysis
Key Insights
📈 Revenue Trends
Revenue showed strong growth throughout 2017–2018
Peak sales occurred during late 2017 holiday periods
🏆 Top Product Categories

Top revenue-generating categories:

cama_mesa_banho
relogios_presentes
beleza_saude
⏰ Customer Shopping Behavior
Highest order activity occurred on weekdays
Peak purchasing hours were between 2PM–4PM
⭐ Customer Reviews
Most reviews were highly positive (4–5 stars)
Small but important group of dissatisfied customers existed
🗄️ SQL Analysis

SQLite was used inside Jupyter notebooks to perform relational data analysis.

SQL Tasks Performed
Revenue analysis by month
Top states by orders
Top-selling products
Average order value calculations
Multi-table joins using relational keys
Example SQL Concepts Used
JOIN
GROUP BY
ORDER BY
Aggregate functions
Subqueries
🤖 Machine Learning Model
Objective

Predict whether a customer would leave a low review score (1–2 stars).

Model Used
Logistic Regression (scikit-learn)
Features Used
Product price
Freight value
Actual delivery time
Estimated delivery time
ML Improvements Applied
Feature engineering
Handling class imbalance using class_weight='balanced'
Missing value handling
Final Results
Metric	Score
Accuracy	71%
Recall for Low Reviews	50%
F1-Score	0.34
Key Finding

Delivery-related features significantly improved prediction performance, suggesting shipping experience strongly influences customer satisfaction.

🛠️ Tools & Technologies
Programming
Python
SQL (SQLite)
Libraries
pandas
numpy
matplotlib
seaborn
scikit-learn
Environment
Jupyter Notebook
VS Code
Visualization
Matplotlib
Seaborn
Power BI