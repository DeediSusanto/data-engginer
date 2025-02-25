#  Customer Segmentation dengan Clustering (K-Means)
# 📌 Tujuan: Mengelompokkan pelanggan berdasarkan pola pembelian.

# Clustering Pelanggan → Segmentasi pelanggan untuk strategi bisnis.

# SQL Query untuk Data Pelanggan

# SELECT customer_name, SUM(total) AS total_spent, COUNT(order_id) AS total_orders 
# FROM sales_data 
# GROUP BY customer_name;

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sqlalchemy import create_engine

# Load data dari MySQL
engine = create_engine("mysql+mysqlconnector://root:@localhost/sales_db")
query = "SELECT customer_name, SUM(total) AS total_spent, COUNT(order_id) AS total_orders FROM sales_data GROUP BY customer_name;"
df = pd.read_sql(query, engine)

# Clustering dengan KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['total_spent', 'total_orders']])

# Visualisasi hasil clustering
plt.figure(figsize=(8,5))
sns.scatterplot(x="total_spent", y="total_orders", hue="cluster", data=df, palette="viridis")
plt.title("Customer Segmentation")
plt.show()
