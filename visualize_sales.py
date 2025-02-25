import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# Koneksi ke MySQL
DB_USER = "root"  # Sesuaikan dengan username MySQL kamu
DB_PASSWORD = ""  # Jika ada password, masukkan di sini
DB_HOST = "localhost"
DB_NAME = "sales_db"

engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

# 1️⃣ **Tren Penjualan per Bulan**
query = """
    SELECT DATE_FORMAT(order_date, '%Y-%m') AS month, SUM(total) AS revenue
    FROM sales_data
    GROUP BY month
    ORDER BY month;
"""
df = pd.read_sql(query, engine)

plt.figure(figsize=(12, 6))
sns.lineplot(x="month", y="revenue", data=df, marker="o", linewidth=2)
plt.xticks(rotation=45)
plt.xlabel("Bulan")
plt.ylabel("Pendapatan")
plt.title("Tren Penjualan per Bulan")
plt.grid(True)
plt.show()

# 2️⃣ **Produk Terlaris**
query = """
    SELECT product_name, SUM(quantity) AS total_sold
    FROM sales_data
    GROUP BY product_name
    ORDER BY total_sold DESC
    LIMIT 5;
"""
df = pd.read_sql(query, engine)

plt.figure(figsize=(10, 5))
sns.barplot(x="total_sold", y="product_name", data=df, palette="coolwarm")
plt.xlabel("Jumlah Terjual")
plt.ylabel("Produk")
plt.title("Top 5 Produk Terlaris")
plt.show()
