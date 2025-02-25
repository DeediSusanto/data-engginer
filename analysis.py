import pandas as pd
from sqlalchemy import create_engine

# Konfigurasi koneksi MySQL
DB_USER = "root"  # Ganti dengan username MySQL kamu
DB_PASSWORD = ""  # Jika ada password, masukkan di sini
DB_HOST = "localhost"
DB_NAME = "sales_db"

# Buat koneksi ke MySQL
engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

# -----------------------
# Analisis Data
# -----------------------

# 1. Total Pendapatan
query = "SELECT SUM(total) AS total_revenue FROM sales_data;"
df_total = pd.read_sql(query, engine)
print("Total Pendapatan:")
print(df_total)
print("\n-------------------------\n")

# 2. Produk Terlaris
query = """
    SELECT product_name, SUM(quantity) AS total_sold 
    FROM sales_data 
    GROUP BY product_name 
    ORDER BY total_sold DESC 
    LIMIT 5;
"""
df_produk = pd.read_sql(query, engine)
print("Top 5 Produk Terlaris:")
print(df_produk)
print("\n-------------------------\n")

# 3. Penjualan per Bulan (Tren)
query = """
    SELECT DATE_FORMAT(order_date, '%Y-%m') AS month, SUM(total) AS revenue 
    FROM sales_data 
    GROUP BY month 
    ORDER BY month;
"""
df_tren = pd.read_sql(query, engine)
print("Tren Penjualan per Bulan:")
print(df_tren)
