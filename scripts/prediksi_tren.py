import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
import numpy as np

# Koneksi ke MySQL
DB_USER = "root"         # Sesuaikan dengan username MySQL kamu
DB_PASSWORD = ""         # Jika ada password, masukkan di sini
DB_HOST = "localhost"
DB_NAME = "sales_db"

engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

# Ambil data tren penjualan per bulan
query = """
    SELECT DATE_FORMAT(order_date, '%Y-%m') AS month, SUM(total) AS revenue
    FROM sales_data
    GROUP BY month
    ORDER BY month;
"""
df_trend = pd.read_sql(query, engine)

# Ubah format bulan ke format numerik untuk model
# Misalnya, kita akan mengubah bulan menjadi indeks angka
df_trend['month_index'] = np.arange(len(df_trend))

# Fitur dan target
X = df_trend[['month_index']]
y = df_trend['revenue']

# Buat dan latih model regresi linear
model = LinearRegression()
model.fit(X, y)

# Prediksi pendapatan
df_trend['predicted_revenue'] = model.predict(X)

# Visualisasi hasil prediksi
plt.figure(figsize=(12, 6))
sns.lineplot(x='month_index', y='revenue', data=df_trend, marker="o", label="Actual Revenue")
sns.lineplot(x='month_index', y='predicted_revenue', data=df_trend, marker="o", label="Predicted Revenue", linestyle="--")
plt.xlabel("Bulan (indeks)")
plt.ylabel("Pendapatan")
plt.title("Prediksi Tren Penjualan per Bulan dengan Regresi Linear")
plt.legend()
plt.show()
