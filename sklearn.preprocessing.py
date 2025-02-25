import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# ------------------------------
# 1. Koneksi ke Database dan Ambil Data
# ------------------------------
DB_USER = "root"         # Ganti sesuai konfigurasi MySQL kamu
DB_PASSWORD = ""         # Jika ada password, masukkan di sini
DB_HOST = "localhost"
DB_NAME = "sales_db"

engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

# Misalnya, kita ambil data penjualan yang sederhana, hanya untuk contoh.
# Pastikan tabel sales_data memiliki kolom: quantity, price, total
query = """
    SELECT quantity, price, total
    FROM sales_data
    LIMIT 1000;
"""
df = pd.read_sql(query, engine)
print("Data Asli:")
print(df.head())

# Jika data kamu belum memiliki kolom quantity atau price, kamu bisa gunakan contoh data berikut:
if df.empty:
    data = {
        "quantity": [1, 2, 5, 10, 20, 40, 60, 80, 100],
        "price": [100, 90, 85, 80, 75, 70, 65, 60, 55]
    }
    df = pd.DataFrame(data)
    df["total"] = df["quantity"] * df["price"]

# ------------------------------
# 2. Preprocessing Data dengan StandardScaler
# ------------------------------
scaler = StandardScaler()
features = df[["quantity", "price"]]
scaled_features = scaler.fit_transform(features)

df_scaled = pd.DataFrame(scaled_features, columns=["quantity_scaled", "price_scaled"])
df_scaled["total"] = df["total"]

print("\nData Setelah Scaling:")
print(df_scaled.head())

# ------------------------------
# 3. Bagi Data Training dan Testing
# ------------------------------
X = df_scaled[["quantity_scaled", "price_scaled"]]
y = df_scaled["total"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nUkuran Training Data: {X_train.shape}, Testing Data: {X_test.shape}")

# ------------------------------
# 4. Membuat Model Linear Regression dan Prediksi
# ------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nPrediksi Total Pendapatan pada Testing Data:")
print(y_pred)

# Evaluasi dengan Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error (MSE):", mse)

# Visualisasi Prediksi vs Data Aktual
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Data Aktual Total")
plt.ylabel("Prediksi Total")
plt.title("Prediksi Total Pendapatan: Data Aktual vs Prediksi")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
plt.show()

# ------------------------------
# 5. Clustering Data dengan KMeans
# ------------------------------
# Kita gunakan seluruh data scaled untuk clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_scaled["cluster"] = kmeans.fit_predict(X)

plt.figure(figsize=(8, 5))
sns.scatterplot(x="quantity_scaled", y="price_scaled", hue="cluster", data=df_scaled, palette="viridis")
plt.title("Clustering Data: Quantity vs Price")
plt.xlabel("Quantity (Scaled)")
plt.ylabel("Price (Scaled)")
plt.show()