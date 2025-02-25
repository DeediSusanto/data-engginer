Hi, let me introduce myself Dedi. and come learn with me

Disini saya akan share Mulai dari konsep Data Warehouse Engineer, instalasi, hingga pengolahan data menggunakan SQL dan Python.  

---  

## **📌 Data Engineering Project: Sales Data Analysis**  

### **📖 Deskripsi Proyek**  
Proyek ini bertujuan untuk melakukan pengolahan dan analisis data penjualan menggunakan **MySQL**, **Python (pandas, scikit-learn)**, dan visualisasi dengan **Matplotlib & Seaborn**. Selain itu, proyek ini juga memanfaatkan **Linear Regression** untuk prediksi tren dan **K-Means Clustering** untuk segmentasi data.

---

## **📂 Struktur Folder**  
```bash
📂 data-engginer-project
├── 📂 data                # Dataset contoh (jika ada)
│   ├── sales_data.csv
├── 📂 scripts             # Script Python
│   ├── query_sales.py      # Query database
│   ├── prediksi_tren.py    # Prediksi menggunakan Linear Regression
│   ├── clustering.py       # Clustering dengan KMeans
├── README.md               # Dokumentasi proyek ini
```

---

## **🛠️ Instalasi & Setup**  

### **1️⃣ Instal MySQL dan phpMyAdmin**  
Pastikan MySQL dan phpMyAdmin sudah terinstall. Jika belum, download dan install dari:  
🔗 [MySQL Download](https://dev.mysql.com/downloads/)  
🔗 [phpMyAdmin Download](https://www.phpmyadmin.net/)  

### **2️⃣ Instal Python dan Paket yang Dibutuhkan**  
Pastikan Python sudah terinstall. Jika belum, download di:  
🔗 [Python Download](https://www.python.org/downloads/)  

Setelah itu, install dependensi dengan menjalankan perintah berikut di terminal/cmd:  
```sh
pip install pandas sqlalchemy mysql-connector-python matplotlib seaborn scikit-learn
```

### **3️⃣ Setup Database di MySQL**  
Jalankan perintah berikut untuk membuat database dan tabel di **phpMyAdmin** atau **MySQL CLI**:
```sql
CREATE DATABASE sales_db;

USE sales_db;

CREATE TABLE sales_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_id VARCHAR(50),
    product_name VARCHAR(100),
    category VARCHAR(50),
    quantity INT,
    price DECIMAL(10,2),
    total DECIMAL(10,2),
    order_date DATE,
    customer_name VARCHAR(100),
    customer_email VARCHAR(100)
);
```
Untuk import data, gunakan `sales_data.csv` melalui phpMyAdmin atau CLI MySQL:
```sql
LOAD DATA INFILE 'sales_data.csv'
INTO TABLE sales_data
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
```

---

Download Example Data Training :
Saya telah membuat dataset berisi 5.000 transaksi penjualan dalam format CSV.
📥 : sales_data.csv

## **📊 Pengolahan Data dengan Python**  

### **🔹 1. Query Data dengan `query_sales.py` & 'visualize_sales.py'**  
File ini mengambil data dari MySQL dan menampilkannya dengan pandas.
```python
import pandas as pd
from sqlalchemy import create_engine

# Konfigurasi koneksi
DB_USER = "root"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_NAME = "sales_db"

engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

query = "SELECT * FROM sales_data LIMIT 10;"
df = pd.read_sql(query, engine)

print(df)
```
**Cara menjalankan:**
```sh
python scripts/query_sales.py
```

---

### **🔹 2. Prediksi Tren dengan Linear Regression (`prediksi_tren.py`)**  
File ini membuat model regresi linier untuk memprediksi pendapatan.
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine

# Koneksi ke database
engine = create_engine("mysql+mysqlconnector://root:@localhost/sales_db")

# Ambil data
query = "SELECT quantity, price, total FROM sales_data"
df = pd.read_sql(query, engine)

# Persiapan data
X = df[['quantity', 'price']]
y = df['total']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
```
**Cara menjalankan:**
```sh
python scripts/prediksi_tren.py
```

---

### **🔹 3. Clustering dengan KMeans (`sklearn.preprocessing.py`)**  
File ini melakukan clustering pada data penjualan berdasarkan jumlah dan harga produk.
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sqlalchemy import create_engine

# Koneksi database
engine = create_engine("mysql+mysqlconnector://root:@localhost/sales_db")
query = "SELECT quantity, price FROM sales_data"
df = pd.read_sql(query, engine)

# Clustering dengan KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(df[["quantity", "price"]])

# Visualisasi hasil clustering
plt.figure(figsize=(8, 5))
sns.scatterplot(x="quantity", y="price", hue="cluster", data=df, palette="viridis")
plt.title("Clustering Data: Quantity vs Price")
plt.xlabel("Quantity")
plt.ylabel("Price")
plt.show()
```
**Cara menjalankan:**
```sh
python scripts/sklearn.preprocessing.py
```

---

## **📈 Visualisasi Data**
Proyek ini menggunakan **Matplotlib & Seaborn** untuk membuat grafik tren dan clustering.  
- Scatter plot untuk prediksi tren penjualan  
- Clustering untuk melihat segmentasi data berdasarkan harga dan jumlah penjualan  

---

## **🎯 Hasil & Insight**
1. **Prediksi Pendapatan:**  
   - Model regresi linier memberikan gambaran perkiraan total pendapatan berdasarkan kuantitas dan harga.  
   - **Mean Squared Error (MSE)** digunakan untuk mengevaluasi seberapa baik model bekerja.  

2. **Segmentasi Produk dengan Clustering:**  
   - Produk dikelompokkan ke dalam 3 kategori berdasarkan jumlah dan harga.  
   - Membantu dalam strategi penjualan, misalnya menentukan harga promo untuk produk dengan volume tinggi.  

3. **Strategi Bisnis:**  
   - Data ini dapat digunakan oleh tim penjualan untuk melihat tren dan memprediksi stok yang perlu disiapkan.  

---

Contoh langsung bagaimana 5 modul dari scikit-learn digunakan dalam pengolahan data secara praktis. Kita akan menggunakan dataset penjualan (sales_data) untuk memprediksi total pendapatan berdasarkan jumlah produk yang terjual.

1. Data Preparation (sklearn.preprocessing)
Kita akan menggunakan StandardScaler untuk menormalkan data agar lebih optimal untuk model machine learning.


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Contoh data penjualan (biasanya dari database)
data = {
    "quantity": [1, 2, 5, 10, 20, 40, 60, 80, 100],
    "price": [100, 90, 85, 80, 75, 70, 65, 60, 55],
    "total": [100, 180, 425, 800, 1500, 2800, 3900, 4800, 5500]
}

df = pd.DataFrame(data)

# Standarisasi fitur (quantity dan price)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[["quantity", "price"]])

# Simpan dalam DataFrame baru
df_scaled = pd.DataFrame(df_scaled, columns=["quantity_scaled", "price_scaled"])
df_scaled["total"] = df["total"]

print(df_scaled.head())
🔹 Hasil: Data sudah dinormalisasi dan siap untuk diproses lebih lanjut.

2. Membagi Data Training dan Testing (sklearn.model_selection)
Kita gunakan train_test_split untuk membagi dataset menjadi 80% training dan 20% testing.


from sklearn.model_selection import train_test_split

# Pisahkan fitur (X) dan target (y)
X = df_scaled[["quantity_scaled", "price_scaled"]]
y = df_scaled["total"]

# Bagi dataset menjadi training (80%) dan testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")
🔹 Hasil: Dataset sudah dibagi menjadi training dan testing set.

3. Membuat Model Prediksi (sklearn.linear_model)
Gunakan LinearRegression untuk memprediksi total penjualan berdasarkan jumlah produk dan harga.


from sklearn.linear_model import LinearRegression

# Inisialisasi model
model = LinearRegression()

# Latih model dengan data training
model.fit(X_train, y_train)

# Prediksi dengan data testing
y_pred = model.predict(X_test)

print("Prediksi Total Pendapatan:", y_pred)
🔹 Hasil: Model berhasil mempelajari hubungan antara kuantitas produk dan total pendapatan.

4. Evaluasi Model (sklearn.metrics)
Gunakan mean_squared_error untuk mengukur performa model.

from sklearn.metrics import mean_squared_error

# Hitung MSE (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
🔹 Hasil: MSE menunjukkan seberapa jauh prediksi model dari data asli. Semakin kecil nilainya, semakin baik modelnya.

5. Clustering Data (sklearn.cluster)
Gunakan KMeans untuk mengelompokkan produk berdasarkan jumlah penjualan dan harga.

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Gunakan clustering dengan 3 grup
kmeans = KMeans(n_clusters=3, random_state=42)
df_scaled["cluster"] = kmeans.fit_predict(X)

# Visualisasi hasil clustering
plt.scatter(df_scaled["quantity_scaled"], df_scaled["price_scaled"], c=df_scaled["cluster"], cmap="viridis")
plt.xlabel("Quantity Scaled")
plt.ylabel("Price Scaled")
plt.title("Clustering Produk berdasarkan Penjualan")
plt.show()

🔹 Hasil: Data dikelompokkan ke dalam 3 kategori berdasarkan pola penjualan.

Kesimpulan
Data Preprocessing (StandardScaler) → Menormalkan data agar tidak bias.
Data Splitting (train_test_split) → Memisahkan data untuk pelatihan dan pengujian.
Linear Regression (LinearRegression) → Memprediksi total pendapatan.
Evaluasi (mean_squared_error) → Mengukur performa model dengan MSE.
Clustering (KMeans) → Mengelompokkan produk berdasarkan pola penjualan.

Dengan langkah ini, kita bisa menganalisis tren, memprediksi penjualan, dan mengelompokkan produk secara otomatis dengan scikit-learn! 🚀

## **💡 Next Steps**
- Menambahkan analisis time-series untuk memprediksi tren penjualan di masa depan.  
- Menggunakan model Machine Learning lebih kompleks seperti **XGBoost**.  
- Implementasi dashboard interaktif dengan **Streamlit atau Power BI**.  

---



## **📜 Referensi**
- [Pandas Documentation](https://pandas.pydata.org/docs/)  
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)  
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)  

---

Sekarang proyekmu sudah siap untuk diunggah ke **GitHub** dan dibagikan ke komunitas! 🚀  
Untuk mengupload ke GitHub, jalankan:
```sh
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/username/data-engginer-project.git
git push -u origin main
```

Semoga bermanfaat! Jika ada tambahan, silakan tanya. 🔥💡