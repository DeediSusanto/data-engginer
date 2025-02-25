import pandas as pd
from sqlalchemy import create_engine
from prophet import Prophet
import matplotlib.pyplot as plt

# Koneksi ke database
engine = create_engine("mysql+mysqlconnector://root:@localhost/sales_db")
query = "SELECT order_date, SUM(total) AS revenue FROM sales_data GROUP BY order_date;"
df = pd.read_sql(query, engine)

# Format data
df.columns = ['ds', 'y']

# Model Time-Series dengan Prophet
model = Prophet()
model.fit(df)

# Prediksi 90 hari ke depan
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Visualisasi hasil prediksi
model.plot(forecast)
plt.title("Prediksi Tren Pendapatan")
plt.show()



# Prediksi Tren Penjualan dengan Time-Series (Facebook Prophet)
# 📌 Tujuan: Memprediksi pendapatan masa depan berdasarkan data historis.

# Prediksi Time-Series → Perkiraan pendapatan di masa depan.


# SQL Query untuk Mengambil Data

# SELECT order_date, SUM(total) AS revenue 
# FROM sales_data 
# GROUP BY order_date;
