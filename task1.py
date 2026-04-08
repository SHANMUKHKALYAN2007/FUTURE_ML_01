# ================================
# 📊 SALES FORECASTING PROJECT
# ================================

# 📌 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 📌 Step 2: Load Dataset (FIXED)
df = pd.read_csv("sales_data.csv", encoding='latin1', engine='python')

# 🔥 Clean column names (remove spaces + lowercase)
df.columns = df.columns.str.strip().str.lower()

print("Columns in dataset:", df.columns)

# 🔥 Auto-detect columns (IMPORTANT)
# Find date column
date_col = None
for col in df.columns:
    if 'date' in col:
        date_col = col

# Find sales column
sales_col = None
for col in df.columns:
    if 'sale' in col:
        sales_col = col

# ❌ If not found, stop
if date_col is None or sales_col is None:
    raise Exception("❌ Could not find Date or Sales column. Check your dataset!")

# Rename columns properly
df.rename(columns={
    date_col: 'Date',
    sales_col: 'Sales'
}, inplace=True)

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Remove invalid rows
df = df.dropna(subset=['Date'])

# Set index
df.set_index('Date', inplace=True)

print("\n📊 Dataset Preview:")
print(df.head())

# 📌 Step 3: Feature Engineering
df['Day'] = df.index.day
df['Month'] = df.index.month
df['Year'] = df.index.year
df['DayOfWeek'] = df.index.dayofweek

# 📌 Step 4: Prepare Data
X = df[['Day', 'Month', 'Year', 'DayOfWeek']]
y = df['Sales']

# Train-Test Split
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 📌 Step 5: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Model Training Completed!")

# 📌 Step 6: Predictions
y_pred = model.predict(X_test)

# 📌 Step 7: Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📈 Model Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)

# 📌 Step 8: Future Forecast
future_dates = pd.date_range(start=df.index[-1], periods=10, freq='D')

future_df = pd.DataFrame(index=future_dates)
future_df['Day'] = future_df.index.day
future_df['Month'] = future_df.index.month
future_df['Year'] = future_df.index.year
future_df['DayOfWeek'] = future_df.index.dayofweek

future_df['Predicted Sales'] = model.predict(future_df)

print("\n🔮 Future Sales Prediction:")
print(future_df)

# ================================
# 📊 VISUALIZATIONS
# ================================

# 1️⃣ Actual vs Predicted
plt.figure()
plt.plot(y_test.index, y_test, label="Actual Sales")
plt.plot(y_test.index, y_pred, label="Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()

# 2️⃣ Sales Trend
plt.figure()
plt.plot(df.index, df['Sales'])
plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid()
plt.show()

# 3️⃣ Monthly Sales
monthly_sales = df['Sales'].resample('M').sum()

plt.figure()
monthly_sales.plot()
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid()
plt.show()

# 4️⃣ Sales Distribution
plt.figure()
plt.hist(df['Sales'], bins=10)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# 5️⃣ Future Forecast
plt.figure()
plt.plot(df.index, df['Sales'], label="Actual Sales")
plt.plot(y_test.index, y_pred, label="Predicted Sales")
plt.plot(future_df.index, future_df['Predicted Sales'], linestyle='dashed', label="Future Forecast")
plt.title("Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()

# ================================
# ✅ END OF PROJECT
# ================================