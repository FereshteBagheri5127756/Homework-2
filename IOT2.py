
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 1. بارگذاری داده‌ها
data = pd.read_csv('C:\Users\F_Bagheri\OneDrive\Desktop')
data['date'] = pd.to_datetime(data['date'])
data['dayofyear'] = data['date'].dt.dayofyear  # ویژگی‌های زمانی
data['hour'] = data['date'].dt.hour  # ویژگی‌های زمانی

# انتخاب ویژگی‌های ورودی و هدف
X = data[['dayofyear', 'hour']]
y = data['consumption']


# 2. تقسیم داده‌ها به دو مجموعه آموزشی و آزمون
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. ساخت و آموزش مدل
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. پیش‌بینی و ارزیابی مدل
y_pred = model.predict(X_test)

# محاسبه خطاها
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("Mean Absolute Error:", mae)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# 5. نمودار مقایسه‌ای داده‌های واقعی و پیش‌بینی‌شده
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Consumption", color='blue')
plt.plot(y_pred, label="Predicted Consumption", color='red', linestyle='dashed')
plt.legend()
plt.title("Actual vs Predicted Energy Consumption")
plt.xlabel("Samples")
plt.ylabel("Consumption")
plt.show()

# 6. نمودار خطاهای پیش‌بینی
errors = y_test.values - y_pred
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=30, color='purple', 

edgecolor='black')
plt.title("Distribution of Prediction Errors")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.show()

# 7. نمودار اهمیت ویژگی‌ها
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(8, 5))
plt.bar(features, importances, color='green')
plt.title("Feature Importance in Energy Consumption Prediction")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()
