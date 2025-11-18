# ==============================================================
# 1. Import Libraries
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================================================
# 2. Load Dataset
# ==============================================================

df = pd.read_csv("/content/gdrive/MyDrive/Colab Notebooks/london_weather.csv")
print("Dataset Loaded Successfully!")
df.head()

# ==============================================================
# 3. Convert Date & Sort
# ==============================================================

df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.sort_values('date')

# ==============================================================
# 4. Handle Missing Values
# ==============================================================

df = df.fillna(df.mean(numeric_only=True))

# ==============================================================
# 5. Create Target Column: RainTomorrow (>= 1 mm rain)
# ==============================================================

df['RainTomorrow'] = df['precipitation'].shift(-1)
df['RainTomorrow'] = (df['RainTomorrow'] >= 1.0).astype(int)
df = df[:-1]   # remove last row because target is shifted

# ==============================================================
# 6. Select Features (NOW INCLUDING precipitation)
# ==============================================================

features = [
    'cloud_cover',
    'sunshine',
    'global_radiation',
    'max_temp',
    'mean_temp',
    'min_temp',
    'precipitation',   
    'pressure',
    'snow_depth'
]

X = df[features]
y = df['RainTomorrow']

# ==============================================================
# 7. Train/Test Split (Stratified)
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================================================
# 8. Hyperparameter Tuning for Random Forest
# ==============================================================

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42)

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("Best Parameters Found:")
print(grid.best_params_)

best_rf = grid.best_estimator_

# ==============================================================
# 9. Evaluate Fine-Tuned Model
# ==============================================================

y_pred_rf = best_rf.predict(X_test)

print("\n📌 Tuned Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# ==============================================================
# 10. Feature Importance
# ==============================================================

importances = best_rf.feature_importances_
fi = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(
    by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=fi, x='Importance', y='Feature')
plt.title("Feature Importance (Tuned Random Forest)")
plt.show()



from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Predict on the test set
y_pred = best_rf.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Optional: Print classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


def predict_rain_simple():
    print("\nEnter today's weather values:")

    # Take inputs
    cloud_cover = float(input("Cloud Cover: "))
    sunshine = float(input("Sunshine: "))
    global_radiation = float(input("Global Radiation: "))
    max_temp = float(input("Max Temperature: "))
    mean_temp = float(input("Mean Temperature: "))
    min_temp = float(input("Min Temperature: "))
    precipitation = float(input("Precipitation: "))
    pressure = float(input("Pressure: "))
    snow_depth = float(input("Snow Depth: "))

    # Convert to array
    input_data = np.array([[
        cloud_cover,
        sunshine,
        global_radiation,
        max_temp,
        mean_temp,
        min_temp,
        precipitation,
        pressure,
        snow_depth
    ]])

    # Predict
    result = best_rf.predict(input_data)[0]

    # Output
    if result == 1:
        print("\n🌧️ Rain Tomorrow: YES")
    else:
        print("\n☀️ Rain Tomorrow: NO")
predict_rain_simple()
