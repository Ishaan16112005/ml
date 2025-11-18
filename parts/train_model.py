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
