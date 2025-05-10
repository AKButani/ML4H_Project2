
# Q2.1: Fit a Lasso Regression Model with L1 Regularization

# Fit the Lasso regression model
#lasso = Lasso(alpha=0.3)
#lasso.fit(X_train_scaled, y_train)
log_lasso_model = LogisticRegression(penalty='l1', C=1, solver='liblinear', random_state=30)  # C is the inverse of alpha
log_lasso_model.fit(X_train_scaled, y_train)
coefficients = log_lasso_model.coef_.flatten()

coefficients.size
feature_names.size

# Print coefficients and corresponding column names
feature_names = X_train_NOlabels.columns
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
print(coefficients_df)

# Predictions
y_pred_binary = log_lasso_model.predict(X_test_scaled)

# Q2.3: Quantify the Performance of the Model
# Compute F1-Score
f1 = f1_score(y_test, y_pred_binary)
# Compute Balanced Accuracy
balanced_accuracy = balanced_accuracy_score(y_test, y_pred_binary)

# Print metrics
print(f"F1 Score: {f1:.3f}")
print(f"Balanced Accuracy: {balanced_accuracy:.3f}")

# Detailed classification report
print(classification_report(y_test, y_pred_binary))

# Q2.4: Visualize the Feature Importance
# Plot feature importance (absolute value of coefficients)

# Calculate importance and sort
feature_importance = np.abs(coefficients)
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_coefficients = coefficients[sorted_idx]
sorted_features = feature_names[sorted_idx]

# Assign colors based on sign
colors = ['green' if coef > 0 else 'red' for coef in sorted_coefficients]

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, feature_importance[sorted_idx], align='center', color=colors)
plt.xlabel('Feature Importance (|Coefficient|) (red= negative controbution to heart disease =1)')
plt.title('Lasso Logistic Regression Feature Importances')
plt.gca().invert_yaxis()  # optional: highest importance on top
plt.savefig('P1_Q2_lasso_feat_imp_plot.png', dpi=300, bbox_inches='tight')