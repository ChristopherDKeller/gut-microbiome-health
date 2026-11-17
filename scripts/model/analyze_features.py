# --------------------------------------------------
# 7. Feature importance
# --------------------------------------------------

# Aus trainierter Pipeline auslesen
preprocessor = model.named_steps["columntransformer"]
classifier = model.named_steps["logisticregression"]

# Feature-Namen nach Imputation und One-Hot-Encoding
feature_names = preprocessor.get_feature_names_out()

# Koeffizienten der logistischen Regression
coefficients = classifier.coef_[0]

# Zusammenführen
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

# Nach Koeffizienten sortieren
coef_df = coef_df.sort_values("coefficient")

zero_features = coef_df[coef_df["coefficient"] == 0]
selected_features = coef_df[coef_df["coefficient"] != 0]
selected_features = selected_features.sort_values("coefficient", ascending=False)

#print("Removed:", len(zero_features))
#print(zero_features.to_string(index=False))

print("\nSelected:", len(selected_features))
print(selected_features.to_string(index=False))