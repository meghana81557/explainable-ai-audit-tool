import shap
import joblib
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("model.pkl")
X = load_iris().data

# Create SHAP explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Generate and save SHAP summary plot
shap.summary_plot(shap_values, X, show=False)
plt.savefig("audit_report.png")

print("✅ Explainability audit completed successfully!")
print("Report saved as audit_report.png")
