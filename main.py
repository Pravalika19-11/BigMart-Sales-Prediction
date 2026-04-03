import os
import pandas as pd
import traceback
import pickle
from flask import Flask, render_template, request, send_file
from datetime import date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

app = Flask(__name__)

# Load Models
def load_model(path):
    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading model {path}: {e}")
        return None

lin_model = load_model("src/linear_regression.pkl")
poly_model = load_model("src/polynomial_regression.pkl")
rid_model = load_model("src/ridge_regression.pkl")
xgb_model = load_model("src/xgboost_regression.pkl")

# Load Test Data
data_path = "src/bigmart_sales_test.csv"
try:
    test_data = pd.read_csv(data_path)
    test_data.columns = test_data.columns.str.strip()
    if "Item_Outlet_Sales" in test_data.columns:
        X_test = test_data.drop(columns=["Item_Outlet_Sales"])
        y_test = test_data["Item_Outlet_Sales"]
    else:
        print("Warning: 'Item_Outlet_Sales' column missing.")
        X_test, y_test = None, None
except Exception as e:
    print("Error loading dataset:", e)
    X_test, y_test = None, None

# Get Raw Metrics
def get_raw_metrics(model, X, y):
    if model is None or X is None or y is None:
        return None
    try:
        preds = model.predict(X)
        return {
            "mae": mean_absolute_error(y, preds),
            "mse": mean_squared_error(y, preds),
            "r2": r2_score(y, preds)
        }
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None

# Generate Chart for Prediction
def generate_chart(predictions):
    plt.figure(figsize=(8, 5))
    sns.histplot(predictions, kde=True, bins=20, color="blue")
    plt.xlabel("Predicted Sales")
    plt.ylabel("Frequency")
    plt.title("Sales Prediction Distribution")
    chart_path = "static/sales_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path

# Home Page
@app.route("/")
def index():
    return render_template("home.html", prediction_text=None, show_buttons=False)

# Predict Sales
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            Item_Weight = float(request.form["Item_Weight"])
            Item_Fat_Content = request.form["Item_Fat_Content"]
            Item_Visibility = float(request.form["Item_Visibility"])
            Item_Type = request.form["Item_Type"]
            Item_MRP = float(request.form["Item_MRP"])
            Outlet_Establishment_Year = int(request.form["Outlet_Establishment_Year"])
            Outlet_Size = request.form["Outlet_Size"]
            Outlet_Location_Type = request.form["Outlet_Location_Type"]
            Outlet_Type = request.form["Outlet_Type"]

            Outlet_Years = date.today().year - Outlet_Establishment_Year

            input_data = pd.DataFrame([{
                'Item_Weight': Item_Weight,
                'Item_Visibility': Item_Visibility,
                'Item_MRP': Item_MRP,
                'Outlet_Establishment_Year': Outlet_Establishment_Year,
                'Outlet_Years': Outlet_Years,
                'Item_Fat_Content': Item_Fat_Content,
                'Outlet_Size': Outlet_Size,
                'Outlet_Location_Type': Outlet_Location_Type,
                'Outlet_Type': Outlet_Type,
                'Item_Type': Item_Type
            }])

            selected_model = request.form.get("model")
            model = {"linear": lin_model, "polynomial": poly_model, "ridge": rid_model, "xgboost": xgb_model}.get(selected_model)

            if model is None:
                return render_template("home.html", prediction_text="Error: Invalid model selection.", show_buttons=False)

            result = model.predict(input_data)[0]
            chart_path = generate_chart([result])

            return render_template("home.html", prediction_text=f"Predicted Sales: ₹{result:.2f}", show_buttons=True, chart_path=chart_path)

        except Exception as e:
            print("Error:", e)
            traceback.print_exc()
            return render_template("home.html", prediction_text=f"Error: {str(e)}", show_buttons=False)
    return render_template("home.html", prediction_text=None, show_buttons=False)

# Model Performance Route
@app.route("/performance")
def performance():
    try:
        if X_test is None or y_test is None:
            return render_template("performance.html", error="Error: Test dataset not loaded.")

        raw_results = {
            "Linear Regression": get_raw_metrics(lin_model, X_test, y_test),
            "Polynomial Regression": get_raw_metrics(poly_model, X_test, y_test),
            "Ridge Regression": get_raw_metrics(rid_model, X_test, y_test),
            "XGBoost Regression": get_raw_metrics(xgb_model, X_test, y_test),
        }

        filtered = {k: v for k, v in raw_results.items() if v is not None}
        df = pd.DataFrame.from_dict(filtered, orient='index')
        df['inv_mae'] = 1 - MinMaxScaler().fit_transform(df[['mae']])
        df['inv_mse'] = 1 - MinMaxScaler().fit_transform(df[['mse']])
        df['r2_norm'] = MinMaxScaler().fit_transform(df[['r2']])
        df['accuracy'] = ((df['inv_mae'] + df['inv_mse'] + df['r2_norm']) / 3).round(3)
        best_model = df['accuracy'].idxmax()

        metrics = {
            "Model": df.index.tolist(),
            "MAE": df['inv_mae'].round(3).tolist(),
            "MSE": df['inv_mse'].round(3).tolist(),
            "R2": df['r2_norm'].round(3).tolist(),
            "Accuracy": df['accuracy'].tolist()
        }

        return render_template("performance.html", metrics=metrics, best_model=best_model)

    except Exception as e:
        print("Performance evaluation failed:", e)
        return render_template("performance.html", error=str(e))

# EDA Report
@app.route("/eda")
def eda():
    eda_report_path = "eda_report.html"
    if not os.path.exists(eda_report_path):
        profile = ProfileReport(test_data, explorative=True)
        profile.to_file(eda_report_path)
    return send_file(eda_report_path)

if __name__ == "__main__":
    app.run(debug=True)
