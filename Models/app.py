from flask import Flask, render_template, send_from_directory
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# --- Paths to results ---
CENTRAL_REPORT = r"results/centralised/power_ensemble_report.txt"
FEDERATED_REPORT = r"results/federated/federated_global_model_report.txt"
CENTRAL_SHAP = r"results/Centralised_Shapely/centralised_shap_summary.png"
FEDERATED_SHAP = r"results/federated_Shap/federated_shap_summary.png"

# --- Read accuracy from report files ---
def extract_accuracy(report_path):
    try:
        with open(report_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Accuracy" in line or "Final Global Model Accuracy" in line:
                    acc = float(line.split(":")[1].strip())
                    return acc
    except:
        return None

@app.route("/")
def index():
    central_acc = extract_accuracy(CENTRAL_REPORT)
    federated_acc = extract_accuracy(FEDERATED_REPORT)

    # --- Create a simple accuracy comparison chart ---
    labels = ["Centralised", "Federated"]
    accuracies = [central_acc or 0, federated_acc or 0]

    plt.figure(figsize=(6,4))
    plt.bar(labels, accuracies)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle="--", alpha=0.6)
    plt.savefig("static/results_comparison.png", bbox_inches="tight")
    plt.close()

    return render_template(
        "index.html",
        central_acc=central_acc,
        federated_acc=federated_acc,
        central_report=os.path.basename(CENTRAL_REPORT),
        federated_report=os.path.basename(FEDERATED_REPORT),
    )

@app.route("/download/<path:filename>")
def download_file(filename):
    directory = "results/centralised" if "central" in filename.lower() else "results/federated"
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
