import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


SERVICES = ["auth-service", "payment-service", "order-service", "search-service", "inventory-service"]

NORMAL_MESSAGES = [
    "Request processed successfully",
    "User login successful",
    "Fetched results from database",
    "Cache hit for request",
    "API call completed",
    "Order created successfully",
    "Payment verified",
    "Session validated"
]

ERROR_MESSAGES = [
    "Database connection timeout",
    "NullPointerException occurred",
    "Service unavailable",
    "Authentication failed",
    "Payment gateway error",
    "OutOfMemoryError detected",
    "Disk read failure",
    "Unexpected token in JSON response"
]


def generate_logs(n_logs=5000, anomaly_ratio=0.03):
    start_time = datetime.now() - timedelta(hours=5)
    logs = []

    for i in range(n_logs):
        timestamp = start_time + timedelta(seconds=i * random.randint(1, 4))
        service = random.choice(SERVICES)

        # Normal behavior
        latency = np.random.normal(loc=120, scale=30)
        request_count = np.random.poisson(lam=20)
        status_code = 200
        level = "INFO"
        message = random.choice(NORMAL_MESSAGES)

        true_anomaly = 0

        # Inject anomalies
        if random.random() < anomaly_ratio:
            true_anomaly = 1
            anomaly_type = random.choice(["latency_spike", "error_spike", "traffic_spike"])

            if anomaly_type == "latency_spike":
                latency = np.random.normal(loc=1200, scale=200)
                level = "ERROR"
                message = "Latency spike detected while processing request"
                status_code = random.choice([500, 503])

            elif anomaly_type == "error_spike":
                latency = np.random.normal(loc=250, scale=80)
                level = "ERROR"
                message = random.choice(ERROR_MESSAGES)
                status_code = random.choice([500, 502, 503])

            elif anomaly_type == "traffic_spike":
                latency = np.random.normal(loc=250, scale=70)
                request_count = np.random.poisson(lam=200)
                level = "WARN"
                message = "Unusual traffic spike detected"
                status_code = 200

        latency = max(1, latency)
        request_count = max(1, request_count)

        logs.append({
            "timestamp": timestamp,
            "service": service,
            "level": level,
            "message": message.lower(),
            "latency_ms": round(latency, 2),
            "request_count": int(request_count),
            "status_code": status_code,
            "true_anomaly": true_anomaly
        })

    return pd.DataFrame(logs)


def add_rolling_features(df, window=20):
    df = df.sort_values("timestamp")

    df["error_flag"] = (df["level"] == "ERROR").astype(int)

    df["latency_roll_mean"] = df.groupby("service")["latency_ms"].transform(
        lambda x: x.rolling(window, min_periods=5).mean()
    )

    df["latency_roll_std"] = df.groupby("service")["latency_ms"].transform(
        lambda x: x.rolling(window, min_periods=5).std()
    )

    df["req_roll_mean"] = df.groupby("service")["request_count"].transform(
        lambda x: x.rolling(window, min_periods=5).mean()
    )

    df["req_roll_std"] = df.groupby("service")["request_count"].transform(
        lambda x: x.rolling(window, min_periods=5).std()
    )

    df["error_rate"] = df.groupby("service")["error_flag"].transform(
        lambda x: x.rolling(window, min_periods=5).mean()
    )

    df = df.fillna(0)
    return df


def build_features(df):
    vectorizer = TfidfVectorizer(max_features=80)
    tfidf = vectorizer.fit_transform(df["message"])
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    numeric = df[[
        "latency_ms", "request_count", "status_code",
        "latency_roll_mean", "latency_roll_std",
        "req_roll_mean", "req_roll_std",
        "error_rate"
    ]].reset_index(drop=True)

    X = pd.concat([numeric, tfidf_df], axis=1)
    return X


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("Generating logs...")
    df = generate_logs(n_logs=5000, anomaly_ratio=0.03)
    df.to_csv("data/raw_logs.csv", index=False)

    print("Feature engineering...")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_rolling_features(df)

    X = build_features(df)

    print("Training Isolation Forest...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    model.fit(X_scaled)

    scores = model.decision_function(X_scaled)
    preds = model.predict(X_scaled)

    df["anomaly_score"] = scores
    df["anomaly_flag"] = (preds == -1).astype(int)

    df.to_csv("data/anomaly_results.csv", index=False)

    # Evaluation
    print("\n--- Evaluation (Synthetic Ground Truth) ---")
    print("Confusion Matrix:")
    print(confusion_matrix(df["true_anomaly"], df["anomaly_flag"]))

    print("\nClassification Report:")
    print(classification_report(df["true_anomaly"], df["anomaly_flag"]))

    # Report writing
    total = len(df)
    anomalies = df["anomaly_flag"].sum()

    report = []
    report.append("LOGANOM REPORT\n")
    report.append(f"Total logs: {total}")
    report.append(f"Anomalies detected: {anomalies}")
    report.append(f"Anomaly percentage: {round((anomalies/total)*100, 2)}%")

    report.append("\nConfusion Matrix:")
    report.append(str(confusion_matrix(df["true_anomaly"], df["anomaly_flag"])))

    report.append("\nClassification Report:")
    report.append(classification_report(df["true_anomaly"], df["anomaly_flag"]))

    report.append("\nTop 10 Most Suspicious Events:\n")
    top = df[df["anomaly_flag"] == 1].sort_values("anomaly_score").head(10)
    report.append(top[["timestamp", "service", "level", "latency_ms", "request_count", "message", "anomaly_score"]].to_string(index=False))

    with open("reports/anomaly_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\nSaved outputs:")
    print(" - data/raw_logs.csv")
    print(" - data/anomaly_results.csv")
    print(" - reports/anomaly_summary.txt")

    # Plotting
    print("\nPlotting anomalies...")

    normal = df[df["anomaly_flag"] == 0]
    anomaly = df[df["anomaly_flag"] == 1]

    plt.figure(figsize=(14, 6))
    plt.plot(normal["timestamp"], normal["latency_ms"], label="Normal")
    plt.scatter(anomaly["timestamp"], anomaly["latency_ms"], marker="x", label="Anomaly")
    plt.title("LogAnom: Anomaly Detection Over Time")
    plt.xlabel("Time")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()