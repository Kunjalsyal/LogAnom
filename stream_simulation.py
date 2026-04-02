import time
import pandas as pd
import matplotlib.pyplot as plt

def stream_logs(path="data/raw_logs.csv", batch_size=200, delay=0.5):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    streamed = []

    print("Starting real-time log streaming simulation...\n")

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        streamed.append(batch)

        current = pd.concat(streamed)

        print(f"Processed logs: {len(current)}")

        # live plot
        plt.clf()
        plt.plot(current["timestamp"], current["latency_ms"])
        plt.title("Real-Time Log Stream (Latency)")
        plt.xlabel("Time")
        plt.ylabel("Latency (ms)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.pause(0.01)

        time.sleep(delay)

    plt.show()
    print("\nStreaming finished.")

if __name__ == "__main__":
    stream_logs()