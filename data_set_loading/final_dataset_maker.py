import pandas as pd
from newspaper import Article
from tqdm import tqdm
import os

# ---------------- CONFIG ----------------
TARGET_PER_CLASS = 250
URL_COLUMN = "news_url"

INPUT_FILES = {
    "politifact_real": r"C:\Users\shrin\Desktop\final_year_project\data\politifact_real.csv",
    "politifact_fake": r"C:\Users\shrin\Desktop\final_year_project\data\politifact_fake.csv",
    "gossipcop_real":  r"C:\Users\shrin\Desktop\final_year_project\data\gossipcop_real.csv",
    "gossipcop_fake":  r"C:\Users\shrin\Desktop\final_year_project\data\gossipcop_fake.csv",
}

OUTPUT_FILE = "final_1000_articles_dataset.csv"
# ---------------------------------------


def normalize_url(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return "http://" + url


def newspaper_download_success(url: str) -> bool:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return bool(article.text and article.text.strip())
    except Exception:
        return False


def collect_and_write(csv_path: str, label: str, write_header: bool):
    df = pd.read_csv(csv_path)
    kept = 0

    print(f"\nProcessing {label}...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if kept >= TARGET_PER_CLASS:
            break

        raw_url = row.get(URL_COLUMN)
        if not isinstance(raw_url, str) or raw_url.strip() == "":
            continue

        url = normalize_url(raw_url.strip())

        if newspaper_download_success(url):
            row = row.copy()
            row[URL_COLUMN] = url
            row["dataset_label"] = label

            # 🔹 write immediately
            pd.DataFrame([row]).to_csv(
                OUTPUT_FILE,
                mode="a",
                header=write_header,
                index=False
            )

            write_header = False
            kept += 1

    print(f"Kept {kept} rows for {label}")
    return write_header


def main():
    # remove old output file if exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    write_header = True

    for label, path in INPUT_FILES.items():
        write_header = collect_and_write(path, label, write_header)

    print("\n===================================")
    print(f"Saved dataset incrementally to: {OUTPUT_FILE}")
    print("===================================")


if __name__ == "__main__":
    main()
