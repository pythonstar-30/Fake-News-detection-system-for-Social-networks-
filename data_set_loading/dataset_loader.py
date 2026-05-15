import requests
import os

urls = [
    "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_fake.csv",
    "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_real.csv",
    "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv",
    "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_real.csv"
]

for url in urls:
    filename = url.split("/")[-1]
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)


for fname in [
    "gossipcop_fake.csv",
    "gossipcop_real.csv",
    "politifact_fake.csv",
    "politifact_real.csv"
]:
    print(f"{fname}: {'✅ exists' if os.path.exists(fname) else '❌ missing'}")
