# Solution to Google Research Football with Manchester City F.C.

In short: **MLP and XGB Imitator Models**.

```bash
python -m football.scraping.scraper
```

```bash
python -m football.scraping.preprocess
python -m football.scraping.create_pickle
```

```bash
python -m football.train_mlp_imitator --epochs 100 --batch-size 256
```

```bash
stickytape mlp_submission.py > main.py
```
