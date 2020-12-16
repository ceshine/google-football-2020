# Solution to Google Research Football with Manchester City F.C.

In short: **MLP and XGB Imitator Models**.

[The link to the competition overview page](https://www.kaggle.com/c/google-football/).

## Commands

Scrape episodes:

```bash
python -m football.scraping.scraper
```

Preprocessing scraped episodes:

```bash
python -m football.scraping.preprocess
python -m football.scraping.create_pickle
```

Train an MLP model:

```bash
python -m football.train_mlp_imitator --epochs 100 --batch-size 256
```

Create the submission:

```bash
cp cache/model.pth scripts
cp cache/scaler.jbl scripts
cd scripts && stickytape mlp_submission.py > main.py
```

Test the submission locally:

```bash
cd scripts && python test.py
```
