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
cp cache/final_weights.pth scripts/model.pth
cp cache/scaler.jbl scripts/
cd scripts && stickytape mlp_submission.py > main.py
```

Test the submission locally (play 5 games):

```bash
cd scripts && python test.py 5
```

Pack the files into one archive for submission:

```bash
cd scripts && zip submission.zip main.py model.pth scaler.jbl
```

## Docker Instructions

This repo comes with a Dockerfile that replicates the development environment.

Install Docker and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Then build the Docker image:

```bash
docker build -t football
```

Now you can run the commands in the above section inside a container (with GPU acceleration):

```bash
docker run --gpus all --shm-size 1g --rm -ti -v $(pwd):/src -w /src football bash
```

Since the submission won't have GPU acceleration available, you can run the test script in CPU mode:

```bash
docker run --rm -ti -v (pwd):/src -w /src/scripts football python test.py 5
```

The match videos will be located at a subfolder in `cache/runs`.
