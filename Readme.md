# Betterpipeline

Betterpipeline is a pipeline for webcrawler for identification of the site that are important to crawl, identification also means to eliminate scammy websites

# Demo

## ![demo](/images/demo.gif)

# Features

- Scamwebsite check using [dataset](https://www.kaggle.com/datasets/shivamb/spam-url-prediction) linear regression
- Weighted crawl using word embedding(vector database)
- Priority queue based on weight for efficient use of resource that important site get crawled first
- Everything containerized

# To run

```bash
git clone https://github.com/Aashish1-1-1/betterpipeline
sudo docker build -t betterpipeline .
sudo docker run -it betterpipeline
```

you can always change seed website(dockerfile) and depth(main.py) from dockerfile and main.py
