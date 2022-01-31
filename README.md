# Replication study of "Data-Driven Methods for Balancing Fairness and Efficiency in Ride-Pooling"

This repository is the official implementation of Replication study of "Data-Driven Methods for Balancing Fairness and Efficiency in Ride-Pooling"

## Requirements

To install requirements for simulating the taxi environment, training the location embeddings, or training the neural value function:

```setup
conda env create -f environment.yml
```

To install requirements for preprocessing raw New York taxi data, obtaining a graph for (part of) a city using OpenStreetMap, or obtaining corresponding travel time and shortest path files:

```setup
conda env create -f preprocessing/environment.yml
```

## Training and Evaluation

To train and evaluate the model(s) in the paper, run these commands:

```python
# driver side fairness objective
python main_pytorch.py training_days 3 testing_days 1 num_agents 200 value_num 10 write_file True print_verbose False lambda 0.67 data_dir "'../data/ny/'"
# income objective
python main_pytorch.py training_days 3 testing_days 1 num_agents 200 value_num 15 write_file True print_verbose False data_dir "'../data/ny/'"
# rider side fairness objective
python main_pytorch.py training_days 2 testing_days 1 num_agents 200 value_num 14 write_file True print_verbose False lambda 1000000000 data_dir "'../data/ny/'"
# requests objective
python main_pytorch.py training_days 3 testing_days 1 num_agents 200 value_num 1 write_file True print_verbose False data_dir "'../data/ny/'"
```

## Results

[The figures required to support the claims:](generate_plots.ipynb)


![image1](/images/50_driv_orig.png)
![image2](/images/200_driv_orig.png)
![image3](/images/200_driv_tot_inc_orig.png)
![image4](/images/200_driv_inc_orig.png)
