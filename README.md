# Replication study of "Data-Driven Methods for Balancing Fairness and Efficiency in Ride-Pooling"

This repository is the official implementation of Replication study of "Data-Driven Methods for Balancing Fairness and Efficiency in Ride-Pooling"

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```

## Training and Evaluation

To train and evaluate the model(s) in the paper, run this command:

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
