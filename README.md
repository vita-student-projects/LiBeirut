# UniTraj

**A Unified Framework for Cross-Dataset Generalization of Vehicle Trajectory Prediction**

UniTraj is a framework for vehicle trajectory prediction, designed by researchers from VITA lab at EPFL. 
It provides a unified interface for training and evaluating different models on multiple dataset, and supports easy configuration and logging. 
Powered by [Hydra](https://hydra.cc/docs/intro/), [Pytorch-lightinig](https://lightning.ai/docs/pytorch/stable/), and [WandB](https://wandb.ai/site), the framework is easy to configure, train and logging.
In this project, we will be using UniTraj to train and evalulate a model we call PTR (predictive transformer) on the provided data accessible via SCITAS in `/work/vita/datasets/DLAV_unitraj`, which also includes the validation and testing sets. For this milestone, `test_easy` will be used to compute the predictions. 


## Our Task
The task is to complete the PTR model and train it on the provided data.

The model is a transformer-based model that takes the past trajectory of the vehicle and its surrounding agents, along with the map, and predicts the future trajectory.
![system](https://github.com/vita-epfl/unitraj-DLAV/blob/main/docs/assets/PTR.png?raw=true)
This is the architecture of the encoder part of model. Supposing we are given the past t time steps for M agents and we have a feature vector of size $d_K$ for each agent at each time step, the encoder part of the model consists of the following steps:
1. Add positional encoding to the input features at the time step dimension for distinguish between different time steps.
2. Perform the temporal attention to capture the dependencies between the trajectories of each agent separately.
3. Perform the spatial attention to capture the dependencies between the different agents at the same time step.
These steps are repeated L times to capture the dependencies between the agents and the time steps.

The model is implemented in `motionnet/models/ptr/ptr_model.py` and the config is in `motionnet/configs/method/ptr.yaml`. 

We are asked to complete three parts of the model in `motionnet/models/ptr/ptr_model.py`:
1. The `temporal_attn_fn` function that computes the attention between the past trajectory and the future trajectory.
2. The `spatial_attn_fn` function that computes the attention between different agents at the same time step.
3. The encoder part of the model in the `_forward` function. 

## Code Structure
There are three main components in UniTraj: dataset, model and config.
The structure of the code is as follows:
```
motionnet
├── configs
│   ├── config.yaml
│   ├── method
│   │   ├── ptr.yaml
├── datasets
│   ├── base_dataset.py
│   ├── ptr_dataset.py
├── models
│   ├── ptr
│   ├── base_model
├── utils
```
There is a base config, dataset and model class, and each model has its own config, dataset and model class that inherit from the base class.

**PTR model**

The training data consists of recorded trajectories of different vehicles over a certain period of time. In order to improve the base prediction performance of the transformer, it is crucial to consider both the spatial relation between the different agents, as well as the temporal evolution of the trajectory of each agent. To that end, the transformer combines temporal and spatial (also called social) attention based on the architecture found in https://arxiv.org/abs/2104.00563. 
The trajectories information are coded into an embedding matrix which will undergo a self-attention layer that enables the model to capture temporal dependencies and spatial correlations within the trajectories. 

**Temporal attention**

`agents_emb` is matrix representation of the cars' trajectories in a latent space. 
A fundamental step of the self-attention process for sequential data is the addition of positional encoding. In our application, the position encoding comes from the time-step of each trajectory point. Therefore, after applying a mask that discards unwanted and irrelevant data, we extract the time information of the embedding, and use the `pos_encoder()` function to append this information to the agent's embedding. We then pass the resulting matrix to the `layer()` function which computes the attention, giving us `agents_temp_emb`. Reshaping of embedding matrix before passing it to the `layer()` function is necessary to respect this function's shape requirements, and we later reshape back to its original dimensions. 

**Spatial attention**

Unlike temporal information, where the sequence order of the data-points naturally carries significant meaning, the order of the sequence representing the position of different agents during one time-step is irrelevant and provides no useful meaning. Consequently, no positional encoding is required when applying spatial attention. Similarly to the temporal attention function, we apply a mask to disregard unwanted data points, permute and reshape the embedding before passing it to `layer()` to get the spatial attention which we re-shape to format of the original embedding.

**_forward function**

This function represents the multi-head attention block that is used to predict the future trajectories of the agents. The fundamental part of the function is the sequential application of the temporal attention and social attention using `temporal_attn_fn()` and `social_attn_fn()` on the agents embedding, repeated L times. 

*NEW for MS2:* we modified the _forward function in ptr.py to add the noise and drift on the ego_in variable for data augmentation. This includes adding a decreasing shift over the first 5 steps, a constant shift, or Guassian noise, all of which are detailed in the report. When computing the predictions for the submission, this section has to be commented and no augmentation or modification should be done to the variables.


**Training and submission**

To run the training process we first need to define the path to the training and validation data in `motionnet/configs/config.yaml` and the hyperparameters in `motionnet/configs/method/ptr.yaml`. To start the training we use the following command:

```bash
python train.py method=ptr
```
which will generate a directory `lightning_logs/version_X` containing checkpoints and metrics describing the evolution of different errors in training and evaluation with epochs. To create the predictions, define the path to the model under the variable ckpt_path in the config.yaml file, change val_data_path to the testing set , then we run:

```bash
python generate_predictions.py method=ptr
```

and submit the resulting file to kaggle.

