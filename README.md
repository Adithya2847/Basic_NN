# PyTorch Model Checkpointing

A simple PyTorch project demonstrating **model checkpointing, saving/loading model weights, saving optimizer state, and resuming training**.

The project uses a small student-performance dataset to predict **exam scores** from:

* Study Hours
* Sleep Hours
* Attendance

The neural network is built using PyTorch and trained with the **AdamW optimizer** and **MSELoss**.

---

## Project Overview

During neural network training, we may want to save the model's progress so that:

* Training can be resumed later.
* The best-performing model can be saved.
* Training does not need to start from the beginning after interruption.
* Model parameters and optimizer information can be preserved.

This project demonstrates these concepts using PyTorch's:

```python
torch.save()
torch.load()
model.state_dict()
optimizer.state_dict()
model.load_state_dict()
optimizer.load_state_dict()
```

---

## Dataset

A small DataFrame is created containing student information.

| Feature       | Description                    |
| ------------- | ------------------------------ |
| `study_hours` | Number of hours spent studying |
| `sleep_hours` | Number of hours slept          |
| `attendance`  | Attendance percentage          |
| `exam_score`  | Student's exam score           |

### Example

```text
study_hours  sleep_hours  attendance  exam_score
2            6            65           50
3            7            70           58
4            7            75           65
5            8            80           72
...
```

The input features are:

```python
X = df.drop(['exam_score'], axis=1)
```

The target is:

```python
y = df['exam_score']
```

---

## Technologies Used

* Python
* PyTorch
* Pandas
* TensorDataset
* DataLoader
* AdamW Optimizer
* MSE Loss

---

## Neural Network Architecture

The model is a fully connected neural network:

```text
Input Layer
    ↓
Linear(3 → 64)
    ↓
ReLU
    ↓
Dropout(0.4)
    ↓
Linear(64 → 128)
    ↓
ReLU
    ↓
Dropout(0.4)
    ↓
Linear(128 → 64)
    ↓
ReLU
    ↓
Dropout(0.4)
    ↓
Linear(64 → 1)
    ↓
Predicted Exam Score
```

The model is created using `nn.Sequential`.

```python
model = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),

    nn.Dropout(p=0.4),
    nn.Linear(64, 128),
    nn.ReLU(),

    nn.Dropout(p=0.4),
    nn.Linear(128, 64),
    nn.ReLU(),

    nn.Dropout(p=0.4),
    nn.Linear(64, 1)
)
```

---

## Training

The model uses:

### Optimizer

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01
)
```

### Loss Function

```python
loss = nn.MSELoss()
```

The model is trained for:

```python
num_epochs = 50
```

The basic training process is:

```text
Forward Pass
     ↓
Calculate Loss
     ↓
Zero Gradients
     ↓
Backpropagation
     ↓
Optimizer Step
     ↓
Calculate Average Loss
```

---

# Model Checkpointing

## What is a Checkpoint?

A **checkpoint** is a saved snapshot of a model's training state.

Instead of saving only the neural network's weights, a training checkpoint can contain information such as:

```text
Model parameters
Optimizer state
Current epoch
Other training information
```

In this project, the checkpoint contains:

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optim_state_dict": optimizer.state_dict()
}
```

---

## Saving the Checkpoint

The checkpoint is saved using:

```python
torch.save(checkpoint, "checkpoint.pth")
```

This creates:

```text
checkpoint.pth
```

The checkpoint stores:

```text
checkpoint.pth
│
├── epoch
├── model_state_dict
└── optim_state_dict
```

---

## Why Save the Optimizer State?

Saving only the model weights is not always enough when we want to **resume training**.

Optimizers such as AdamW maintain internal information about previous gradients and parameter updates.

Therefore, the project saves:

```python
optimizer.state_dict()
```

along with:

```python
model.state_dict()
```

This allows training to continue with the optimizer's previous state.

---

# Loading a Checkpoint

A new model and optimizer are created:

```python
newer_model = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),

    nn.Dropout(p=0.4),
    nn.Linear(64, 128),
    nn.ReLU(),

    nn.Dropout(p=0.4),
    nn.Linear(128, 64),
    nn.ReLU(),

    nn.Dropout(p=0.4),
    nn.Linear(64, 1)
)
```

A new optimizer is also created:

```python
new_optimizer = torch.optim.AdamW(
    newer_model.parameters(),
    lr=0.01
)
```

The checkpoint is then loaded:

```python
checkpoint = torch.load("checkpoint.pth")
```

---

## Restoring Model Parameters

```python
newer_model.load_state_dict(
    checkpoint["model_state_dict"]
)
```

This restores the saved neural network parameters.

---

## Restoring Optimizer State

```python
new_optimizer.load_state_dict(
    checkpoint["optim_state_dict"]
)
```

This restores the optimizer's internal state.

---

## Restoring the Epoch

The saved epoch is retrieved using:

```python
start_epoch = checkpoint["epoch"]
```

This allows the program to know where the previous training process stopped.

---

# Best Model Saving

The project also saves the model's `state_dict`:

```python
torch.save(
    newer_model.state_dict(),
    "best_model.pth"
)
```

This creates:

```text
best_model.pth
```

Unlike the training checkpoint, this file contains only the model parameters.

---

## Checkpoint vs Best Model

There are two different concepts in this project.

### `checkpoint.pth`

Used for **resuming training**.

Contains:

```text
Model state
Optimizer state
Epoch
```

### `best_model.pth`

Used for storing the **model parameters**.

Contains:

```text
Model state
```

Conceptually:

```text
checkpoint.pth
        │
        ├── Model weights
        ├── Optimizer state
        └── Epoch
```

while:

```text
best_model.pth
        │
        └── Model weights
```

---

# Loading the Best Model

The saved model parameters can be loaded using:

```python
state_dict = torch.load("best_model.pth")

newer_model.load_state_dict(state_dict)
```

The model can then be used for evaluation or prediction.

---

# Inspecting the Saved Parameters

The project also demonstrates how to inspect the saved `state_dict`:

```python
for key, value in state_dict.items():
    print(key, value)
```

A PyTorch `state_dict` contains parameter names and their corresponding tensors.

For example:

```text
0.weight
0.bias
4.weight
4.bias
...
```

These represent the parameters of the layers in the neural network.

---

# Project Structure

```text
Basic_NN_save_structure.ipynb
│
├── Create Dataset
├── Convert DataFrame → Tensors
├── Create TensorDataset
├── Create DataLoader
├── Build Neural Network
├── Define AdamW Optimizer
├── Define MSE Loss
├── Train Model
├── Save Checkpoint
├── Create New Model
├── Load Checkpoint
├── Restore Optimizer
├── Resume Training
├── Save Best Model
└── Load Best Model
```

---

# Key PyTorch Concepts Learned

This project demonstrates the following important concepts:

* `TensorDataset`
* `DataLoader`
* `nn.Sequential`
* `nn.Linear`
* `nn.ReLU`
* `nn.Dropout`
* `MSELoss`
* `AdamW`
* Training loop
* `state_dict`
* `torch.save()`
* `torch.load()`
* Model checkpointing
* Optimizer checkpointing
* Resuming training
* Saving model weights
* Loading model weights

---

# Important Difference

### Saving only model weights

```python
torch.save(model.state_dict(), "model.pth")
```

This is useful when you only need the trained model for prediction.

### Saving a complete training checkpoint

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optim_state_dict": optimizer.state_dict()
}

torch.save(checkpoint, "checkpoint.pth")
```

This is useful when you want to **continue training later**.

---

# Learning Outcome

After completing this project, you should understand the basic idea of how deep-learning training can be saved and restored.

The main workflow is:

```text
Train Model
     ↓
Monitor Loss
     ↓
Save Checkpoint
     ↓
Training Stops
     ↓
Create/Load Model
     ↓
Load Checkpoint
     ↓
Restore Optimizer
     ↓
Resume Training
```

This is an important practical concept for larger deep-learning projects where training may take hours or days.

---

## Future Improvements

The project can be extended by adding:

* Train/validation split
* Validation loss
* Early stopping
* Learning-rate scheduler
* Saving the best validation model
* Test-set evaluation
* GPU training
* Prediction on new student data
* Checkpointing based on validation loss
* Training-loss visualization

---

## Files Generated

After running the notebook, the following files may be created:

```text
checkpoint.pth
best_model.pth
```

`checkpoint.pth` is intended for **resuming training**, while `best_model.pth` is intended for **loading the saved model parameters**.

---

## Author

**Adithya**

Learning PyTorch and Deep Learning through practical implementations.
