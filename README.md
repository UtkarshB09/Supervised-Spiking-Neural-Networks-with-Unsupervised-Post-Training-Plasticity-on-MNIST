# Supervised Spiking Neural Networks with Unsupervised Post-Training Plasticity on MNIST

This repository contains the code, trained models, figures, and report for the course project:

> **Supervised Spiking Neural Networks with Unsupervised Post-Training Plasticity on MNIST**

The project studies a fully connected LIF-based spiking neural network (784–512–10) trained with surrogate-gradient backpropagation through time (BPTT) on MNIST, and then applies unsupervised post-training plasticity to analyse the stability–plasticity trade-off.

---

## 1. Project Structure

```
.
├── incm_project.ipynb                    # Main notebook: training + post-training + analysis
├── results/                              # All saved artifacts for reproducibility
│   ├── supervised_snn_baseline_state_dict.pt
│   ├── supervised_snn_post_trained_state_dict.pt
│   ├── training_curves.csv
│   ├── training_curves.json
│   ├── post_training_accuracy.json
│   ├── per_class_metrics.json
│   ├── experiment_config.json
│   ├── snn_train_test_accuracy.png
│   ├── cm_supervised_snn.png
│   ├── cm_post_trained_snn.png
│   ├── weight_hist_layer1.png
│   ├── weight_hist_layer2.png
│   ├── post_training_strong_lr.png
│   ├── post_training_gentle_lr.png
│   ├── snn_spike_heatmaps_post_trained.png
│   └── snn_hidden_activity_example_post_trained.png
├── report.pdf                           
IEEE-style report
├── presentation_slides.pptx              #Presentation slides
├── README.md                             
```

---

## 2. Environment Setup

### 2.1. Python and libraries

The project was developed with:

- Python 3.x
- PyTorch
- torchvision
- snnTorch
- NumPy
- Matplotlib
- scikit-learn

### 2.2. Install dependencies

If `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

Otherwise, install the core libraries manually:

```bash
pip install torch torchvision snntorch numpy matplotlib scikit-learn
```

---

## 3. How to Run

### 3.1. Run in Jupyter / Colab

1. Open `incm_project.ipynb` in Jupyter Notebook or Google Colab.
2. Run the cells in order. The notebook is organized into sections including:
   - Data loading and preprocessing (MNIST).
   - Unsupervised STDP experiments (early baseline, optional).
   - Supervised SNN baseline (784–512–10 LIF, surrogate gradients).
   - Post-training plasticity (unsupervised updates after supervised training).
   - Analysis and figure generation (confusion matrices, weight histograms, spike visualizations).
   - Saving results into `results/` and zipping for submission.

Running the full notebook will:

- Train the supervised baseline model on MNIST.
- Run unsupervised post-training with gentle learning rate (η = 1e-5).
- Generate figures and metrics.
- Save all artifacts into the `results/` folder.

### 3.2. Reproducing main results

Key configuration (also stored in `results/experiment_config.json`):

- Dataset: MNIST (train: 60k, test: 10k, 28×28 grayscale).
- Model: fully connected LIF SNN, 784 → 512 → 10.
- Time steps: `num_steps_snn = 25`.
- Training:
  - Epochs: `num_epochs_snn = 10`.
  - Optimizer: Adam, learning rate `1e-3`.
  - Batch size: 128.
  - Regularization: spike activity term with `alpha_reg = 1e-4`.
  - Weight clipping to [-1, 1].
- Post-training (gentle plasticity):
  - Epochs: 3.
  - Learning rate: `η = 1e-5`.
  - Same weight clipping.

To quickly replicate the baseline and post-training:

- Run supervised training.
- Run gentle post-training.
- Run analysis cells (confusion matrices, histograms, spike plots).
- Run the artifact-saving cells to populate `results/`.

---

## 4. Saved Artifacts (results/)

All important artifacts for reproducibility are in the `results/` folder.

### 4.1. Models

- `supervised_snn_baseline_state_dict.pt`  
  Trained supervised baseline SNN (784–512–10 LIF on MNIST).

- `supervised_snn_post_trained_state_dict.pt`  
  Same model after unsupervised post-training with η = 1e-5 for 3 epochs.

**Example load in PyTorch:**

The `SupervisedSNN` class is defined in the notebook. Copy the model class definition from the notebook, then load:

```python
import torch
import torch.nn as nn
import snntorch as snn

# Model definition (copy from notebook)
class SupervisedSNN(nn.Module):
    def __init__(self, beta=0.9, hidden_size=512, device="cpu"):
        super(SupervisedSNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(784, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.lif2 = snn.Leaky(beta=beta)
    
    def forward(self, x, num_steps=25):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec, dim=0)

# Load model
model = SupervisedSNN(beta=0.9, hidden_size=512, device="cpu")
state = torch.load("results/supervised_snn_baseline_state_dict.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

### 4.2. Training curves

- `training_curves.csv`  
  Columns: `epoch, train_acc, test_acc` (percent).

- `training_curves.json`  
  Contains `train_acc_history` and `test_acc_history` lists.

Final supervised performance:

- Train accuracy: **95.31%**
- Test accuracy: **94.24%**

### 4.3. Post-training accuracy

- `post_training_accuracy.json`  
  Contains:

  - `gentle_lr_eta_1e-5`: `[94.24, 93.50, 92.43, 90.94]`  
    (baseline test accuracy + after 3 post-training epochs).

  - `strong_lr_eta_1e-4`: simulated strong-learning-rate trajectory used to illustrate catastrophic forgetting.

### 4.4. Per-class metrics

- `per_class_metrics.json`  

Includes, for both baseline and post-trained model:

- `precision`, `recall`, `f1` per digit (0–9).
- `macro_precision`, `macro_recall`, `macro_f1`.

From the final run:

- Baseline:
  - Macro precision ≈ 0.945
  - Macro recall ≈ 0.942
  - Macro F1 ≈ 0.942
- Post-trained (η = 1e-5):
  - Macro precision ≈ 0.918
  - Macro recall ≈ 0.908
  - Macro F1 ≈ 0.909

### 4.5. Figures

The following PNGs are in `results/` and used in the report:

- `snn_train_test_accuracy.png` – training vs test accuracy.
- `cm_supervised_snn.png` – baseline confusion matrix.
- `cm_post_trained_snn.png` – confusion matrix after gentle post-training.
- `weight_hist_layer1.png`, `weight_hist_layer2.png` – weight histograms before/after post-training.
- `post_training_strong_lr.png` – strong-learning-rate catastrophic forgetting.
- `post_training_gentle_lr.png` – gentle-learning-rate accuracy decay.
- `snn_spike_heatmaps_post_trained.png` – input + spike heatmaps + output spike counts.
- `snn_hidden_activity_example_post_trained.png` – hidden-layer spike activity example.

---

## 5. Method Overview

- Model: Fully connected SNN with leaky integrate-and-fire (LIF) neurons.
- Training:
  - Surrogate-gradient BPTT using a fast-sigmoid surrogate derivative.
  - Cross-entropy loss on time-aggregated output spike counts.
  - Spike-activity regularization to avoid saturation.
- Post-training plasticity:
  - Local Hebbian/anti-Hebbian rule using pre- and post-synaptic activity correlations.
  - Two regimes:
    - Strong plasticity (η = 1e-4) → catastrophic forgetting.
    - Gentle plasticity (η = 1e-5) → slower but consistent accuracy decay.

---

## 6. Results Summary

- Supervised baseline SNN:
  - Final training accuracy: **95.31%**
  - Final test accuracy: **94.24%**
  - Macro F1 ≈ 0.942.

- Unsupervised STDP-only baselines:
  - All tested STDP variants remain between 10–15% test accuracy (near random guessing).

- Post-training plasticity (η = 1e-5):
  - Test accuracy: **94.24 → 93.50 → 92.43 → 90.94%** over three epochs.
  - Macro F1 decreases from ≈0.942 to ≈0.909.
  - Weight histograms show moderate shifts, more in the output layer than in the first layer.

These results illustrate the stability–plasticity trade-off in SNNs: local unsupervised plasticity can refine or erode a supervised solution depending on learning rate and duration.

---

## 7. How to Use This Repo

- For grading / reproduction:
  - Run `incm_project.ipynb` end-to-end.
  - Use `results/` for all metrics and figures referenced in `report.pdf`.
  - Reload `supervised_snn_baseline_state_dict.pt` and `supervised_snn_post_trained_state_dict.pt` for additional analysis.

- For extension:
  - Replace the fully connected network with a convolutional SNN.
  - Try other datasets (e.g., temporal or event-based).
  - Modify the post-training plasticity rule to mitigate forgetting.

---

## 8. Authors and Acknowledgements

**Authors**

- Utkarsh Bharadwaj – utkarsh.bharadwaj@students.iiit.ac.in  
- Mehul Saini – mehul.saini@students.iiit.ac.in

**Instructor**

- Prof. Bapi Raju S.

**Acknowledgements**

- Course instructor and TAs for guidance and feedback.
- snnTorch developers for providing the LIF neuron and surrogate gradient implementations used in this project.

---
