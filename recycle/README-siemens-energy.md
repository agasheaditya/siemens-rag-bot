# Synthetic Defect Image Generation with Stable Diffusion

### Overview

This project focuses on generating synthetic defect images using Stable Diffusion , a state-of-the-art text-to-image diffusion model. The goal is to condition the generation process on both textual prompts (e.g., "a pill with a vertical scratch") and binary masks (e.g., regions indicating defects). This approach enables the creation of highly realistic synthetic images for use in industrial applications such as quality control, defect detection, and dataset augmentation.


Key Features:

- Fine-tuned Stable Diffusion model with LoRA (Low-Rank Adaptation) for efficient training.
- Support for conditional image generation using both prompts and masks.
- Modular codebase for easy customization and extension.

---
### Task Description

The primary task involves:
#### Training :
  * Fine-tuning a Stable Diffusion model to generate synthetic defect images conditioned on textual descriptions and binary masks.
  * Incorporating LoRA for parameter-efficient fine-tuning.
#### Inference :
  * Generating high-quality defect images from prompts alone or in combination with masks.
  * Ensuring compatibility with both masked and unmasked inputs during inference.

### Applications:

- Synthetic data generation for defect detection algorithms.
- Augmenting real-world datasets for improved model training.
- Visualizing potential defect scenarios for industrial design and testing.

### Technologies Used
This project leverages the following tools and libraries:

* **Stable Diffusion** : A powerful diffusion-based text-to-image generation model.
* **LoRA (Low-Rank Adaptation)** : Efficient fine-tuning technique to adapt pre-trained models with minimal additional parameters.
* **PyTorch** : Deep learning framework for model training and inference.
* **Diffusers Library** : Provides utilities for working with Stable Diffusion pipelines.
* **PEFT (Parameter-Efficient Fine-Tuning)** : Library for implementing LoRA and other parameter-efficient fine-tuning techniques.
* **CUDA and Mixed Precision** : Accelerates training and inference on NVIDIA GPUs.
* **safetensors** : Safe and efficient format for storing model weights.

---
### Setup and Installation
#### Prerequisites
* Hardware Requirements :
NVIDIA GPU with CUDA support (recommended: 16GB+ VRAM).
Sufficient disk space for model checkpoints and datasets.
* Software Requirements :
Python 3.8 or higher.
CUDA Toolkit (compatible with your GPU drivers).

### Setup and Usage: 
* Hardware used:
  - CPU: Intel i7-10750H (2.60 GHz)
  - RAM: 16 GB
  - GPU: NVIDIA GeForce RTX 2060 (6 GB)
    
* Create virtual environment
```code
virtualenv env
```

* Activate virtual environment
```code
./env/Scripts/activate
```

* Installing dependancies
```code
pip install -r requirements.txt
```
---
# Synthetic Defect Image Generation with Stable Diffusion

### Overview

This project focuses on generating synthetic defect images using Stable Diffusion , a state-of-the-art text-to-image diffusion model. The goal is to condition the generation process on both textual prompts (e.g., "a pill with a vertical scratch") and binary masks (e.g., regions indicating defects). This approach enables the creation of highly realistic synthetic images for use in industrial applications such as quality control, defect detection, and dataset augmentation.


Key Features:

- Fine-tuned Stable Diffusion model with LoRA (Low-Rank Adaptation) for efficient training.
- Support for conditional image generation using both prompts and masks.
- Modular codebase for easy customization and extension.

---
### Task Description

The primary task involves:
#### Training :
  * Fine-tuning a Stable Diffusion model to generate synthetic defect images conditioned on textual descriptions and binary masks.
  * Incorporating LoRA for parameter-efficient fine-tuning.
#### Inference :
  * Generating high-quality defect images from prompts alone or in combination with masks.
  * Ensuring compatibility with both masked and unmasked inputs during inference.

### Applications:

- Synthetic data generation for defect detection algorithms.
- Augmenting real-world datasets for improved model training.
- Visualizing potential defect scenarios for industrial design and testing.

### Technologies Used
This project leverages the following tools and libraries:

* **Stable Diffusion** : A powerful diffusion-based text-to-image generation model.
* **LoRA (Low-Rank Adaptation)** : Efficient fine-tuning technique to adapt pre-trained models with minimal additional parameters.
* **PyTorch** : Deep learning framework for model training and inference.
* **Diffusers Library** : Provides utilities for working with Stable Diffusion pipelines.
* **PEFT (Parameter-Efficient Fine-Tuning)** : Library for implementing LoRA and other parameter-efficient fine-tuning techniques.
* **CUDA and Mixed Precision** : Accelerates training and inference on NVIDIA GPUs.
* **safetensors** : Safe and efficient format for storing model weights.

---
### Setup and Installation
#### Prerequisites
* Hardware Requirements :
NVIDIA GPU with CUDA support (recommended: 16GB+ VRAM).
Sufficient disk space for model checkpoints and datasets.
* Software Requirements :
Python 3.8 or higher.
CUDA Toolkit (compatible with your GPU drivers).

### Setup and Usage: 
* Hardware used:
  - CPU: Intel i7-10750H (2.60 GHz)
  - RAM: 16 GB
  - GPU: NVIDIA GeForce RTX 2060 (6 GB)
    
* Create virtual environment
```code
virtualenv env
```

* Activate virtual environment
```code
./env/Scripts/activate
```

* Installing dependancies
```code
pip install -r requirements.txt
```
---

Once ready, you can run this notebook: https://github.com/agasheaditya/siemens-rag-bot/blob/main/siemens-energy/notebooks/P1.ipynb
