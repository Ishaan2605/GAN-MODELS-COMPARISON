# 🔍 Project Description: GAN-Models-Comparison

This project presents a comprehensive comparison of three popular Generative Adversarial Network (GAN) architectures – LSGAN, WGAN, and WGAN-GP – using the MedMNIST (PathMNIST) dataset for medical image generation.

The aim is to explore how different loss functions and training strategies affect the quality, stability, and realism of generated images.

We train and evaluate each model under the same settings and visualize results through TensorBoard, while also reporting quantitative metrics including:

Inception Score (IS)

Fréchet Inception Distance (FID)

The project provides clear insights into how different GAN variants perform on the same dataset, making it ideal for academic study, benchmarking, or practical GAN experimentation.

✨ Features
✅ Fully implemented LSGAN, WGAN, and WGAN-GP from scratch using PyTorch

✅ TensorBoard integration for real-time image & metric comparison

✅ Quantitative evaluation using IS and FID

✅ Lightweight implementation optimized for MedMNIST dataset

✅ Clean modular code structure with separate files for models, training, metrics, and visualizations

## 📦 Project Structure

```
GAN-ASSIGNMENT/
│
├── lsgan.py            # LSGAN implementation
├── wgan.py             # WGAN implementation
├── wgan_gp.py          # WGAN-GP implementation
│
├── utils/
│   ├── metrics.py      # Metrics for IS and FID
│   └── visualize.py    # Image saving and TensorBoard image logging
│
├── datasets/
│   └── load_medmnist.py  # MedMNIST dataset loader
│
├── generated_images/   # Output sample images from each GAN

```

---

## 🧪 Dataset

- **PathMNIST** from MedMNIST: a dataset of histopathological images from colorectal cancer.
- Input images are resized to **28×28** with **3 color channels**.

---

## ⚙️ Models Overview

| Model     | Discriminator | Generator | Key Difference     |
|-----------|---------------|-----------|---------------------|
| **LSGAN** | Uses MSE loss | MLP       | Uses Least Squares instead of BCE loss |
| **WGAN**  | Critic only   | MLP       | Uses Wasserstein loss with weight clipping |
| **WGAN-GP** | Critic       | MLP       | Uses Gradient Penalty to enforce Lipschitz constraint |

---

## 📊 Quantitative Results

| Model     | Inception Score ↑ | FID Score ↓ |
|-----------|-------------------|-------------|
| **LSGAN** | 1.2369 ± 0.0495   | 11.0367     |
| **WGAN**  | 1.6616 ± 0.0388   | 8.3756      |
| **WGAN-GP** | 1.4208 ± 0.0381 | 0.1080|

✅ **WGAN-GP** produces the most realistic samples with the lowest FID, making it the best performer in this experiment.

---

## 🖼️ Sample Outputs

### 📍 TensorBoard Generated Samples
![Screenshot 2025-04-10 093123](https://github.com/user-attachments/assets/47cebb43-3a0d-4e46-abeb-68edf4ac6cb2)


- From left to right: LSGAN → WGAN → WGAN-GP
- WGAN-GP images show better texture and less noise.

---

## 🧪 Console Evaluation Summary

```
[LSGAN] Inception Score: 1.2369 ± 0.0495
[LSGAN] FID Score: 11.0367


[WGAN] Inception Score: 1.6616 ± 0.0388
[WGAN] FID Score: 8.3756



[WGAN-GP] Inception Score: 1.4208 ± 0.0381
[WGAN-GP] FID Score: 0.1080

```





## 📈 TensorBoard Usage
```
Start visualization by running:

bash
tensorboard --logdir=runs/



- Compare `runs/lsgan`, `runs/wgan`, and `runs/wgan_gp`
- Visually inspect sample generations and losses


## 🧪 Metrics Used

- **Inception Score (IS)**: Evaluates clarity and classifiability of images.
- **Fréchet Inception Distance (FID)**: Measures similarity between real and generated distributions.
```



## 🚀 How to Run
```
Make sure to install all dependencies from requirements.txt :
pip install -r requirements.txt

# Run any model
python lsgan.py
python wgan.py
python wgan_gp.py

```


## 📬 Acknowledgements

- [MedMNIST: A Lightweight Benchmark for Medical Image Classification](https://medmnist.com/)
- [TorchMetrics](https://torchmetrics.readthedocs.io/)
- [Pytorch GAN Examples](https://github.com/eriklindernoren/PyTorch-GAN)

---

## 🧠 Author

**Ishaan Deshpande**  
GAN Architectures | Metric Evaluation | TensorBoard Logging

---

