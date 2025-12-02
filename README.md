# GlucoseBERT: Self-Supervised Learning for Glucose Forecasting

**GlucoseBERT** is a Transformer-based **Self-Supervised Learning** model designed to learn the "grammar" of glucose dynamics from Continuous Glucose Monitoring (CGM) data.

By using **Masked Pre-training** (similar to BERT in NLP), the model learns to reconstruct missing segments of glucose history, capturing complex physiological patterns without needing labeled future data. This pre-trained encoder is then fine-tuned for clinical forecasting tasks, such as predicting hypoglycemia 30 minutes in advance.

## Methodology

### 1. Masked Pre-training (Self-Supervised)
We treat glucose readings as a sequence of tokens. During pre-training:
-   **Input**: A 2-hour window of CGM data (96 steps) with 15% of time steps randomly masked.
-   **Objective**: The model must reconstruct the original glucose values for the masked steps.
-   **Architecture**: A 3-layer Transformer Encoder with Multi-Head Self-Attention.
-   **Result**: The model learns a robust internal representation of glucose dynamics (e.g., post-prandial spikes, dawn phenomenon).

### 2. Fine-tuning (Forecasting)
We freeze the pre-trained encoder and attach a lightweight linear head.
-   **Task**: Predict the glucose level 30 minutes into the future (t+30).
-   **Benefit**: The model achieves high accuracy with less labeled data compared to training from scratch.

## Installation

```bash
git clone https://github.com/Zeesky-code/glucose-bert.git
cd glucose-bert
pip install torch pandas numpy matplotlib
```

## Usage

### 1. Pre-training
Train the GlucoseBERT encoder on your unlabeled CGM data.
```bash
python glucose_bert.py
```
*Outputs: `glucose_bert_pretrained.pth`*

### 2. Fine-tuning
Fine-tune the model for the forecasting task.
```bash
python glucose_bert_finetune.py
```
*Outputs: `glucose_forecaster.pth`*

### 3. Evaluation
Generate forecast plots and metrics.
```bash
python evaluate_glucose_bert.py
```
*Outputs: Forecast plots in `../analysis_results`*

## Results
The model successfully predicts future glucose trends, capturing both rapid drops and post-meal rises. (See `analysis_results` for generated plots).

## License
MIT