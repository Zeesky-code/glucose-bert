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

### 3.  **Evaluation & Explainability**:
    ```bash
    python evaluate_glucose_bert.py
    python explain_predictions.py
    ```
*Outputs: Forecast plots in `../analysis_results`, Explainability plots in `explainability_results`*

## Explainable AI
The `explain_predictions.py` script uses gradient-based attribution to explain which parts of the 2-hour glucose history contribute most to each forecast. This helps clinicians understand *why* the model makes specific predictions.

## Results
The model successfully predicts future glucose trends. Explainability analysis reveals that the most recent glucose values (15-30 minutes ago) have the highest influence on predictions, which aligns with clinical knowledge.

## License
MIT