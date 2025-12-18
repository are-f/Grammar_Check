# Spoken Grammar Scoring System (Audio → Score 1–5)

##  Project Overview
This project implements an end-to-end system to automatically evaluate the grammatical quality of spoken audio responses. Given a `.wav` audio file, the system predicts a grammar proficiency score on a scale of **1 to 5**, aligned with a predefined human scoring rubric. The solution leverages speech-to-text conversion and transformer-based natural language understanding, and is evaluated using **RMSE** and **Pearson Correlation**.

---

##  Approach
1. **Speech-to-Text (ASR):**  
   Audio recordings are transcribed using the pretrained **Whisper** model, preserving grammatical errors present in speech.

2. **Text Modeling:**  
   Transcriptions are tokenized and encoded using a pretrained **RoBERTa** transformer to capture sentence structure and grammatical patterns.

3. **Regression Head:**  
   A linear regression layer maps the transformer’s sentence representation to a continuous grammar score (1–5).

4. **Training & Evaluation:**  
   The model is trained using **Mean Squared Error (MSE)** loss and evaluated using **RMSE** and **Pearson Correlation** on a validation split.

---

##  Dataset Structure
    dataset/
    ├──  train/     # Training audio (.wav)
    ├── test/       # Test audio (.wav)
    └── csvs/  
         ├── train.csv # filename, label    
         └── test.csv  # filename


---

## Evaluation Metrics
- **RMSE (Root Mean Squared Error):** Measures prediction error magnitude  
- **Pearson Correlation:** Measures ranking and linear agreement between predictions and true scores  

> Training RMSE and validation metrics are explicitly reported in the notebook as required.

---

## Visualizations Included
- Training vs Validation RMSE per epoch  
- Training vs Validation Pearson per epoch  
- True vs Predicted scores (validation set)  
- Prediction distribution histograms  
- Error distribution and mean error per grammar level  

These visualizations help assess learning stability, generalization, and interpretability.

---

## How to Run
1. Install dependencies:
   ```bash
   pip install torch transformers openai-whisper librosa scikit-learn matplotlib
2. Place the dataset in the expected directory structure.

3. Run the Jupyter Notebook step by step.

4. Generate submission.csv.
## License
This project is intended for academic and research purposes. 
