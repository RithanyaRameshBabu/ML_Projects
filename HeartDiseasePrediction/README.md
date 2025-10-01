# Heart Disease Prediction 🫀

This project uses **Machine Learning (Support Vector Machine - SVM)** to predict the presence of **heart disease** based on patient health data.

---

## 📊 Dataset

- **File:** `heart_v2.csv`
- **Features:** Age, Sex, Blood Pressure, Cholesterol, etc.
- **Target Column:** `heart disease`
  - `0` → No Heart Disease  
  - `1` → Heart Disease

> Dataset can be placed in the `data/` folder, or you can download it from [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease).

---

## ⚙️ Project Workflow

1. Load and explore the dataset
2. Data preprocessing & standardization
3. Train/Test split (80/20)
4. Model training → **SVM (linear kernel)**
5. Model evaluation using accuracy score
6. Predictive system for new patient data

---

## 🚀 How to Run

Clone the repository and install dependencies:

```bash
git clone https://github.com/RithanyaRameshBabu/ML_Projects.git
cd ML_Projects/HeartDiseasePrediction
pip install -r requirements.txt
