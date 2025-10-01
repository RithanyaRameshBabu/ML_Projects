# 🏡 California House Price Prediction (Machine Learning Project)

This project predicts **house prices in California** using the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).  
The model uses **XGBoost Regressor** and includes data exploration, visualization, model training, and evaluation.

Project is hosted on my GitHub: [ML_Projects](https://github.com/RithanyaRameshBabu/ML_Projects)

---

## 📂 Project Workflow
1. **Dataset Loading**  
   - Used `fetch_california_housing()` from `sklearn.datasets`.

2. **Exploratory Data Analysis (EDA)**  
   - Distribution of house prices.  
   - Relationship between Median Income and House Prices.  
   - Correlation heatmap for all features.

3. **Model Training**  
   - Train/test split (80/20).  
   - Applied `XGBRegressor` for prediction.  

4. **Evaluation Metrics**  
   - R² Score  
   - Mean Absolute Error (MAE)  
   - Visual comparison between actual vs predicted prices.  

---

## 📊 Example Visualizations
- House price distribution  
- Median income vs house prices scatter plot  
- Correlation heatmap  
- Actual vs Predicted house prices  

---

## 🚀 Technologies Used
- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

---

## 📌 Results
- The trained model achieves a reasonable accuracy for predicting house prices.  
- Shows clear relationship between **Median Income** and **Price**.  

---

## ▶️ How to Run
```bash
# Clone repository
git clone https://github.com/RithanyaRameshBabu/ML_Projects.git
cd ML_Projects/california-house-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run the script
python house_price_prediction.py
