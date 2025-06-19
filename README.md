# 🛒 Shopping Behavior Predictor  
**By Swayam Sharma**

This project is a machine learning application that uses **K-Nearest Neighbors** (K=1) to predict whether a customer will complete a purchase based on their browsing behavior. Built as part of the **CS50 AI course**, it explores customer intent using clickstream data from an e-commerce website.

---

## 📂 Project Structure
- **shopping.py** – Core script: loads data, trains model, evaluates performance.
- **shopping.csv** – Raw data from 12,000+ customer sessions.
- **README.md** – You’re reading it.

---

## 🧠 How It Works
1. **Data Preprocessing**  
   - Converts months and boolean fields to numerical form.
   - Encodes visitor type and weekend values as 0/1.
2. **Model Training**  
   - Uses `scikit-learn`'s `KNeighborsClassifier` with `k=1`.
   - Splits data into training/testing sets (60/40).
3. **Evaluation**  
   - Reports **sensitivity** (true positive rate) and **specificity** (true negative rate).

---

## 📊 Example Output
```bash
$ python shopping.py shopping.csv
Correct: 4088
Incorrect: 844
True Positive Rate: 41.02%
True Negative Rate: 90.55%
```

---

## 🛠️ Technologies Used
- Python 3.12
- scikit-learn
- CSV Parsing

---

## 🔍 Key Learnings
Through this project, I understood how user interaction metrics can be used to build predictive models for real-world business problems like revenue forecasting in e-commerce.

---

## 💡 Future Work
- Try alternative classifiers like Decision Trees or Logistic Regression.
- Incorporate more behavioral features like mouse movement or time of day.