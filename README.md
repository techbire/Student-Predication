---
annotations_creators:
- machine-generated
language:
- en
license:
- mit
task_categories:
- tabular-regression
tags:
- education
- machine-learning
- regression
- student-performance
- mongodb
size_categories:
- 1K<n<10K
---

# Student Performance Prediction Using Python, MongoDB, and Scikit-Learn

A beginner-friendly Machine Learning project for B.Tech Computer Science and Engineering (CSE) students.

---

## 📌 Project Overview
- **Objective:** Predict students' final examination marks out of 100 based on study hours, attendance, previous marks, and assignment completion.
- **Database:** MongoDB (`student_performance_db`)
- **Dataset:** 1,000 realistic student records
- **Python Driver:** PyMongo
- **Machine Learning Model:** Linear Regression
- **Evaluation Metrics:** $R^2$ Score (96.1%) and Mean Absolute Error (2.04 marks)

---

## 📂 Project Structure
```text
ca2 mdb/
│
├── main.py                          # Complete all-in-one executable script
├── step1_insert_to_mongodb.py       # Step 1: Generate dataset & insert into MongoDB
├── step2_train_and_predict.py       # Step 2: Query MongoDB, train model, evaluate & predict
├── Student_Performance_Prediction.ipynb # Jupyter Notebook implementation
├── requirements.txt                 # Required Python packages
├── student_data.csv                 # Generated dataset backup
├── actual_vs_predicted.png          # High-resolution output evaluation plot
├── PROJECT_REPORT.md                # Full academic project report (8 sections)
├── VIVA_QUESTIONS.md                # Viva / oral exam preparation guide
└── README.md                        # Project instructions and quickstart
```

---

## ⚙️ Prerequisites & Installation

### 1. Ensure MongoDB Server is Running
If you have MongoDB Community Server installed on Windows:
- MongoDB runs as a background service on port `27017`.
- You can verify with PowerShell:
  ```powershell
  Get-Service -Name "*mongo*"
  ```
  If stopped, start it with:
  ```powershell
  net start MongoDB
  ```

### 2. Install Required Python Libraries
Open your terminal / command prompt in this project folder and run:
```bash
pip install -r requirements.txt
```
*(Or install individually: `pip install pymongo pandas numpy scikit-learn matplotlib`)*

---

## 🚀 How to Run the Project

### Option A: Run the Complete Pipeline (Recommended)
Run the all-in-one script:
```bash
python main.py
```
This script will:
1. Generate 1,000 realistic student records.
2. Connect to MongoDB and insert all 1,000 documents.
3. Retrieve records from MongoDB using PyMongo.
4. Display 5 sample records.
5. Preprocess the dataset and split it 80/20 into train/test sets.
6. Train a Linear Regression model.
7. Print $R^2$ Score, MAE, and RMSE.
8. Display a table of Actual vs Predicted Marks for sample students.
9. Save visualization graphs to `actual_vs_predicted.png`.
10. Test prediction on a custom student profile.

---

### Option B: Run Modular Scripts Step-by-Step

**Step 1: Populate MongoDB**
```bash
python step1_insert_to_mongodb.py
```

**Step 2: Train Model & Evaluate**
```bash
python step2_train_and_predict.py
```

---

### Option C: Jupyter Notebook / VS Code Interactive
Open `Student_Performance_Prediction.ipynb` in VS Code or Jupyter Lab, and run the cells sequentially.

---

## 📊 Expected Output Summary

```text
==================================================
STEP 3: FETCHING DATA FROM MONGODB USING PYMONGO
==================================================
-> Successfully fetched 1000 records from MongoDB.

--- 5 Sample Records Retreived from MongoDB ---
                     _id student_id  study_hours  attendance  previous_marks  assignments_score  final_marks
6ae78958eca66c4748b15e77    STU1001          4.2        67.2            54.4                  6         70.2
6ae78958eca66c4748b15e78    STU1002          9.1        81.1            53.6                  4         81.3
6ae78958eca66c4748b15e79    STU1003          7.2        94.0            89.8                  8        100.0
6ae78958eca66c4748b15e7a    STU1004          6.1        88.6            53.7                 10         84.1
6ae78958eca66c4748b15e7b    STU1005          2.3        91.5            55.0                  9         67.2

==================================================
STEP 6: MODEL EVALUATION METRICS
==================================================
1. R-Squared (R2) Score : 0.9610 (96.10% variance explained)
2. Mean Absolute Error  : 2.04 marks
3. Root Mean Sq. Error  : 2.53 marks

==================================================
STEP 7: ACTUAL VS PREDICTED RESULTS (FIRST 10 TEST STUDENTS)
==================================================
   Actual Marks  Predicted Marks  Difference (Error)
0          66.7             66.1                 0.6
1          86.9             89.4                 2.5
2          76.9             77.5                 0.6
3          73.3             76.3                 3.0
4          81.4             80.2                 1.2
5          86.6             83.3                 3.3
6          58.2             60.9                 2.7
7          83.9             81.6                 2.3
8          65.3             63.4                 1.9
9          69.4             72.5                 3.1
```
