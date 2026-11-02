"""
STEP 2: Fetch Data from MongoDB, Train ML Model, Evaluate & Predict
===================================================================
This script retrieves the 1,000 student documents from MongoDB using
PyMongo, performs preprocessing, splits into train/test, trains a
Linear Regression model, calculates R2 & MAE, and outputs comparisons.
"""

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def main():
    print("==================================================")
    print("   STEP 2: FETCH, PREPROCESS, TRAIN & EVALUATE    ")
    print("==================================================")
    
    # 1. Connect to MongoDB and fetch records
    print("[1] Connecting to MongoDB to fetch records...")
    client = MongoClient("mongodb://localhost:27017/")
    db = client["student_performance_db"]
    collection = db["students"]
    
    cursor = collection.find()
    df = pd.DataFrame(list(cursor))
    print(f"[+] Retrieved {len(df)} records from MongoDB.")
    
    print("\n--- 5 Sample Records ---")
    print(df.head(5).to_string(index=False))
    
    # 2. Data Preprocessing
    print("\n[2] Preprocessing Data...")
    # Drop MongoDB '_id' and 'student_id' because they are non-numerical identifiers
    df_clean = df.drop(columns=['_id', 'student_id'])
    
    # Check for missing values
    print("[+] Missing values check:\n", df_clean.isnull().sum())
    
    # Split into Features (X) and Target (y)
    X = df_clean[['study_hours', 'attendance', 'previous_marks', 'assignments_score']]
    y = df_clean['final_marks']
    
    # 80% Training and 20% Testing Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[+] Train dataset size : {len(X_train)} students (80%)")
    print(f"[+] Test dataset size  : {len(X_test)} students (20%)")
    
    # 3. Train Linear Regression Model
    print("\n[3] Training Linear Regression Model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[+] Model trained successfully!")
    
    # 4. Model Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n--- Model Evaluation Results ---")
    print(f"R-squared (R2) Score : {r2:.4f} ({r2*100:.2f}% accuracy)")
    print(f"Mean Absolute Error  : {mae:.2f} marks")
    print(f"Root Mean Sq. Error  : {rmse:.2f} marks")
    
    # 5. Display First 10 Sample Predictions
    print("\n--- Actual vs Predicted Marks (First 10 Test Students) ---")
    comparison = pd.DataFrame({
        "Actual Marks": y_test.values[:10],
        "Predicted Marks": np.round(y_pred[:10], 1),
        "Difference": np.round(np.abs(y_test.values[:10] - y_pred[:10]), 1)
    })
    print(comparison.to_string(index=True))
    
    # 6. Save Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, color='#2563EB', alpha=0.6, edgecolors='k', s=40)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#DC2626', linestyle='--', linewidth=2, label='Perfect Fit')
    plt.title("Actual vs Predicted Marks (Linear Regression)", fontweight='bold')
    plt.xlabel("Actual Marks")
    plt.ylabel("Predicted Marks")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    indices = np.arange(10)
    width = 0.35
    plt.bar(indices - width/2, y_test.values[:10], width, label='Actual', color='#3B82F6')
    plt.bar(indices + width/2, y_pred[:10], width, label='Predicted', color='#10B981')
    plt.title("Comparison for 10 Students", fontweight='bold')
    plt.xlabel("Student Index")
    plt.ylabel("Marks")
    plt.xticks(indices, [f"S{i+1}" for i in indices])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=300)
    plt.close()
    print("\n[+] Saved visualization plot as 'actual_vs_predicted.png'")
    
    # 7. Predict for a new student
    sample_student = pd.DataFrame([{
        'study_hours': 7.0,
        'attendance': 92.0,
        'previous_marks': 80.0,
        'assignments_score': 9
    }])
    prediction = model.predict(sample_student)[0]
    print(f"\n[+] Demo Prediction for Student with 7 hrs study, 92% attendance, 80 prev marks, 9/10 assignments:")
    print(f"    --> Predicted Final Marks: {prediction:.1f} / 100")
    
    client.close()

if __name__ == "__main__":
    main()
