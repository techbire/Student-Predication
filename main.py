"""
Student Performance Prediction - Machine Learning with MongoDB & Python
Complete End-to-End Project Script

Author: B.Tech CSE Student Project
Database: MongoDB (Localhost: 27017)
Library: PyMongo, Pandas, NumPy, Scikit-learn, Matplotlib
Algorithm: Linear Regression
"""

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: GENERATE REALISTIC DATASET (1,000 RECORDS)
# ==========================================
def generate_dataset(num_records=1000):
    print("\n" + "="*50)
    print("STEP 1: GENERATING 1,000 REALISTIC STUDENT RECORDS")
    print("="*50)
    
    # Set seed for reproducibility so outputs remain consistent
    np.random.seed(42)
    
    student_ids = [f"STU{1001 + i}" for i in range(num_records)]
    
    # Realistic feature distributions:
    # 1. Study Hours: between 1.0 to 10.0 hours daily
    study_hours = np.round(np.random.uniform(1.0, 9.5, num_records), 1)
    
    # 2. Attendance Percentage: between 60% and 99%
    attendance = np.round(np.random.uniform(60.0, 99.0, num_records), 1)
    
    # 3. Previous Exam Marks: between 40 and 95
    previous_marks = np.round(np.random.uniform(40.0, 95.0, num_records), 1)
    
    # 4. Assignments Completed: between 4 and 10
    assignments_score = np.random.randint(4, 11, num_records)
    
    # Realistic formula for Final Marks with slight realistic noise:
    # Final Marks depends on previous marks (40%), study hours (30%), attendance (20%), assignments (10%)
    noise = np.random.normal(0, 2.5, num_records)
    final_marks = (
        0.45 * previous_marks +
        3.5 * study_hours +
        0.25 * attendance +
        1.5 * assignments_score +
        noise
    )
    
    # Clip marks between 30 and 100
    final_marks = np.round(np.clip(final_marks, 30.0, 100.0), 1)
    
    df = pd.DataFrame({
        "student_id": student_ids,
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignments_score": assignments_score,
        "final_marks": final_marks
    })
    
    print(f"-> Successfully generated {len(df)} student records.")
    return df

# ==========================================
# STEP 2: STORE DATASET IN MONGODB
# ==========================================
def save_to_mongodb(df, db_name="student_performance_db", collection_name="students"):
    print("\n" + "="*50)
    print("STEP 2: CONNECTING TO MONGODB & INSERTING RECORDS")
    print("="*50)
    
    # Connect to local MongoDB instance
    client = MongoClient("mongodb://localhost:27017/")
    db = client[db_name]
    collection = db[collection_name]
    
    # Clean up old records if re-running
    collection.delete_many({})
    print("-> Cleared existing data in collection (if any).")
    
    # Convert DataFrame to dictionary format suitable for MongoDB
    records = df.to_dict(orient="records")
    
    # Insert all 1,000 records
    result = collection.insert_many(records)
    print(f"-> Connected to Database: '{db_name}'")
    print(f"-> Connected to Collection: '{collection_name}'")
    print(f"-> Successfully inserted {len(result.inserted_ids)} records into MongoDB!")
    
    client.close()

# ==========================================
# STEP 3: FETCH DATASET FROM MONGODB USING PYMONGO
# ==========================================
def fetch_from_mongodb(db_name="student_performance_db", collection_name="students"):
    print("\n" + "="*50)
    print("STEP 3: FETCHING DATA FROM MONGODB USING PYMONGO")
    print("="*50)
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client[db_name]
    collection = db[collection_name]
    
    # Retrieve all documents
    cursor = collection.find()
    
    # Convert MongoDB cursor directly to a Pandas DataFrame
    df = pd.DataFrame(list(cursor))
    print(f"-> Successfully fetched {len(df)} records from MongoDB.")
    
    # Display 5 sample records
    print("\n--- 5 Sample Records Retreived from MongoDB ---")
    print(df.head(5).to_string(index=False))
    
    client.close()
    return df

# ==========================================
# STEP 4: DATA PREPROCESSING
# ==========================================
def preprocess_data(df):
    print("\n" + "="*50)
    print("STEP 4: DATA PREPROCESSING")
    print("="*50)
    
    # 1. Drop MongoDB default '_id' and identifier 'student_id'
    # These are identifiers, not predictive numerical features
    cols_to_drop = [col for col in ['_id', 'student_id'] if col in df.columns]
    data = df.drop(columns=cols_to_drop)
    print(f"-> Removed non-predictive columns: {cols_to_drop}")
    
    # 2. Check for missing / null values
    null_counts = data.isnull().sum()
    print("-> Missing values in each column:")
    for col, count in null_counts.items():
        print(f"   - {col}: {count} missing")
        
    # 3. Separate features (X) and target label (y)
    # Features (Inputs): study_hours, attendance, previous_marks, assignments_score
    # Target (Output to predict): final_marks
    X = data.drop(columns=['final_marks'])
    y = data['final_marks']
    
    print("\n-> Features (Inputs - X):", list(X.columns))
    print("-> Target (Output to Predict - y): final_marks")
    
    # 4. Train-Test Split (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    print(f"\n-> Total records: {len(data)}")
    print(f"-> Training dataset size (80%): {X_train.shape[0]} records")
    print(f"-> Testing dataset size (20%):  {X_test.shape[0]} records")
    
    return X_train, X_test, y_train, y_test, X.columns

# ==========================================
# STEP 5: TRAIN MACHINE LEARNING MODEL
# ==========================================
def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    print("\n" + "="*50)
    print("STEP 5: TRAINING MACHINE LEARNING MODEL (LINEAR REGRESSION)")
    print("="*50)
    
    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Train (fit) the model on the training data
    model.fit(X_train, y_train)
    print("-> Model training complete!")
    
    # Display learned weights (coefficients)
    print("\n--- Learned Equation Parameters ---")
    for feat, coef in zip(feature_names, model.coef_):
        print(f"   Weight for '{feat}': {coef:.4f}")
    print(f"   Intercept (Base value): {model.intercept_:.4f}")
    
    # Make predictions on test dataset
    y_pred = model.predict(X_test)
    
    # ==========================================
    # STEP 6: EVALUATION METRICS
    # ==========================================
    print("\n" + "="*50)
    print("STEP 6: MODEL EVALUATION METRICS")
    print("="*50)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"1. R-Squared (R2) Score : {r2:.4f} ({r2*100:.2f}% variance explained)")
    print(f"2. Mean Absolute Error  : {mae:.2f} marks")
    print(f"3. Root Mean Sq. Error  : {rmse:.2f} marks")
    
    # ==========================================
    # STEP 7: ACTUAL VS PREDICTED COMPARISON
    # ==========================================
    print("\n" + "="*50)
    print("STEP 7: ACTUAL VS PREDICTED RESULTS (FIRST 10 TEST STUDENTS)")
    print("="*50)
    
    comparison_df = pd.DataFrame({
        "Actual Marks": y_test.values[:10],
        "Predicted Marks": np.round(y_pred[:10], 1),
        "Difference (Error)": np.round(np.abs(y_test.values[:10] - y_pred[:10]), 1)
    })
    print(comparison_df.to_string(index=True))
    
    # ==========================================
    # STEP 8: VISUALIZATION
    # ==========================================
    print("\n" + "="*50)
    print("STEP 8: SAVING VISUALIZATION PLOT")
    print("="*50)
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Actual vs Predicted Scatter Plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, color='#2563EB', alpha=0.6, edgecolors='k', s=40)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#DC2626', linestyle='--', linewidth=2, label='Perfect Fit (y=x)')
    plt.title("Actual Marks vs Predicted Marks", fontsize=12, fontweight='bold')
    plt.xlabel("Actual Final Marks", fontsize=10)
    plt.ylabel("Predicted Final Marks", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Subplot 2: Bar Chart for first 10 students
    plt.subplot(1, 2, 2)
    indices = np.arange(10)
    bar_width = 0.35
    plt.bar(indices - bar_width/2, y_test.values[:10], bar_width, label='Actual', color='#3B82F6')
    plt.bar(indices + bar_width/2, y_pred[:10], bar_width, label='Predicted', color='#10B981')
    plt.title("Comparison for 10 Students", fontsize=12, fontweight='bold')
    plt.xlabel("Sample Student Index", fontsize=10)
    plt.ylabel("Marks", fontsize=10)
    plt.xticks(indices, [f"S{i+1}" for i in indices])
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_file = "actual_vs_predicted.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"-> Plot successfully saved as '{plot_file}'")
    
    # ==========================================
    # STEP 9: SAMPLE CUSTOM PREDICTION
    # ==========================================
    print("\n" + "="*50)
    print("STEP 9: TESTING PREDICTION WITH A NEW CUSTOM STUDENT")
    print("="*50)
    # Example student: study_hours=6.5, attendance=88.0, previous_marks=78.0, assignments_score=9
    new_student = pd.DataFrame([{
        "study_hours": 6.5,
        "attendance": 88.0,
        "previous_marks": 78.0,
        "assignments_score": 9
    }])
    predicted_score = model.predict(new_student)[0]
    print("Input Features for New Student:")
    print(" - Daily Study Hours: 6.5 hrs")
    print(" - Attendance: 88.0%")
    print(" - Previous Exam Marks: 78.0 / 100")
    print(" - Assignments Score: 9 / 10")
    print(f"-> Predicted Final Marks: {predicted_score:.1f} / 100")
    print("="*50 + "\n")
    
    return model

# ==========================================
# MAIN EXECUTION CONTROLLER
# ==========================================
if __name__ == "__main__":
    # 1. Create dataset
    raw_df = generate_dataset(num_records=1000)
    
    # Also save to CSV for local inspection
    raw_df.to_csv("student_data.csv", index=False)
    print("-> Saved backup copy as 'student_data.csv'")
    
    # 2. Store to MongoDB
    save_to_mongodb(raw_df)
    
    # 3. Read back from MongoDB
    df_from_db = fetch_from_mongodb()
    
    # 4. Preprocess data
    X_train, X_test, y_train, y_test, features = preprocess_data(df_from_db)
    
    # 5. Train & evaluate model
    model = train_and_evaluate(X_train, X_test, y_train, y_test, features)
    print("ALL PROJECT STEPS EXECUTED SUCCESSFULLY!")
