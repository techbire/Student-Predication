"""
STEP 1: Generate 1,000 Student Records and Insert into MongoDB
==============================================================
This script creates a realistic dataset of 1,000 students and
stores them inside MongoDB using the PyMongo library.
"""

import numpy as np
import pandas as pd
from pymongo import MongoClient

def main():
    print("==================================================")
    print("   STEP 1: GENERATE DATASET & INSERT TO MONGODB   ")
    print("==================================================")
    
    # Set seed so dataset values are reproducible
    np.random.seed(42)
    num_records = 1000
    
    # Generate realistic student attributes
    student_ids = [f"STU{1001 + i}" for i in range(num_records)]
    study_hours = np.round(np.random.uniform(1.0, 9.5, num_records), 1)
    attendance = np.round(np.random.uniform(60.0, 99.0, num_records), 1)
    previous_marks = np.round(np.random.uniform(40.0, 95.0, num_records), 1)
    assignments_score = np.random.randint(4, 11, num_records)
    
    # Calculate realistic final marks with slight noise
    noise = np.random.normal(0, 2.5, num_records)
    final_marks = (
        0.45 * previous_marks +
        3.5 * study_hours +
        0.25 * attendance +
        1.5 * assignments_score +
        noise
    )
    final_marks = np.round(np.clip(final_marks, 30.0, 100.0), 1)
    
    # Combine into a Pandas DataFrame
    df = pd.DataFrame({
        "student_id": student_ids,
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignments_score": assignments_score,
        "final_marks": final_marks
    })
    
    print(f"[+] Generated {len(df)} realistic records.")
    
    # Save a CSV backup
    df.to_csv("student_data.csv", index=False)
    print("[+] Saved local CSV backup: 'student_data.csv'")
    
    # Connect to MongoDB
    print("[+] Connecting to MongoDB on localhost:27017...")
    client = MongoClient("mongodb://localhost:27017/")
    db = client["student_performance_db"]
    collection = db["students"]
    
    # Clear existing data to avoid duplicate inserts on re-runs
    collection.delete_many({})
    
    # Convert DataFrame to list of dictionaries for MongoDB insertion
    records = df.to_dict(orient="records")
    result = collection.insert_many(records)
    
    print(f"[SUCCESS] Inserted {len(result.inserted_ids)} documents into:")
    print("   Database   : student_performance_db")
    print("   Collection : students")
    
    client.close()

if __name__ == "__main__":
    main()
