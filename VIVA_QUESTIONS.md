# College Viva & Evaluation Q&A Guide
## Student Performance Prediction Using Python, MongoDB & ML

This guide prepares you for common oral questions asked by university faculty and external examiners.

---

### Q1: What is the main objective of your project?
**Answer:**  
"The objective is to predict a student's final examination marks based on academic parameters like study hours, attendance, previous marks, and assignment completion. We store 1,000 student records in a MongoDB NoSQL database, retrieve them using PyMongo in Python, preprocess the features, and train a Linear Regression model to make predictions."

---

### Q2: Why did you use MongoDB instead of MySQL or a CSV file?
**Answer:**  
"1. MongoDB is a modern NoSQL document database that stores records as BSON (JSON-like documents), which maps naturally to Python dictionaries and Pandas DataFrames.  
2. It offers high write/read speed and flexible schema design, making it suitable for modern web applications and educational management systems where student attributes can evolve over time.  
3. In a real-world production environment, data resides in databases rather than static CSV files."

---

### Q3: What is PyMongo and how does it connect Python to MongoDB?
**Answer:**  
"PyMongo is the official Python client library for MongoDB. We use `MongoClient('mongodb://localhost:27017/')` to connect to the local MongoDB server on standard port `27017`. We then access our database `student_performance_db` and collection `students` just like Python dictionaries."

---

### Q4: Why did you choose Linear Regression for this project?
**Answer:**  
"Linear Regression is a supervised learning algorithm designed for predicting continuous numerical values (final marks from 0 to 100). It finds the best linear relationship ($y = w_1 x_1 + w_2 x_2 + \dots + b$) that minimizes the error between actual marks and predicted marks. It is simple, fast, highly interpretable, and allows us to clearly understand how each factor (like study hours) directly contributes to the student's score."

---

### Q5: What preprocessing steps did you perform on the data?
**Answer:**  
"1. We dropped `_id` (the MongoDB primary key) and `student_id` because they are unique non-predictive identifiers that shouldn't influence mathematical predictions.  
2. We verified there were no missing or null values using `df.isnull().sum()`.  
3. We separated independent input features ($X$) from the dependent target label ($y$).  
4. We performed an 80-20 train-test split using `train_test_split(..., test_size=0.2, random_state=42)`."

---

### Q6: Why did you split the dataset into 80% training and 20% testing?
**Answer:**  
"To prevent overfitting and test generalization. If we test the model on the same data it was trained on, it might simply memorize the answers. The 80% (800 records) is used to learn weights, and the 20% (200 records) acts as unseen real-world student data to fairly evaluate true prediction accuracy."

---ca2 mdb

### Q7: What evaluation metrics did you use, and what do they indicate?
**Answer:**  
"We used:
1. **$R^2$ Score (Coefficient of Determination): ~0.961 (96.1%)**  
   It indicates that 96.1% of the variation in final marks is accurately explained by the student's study hours, attendance, previous marks, and assignment score.
2. **Mean Absolute Error (MAE): ~2.04 marks**  
   It represents the average absolute difference between the predicted mark and the actual mark. On average, our prediction is off by only about 2 marks out of 100."

---

### Q8: What do the coefficients (weights) tell us about student success?
**Answer:**  
"In our trained model:
- The coefficient for **daily study hours** is around `+3.47`, meaning every additional hour of focused daily study increases the predicted final exam score by nearly 3.5 marks.
- **Previous marks** carry a weight of `+0.44`, showing that past academic foundation strongly anchors performance.
- **Attendance** and **assignments** add positive linear increments, showing that regular class participation steadily improves scores."

---

### Q9: Can this model be used for Classification (e.g. Pass/Fail)?
**Answer:**  
"Yes. If the problem is converted to classification (e.g. Pass if Final Marks $\ge$ 50, otherwise Fail), we can use **Logistic Regression** or **Decision Tree Classifier**, and evaluate it using Accuracy, Precision, Recall, and F1-Score."
