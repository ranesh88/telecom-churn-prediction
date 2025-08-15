# 1. Use official Python image
FROM python:3.9-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy app files
COPY app.py .
COPY Churn_Prediction_Final.csv .
COPY templates/ ./templates/

# 5. Copy trained model and preprocessor
COPY src/models/rf_classifier.pkl .
COPY src/models/preprocessor.pkl .

# 6. Expose Flask port
EXPOSE 5000

# 7. Run the app
CMD ["python", "app.py", "--host=0.0.0.0"]
