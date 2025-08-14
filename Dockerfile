# 1. Use an official Python base image
FROM python:3.9-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy requirements.txt into the container
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the app files
COPY app.py .
COPY rf_classifier.pkl .
COPY preprocessor.pkl .
COPY Churn_Prediction_Final.csv .
COPY templates/ ./templates/

# 6. Expose Flask's default port
EXPOSE 5000

# 7. Run the app
CMD ["python", "app.py"]

