FROM python:3.13-slim

LABEL authors="howlingwolfs"

WORKDIR /app

# Install necessary system dependencies for scientific computing packages (like numpy, scipy)
RUN apt-get update && apt-get install -y build-essential libblas-dev liblapack-dev && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
# We assume requirements.txt contains: click, numpy, pandas, requests, scikit-learn, scipy, xgboost, etc.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run the application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]