# Credit Card Fraud Detection MLOps

> An end-to-end machine learning project for detecting fraudulent credit card transactions, with a production-oriented inference API, model artifacts, testing, containerization, and CI/CD.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions)](https://github.com/features/actions)

## 📌 Overview

**Credit Card Fraud Detection MLOps** is a machine learning project focused on identifying potentially fraudulent credit card transactions.

The project goes beyond model training by organizing the solution as a deployable ML application. It includes:

* Machine learning model development
* Data processing and prediction pipeline
* Saved model artifacts
* FastAPI-based inference service
* Docker containerization
* Automated testing
* CI/CD workflow
* Generated reports and evaluation artifacts

The goal is to demonstrate how a fraud detection model can move from experimentation toward a maintainable and deployable machine learning service.

---

## 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │   Transaction Data  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Processing &    │
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   ML Model Training │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Model Artifacts   │
                    │      /models        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    FastAPI Service  │
                    │      /predict       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Prediction      │
                    │ Fraud / Legitimate  │
                    └─────────────────────┘
```

The repository also includes automated tests and GitHub Actions to help validate changes throughout development.

---

## ✨ Features

### Machine Learning

* Fraud detection model training
* Data preprocessing and feature preparation
* Model serialization and loading
* Prediction pipeline
* Evaluation/report generation

### API

The trained model can be exposed through a **FastAPI** service.

FastAPI provides a lightweight interface for sending transaction information to the model and receiving fraud predictions.

Interactive API documentation is available through FastAPI's built-in Swagger UI when the application is running.

### MLOps

The project includes several practices commonly used in production ML workflows:

* Reproducible project structure
* Version-controlled source code
* Model artifact management
* Automated tests
* Docker containerization
* GitHub Actions CI/CD
* Generated reports

### Testing

Automated tests are included under:

```text
test/
```

These tests help verify that the application's core functionality continues to work as the project evolves.

---

## 🛠️ Tech Stack

| Category         | Technology               |
| ---------------- | ------------------------ |
| Language         | Python                   |
| API              | FastAPI                  |
| Machine Learning | Python ML ecosystem      |
| Containerization | Docker                   |
| CI/CD            | GitHub Actions           |
| Testing          | Python testing framework |
| Model Storage    | `models/`                |
| Reports          | `reports/`               |

---

## 📁 Project Structure

```text
fraud-detection-mlops/
│
├── .github/
│   └── workflows/
│       └── # CI/CD workflows
│
├── models/
│   └── # Trained model artifacts
│
├── reports/
│   └── # Evaluation / generated reports
│
├── src/
│   └── # Application and ML source code
│
├── test/
│   └── # Automated tests
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

* Python 3.x
* Git
* Docker (optional, for containerized execution)

---

### 1. Clone the Repository

```bash
git clone https://github.com/howlingwolfs/fraud-detection-mlops.git

cd fraud-detection-mlops
```

---

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

The project contains a FastAPI-based inference service.

Depending on the application entry point in `src`, start the API using the corresponding Uvicorn command.

For example:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Once running, open:

```text
http://localhost:8000/docs
```

to access the interactive Swagger API documentation.

> **Note:** If your FastAPI application uses a different module path, update the Uvicorn command accordingly.

---

## 🐳 Running with Docker

Build the Docker image:

```bash
docker build -t fraud-detection-mlops .
```

Run the container:

```bash
docker run -p 8000:8000 fraud-detection-mlops
```

The API should then be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## 🧪 Running Tests

Run the test suite with:

```bash
pytest
```

For more detailed output:

```bash
pytest -v
```

---

## 🔄 CI/CD

The repository contains GitHub Actions workflows under:

```text
.github/workflows/
```

These workflows are used to automate project validation and help ensure that changes can be tested consistently.

Typical CI checks can include:

```text
Push / Pull Request
        │
        ▼
┌──────────────────┐
│ Install Python   │
│ Dependencies     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Run Tests        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Build / Validate │
└──────────────────┘
```

---

## 📊 Machine Learning Workflow

The general workflow of the project is:

```text
Raw Transaction Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Save Model
        │
        ▼
FastAPI Inference Service
        │
        ▼
Fraud Prediction
```

### Why Fraud Detection Is Different

Fraud detection is typically an imbalanced classification problem where fraudulent transactions represent a much smaller portion of the overall transaction volume.

Because of this, **accuracy alone is not sufficient** for evaluating a fraud detection model.

Useful evaluation metrics can include:

* Precision
* Recall
* F1-score
* ROC-AUC
* Precision-Recall AUC
* Confusion matrix

The appropriate metric depends on the operational requirements of the fraud detection system.

---

## 🔌 API

The FastAPI service provides an HTTP interface around the trained model.

A typical prediction request follows this pattern:

```http
POST /predict
Content-Type: application/json
```

Example structure:

```json
{
  "feature_1": 0.12,
  "feature_2": 1.45,
  "feature_3": 0.87
}
```

Example response:

```json
{
  "prediction": 0,
  "probability": 0.03
}
```

> The exact request and response schema depends on the implementation currently present in `src/`.

---

## 📈 Reports

Generated analysis and evaluation artifacts are stored in:

```text
reports/
```

These reports can be used to inspect model performance and experiment results without requiring every experiment to be reproduced manually.

---

## 🔒 Data & Security

Fraud detection datasets may contain sensitive or proprietary information.

This repository should **not** commit:

* Production transaction data
* Personally identifiable information (PII)
* API keys
* Credentials
* Secrets
* Private datasets
* Environment-specific configuration containing sensitive values

Use `.gitignore` and environment variables for sensitive configuration.

---

## 🧭 Roadmap

Potential improvements for the project include:

* [ ] Add experiment tracking with MLflow
* [ ] Add model versioning
* [ ] Add data validation
* [ ] Add data and model drift detection
* [ ] Add Prometheus metrics
* [ ] Add Grafana dashboards
* [ ] Add automated model retraining
* [ ] Add model explainability with SHAP
* [ ] Add API authentication
* [ ] Add structured logging
* [ ] Add health and readiness endpoints
* [ ] Add integration tests
* [ ] Add automated Docker image publishing
* [ ] Add deployment configuration for cloud/Kubernetes environments

---

## 🤝 Contributing

Contributions, improvements, and suggestions are welcome.

### Development Workflow

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/my-feature
```

3. Make your changes.
4. Run the tests.

```bash
pytest
```

5. Commit your changes.

```bash
git commit -m "Add my feature"
```

6. Push the branch.

```bash
git push origin feature/my-feature
```

7. Open a Pull Request.

---

## 📄 License

See the repository for the current license information.

If this project does not yet have a license, consider adding an appropriate open-source license before publishing it for external contributions.

---

## 👤 Author

**Howlingwolfs**

GitHub:
https://github.com/howlingwolfs

Project:
https://github.com/howlingwolfs/fraud-detection-mlops

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📚 Disclaimer

This project is intended for educational and research purposes.

Fraud detection systems used in real financial environments require additional considerations including security, privacy, regulatory compliance, monitoring, human review processes, false-positive handling, and extensive validation before deployment.
