# Fake News Detection Using NLP

### End-to-End Machine Learning Application with PostgreSQL, Docker & Jenkins CI/CD

A machine learning-based web application that classifies news articles as **REAL** or **FAKE** using Natural Language Processing (NLP), TF-IDF vectorization, and Logistic Regression.

The project integrates an interactive Streamlit interface, PostgreSQL prediction history, Docker containerization, and an automated Jenkins CI/CD pipeline.

---

## 1. Project Overview

The Fake News Detection System is designed to analyze news articles and predict whether their content resembles real or fake news based on patterns learned from a labeled dataset.

Users can enter news text through the Streamlit web interface, receive a classification result with a confidence score, and review previous predictions stored in PostgreSQL.

The application is deployed locally using Docker Compose. Jenkins automates source-code checkout, Docker image creation, automated testing, deployment, and verification.

**Note:** Predictions are statistical estimates based on the trained model, not independent verification of factual claims.

## 2. Project Objectives

- Develop a machine learning-based news classification system.
- Implement NLP preprocessing and TF-IDF feature extraction.
- Train a Logistic Regression classification model.
- Build an interactive Streamlit web application.
- Store prediction results in a PostgreSQL database.
- Maintain prediction history across application restarts.
- Containerize the application using Docker.
- Automate testing and local deployment using Jenkins CI/CD.
- Maintain project source code through Git and GitHub.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Natural Language Processing | NLTK |
| Feature Extraction | TF-IDF Vectorizer |
| Machine Learning | Scikit-learn |
| Classification Algorithm | Logistic Regression |
| Web Application | Streamlit |
| Database | PostgreSQL 16 |
| Database Connectivity | Psycopg |
| Containerization | Docker, Docker Compose |
| CI/CD Automation | Jenkins |
| Version Control | Git, GitHub |
| Automated Testing | Python unittest |

## 4. System Architecture

The system consists of five main components:

1. **Streamlit Interface:** Accepts news articles and displays classification results.
2. **NLP Preprocessing:** Cleans and prepares the input text for model inference.
3. **Machine Learning Model:** Uses TF-IDF features and Logistic Regression to predict REAL or FAKE.
4. **PostgreSQL Database:** Stores prediction records and supports history retrieval.
5. **DevOps Pipeline:** Uses GitHub, Jenkins, and Docker to automate testing and local deployment.

### Application Workflow

```text
                  USER
                    |
                    v
            Streamlit Web App
                    |
                    v
             News Article Input
                    |
                    v
             NLP Preprocessing
                    |
                    v
             TF-IDF Vectorizer
                    |
                    v
          Logistic Regression Model
                    |
                    v
             Prediction Result
               REAL / FAKE
                    |
                    v
            Confidence Score
                    |
                    v
           PostgreSQL Database
                    |
                    v
             Prediction History
```

### CI/CD Workflow

```text
           GitHub Repository
                    |
                    v
              Jenkins Pipeline
                    |
                    v
             Source Code Checkout
                    |
                    v
              Docker Image Build
                    |
                    v
             Automated Unit Tests
                    |
                    v
          Local Docker Deployment
                    |
                    v
          Application Health Checks
                    |
                    v
             Deployment Success
```

## 5. Dataset and Machine Learning

### Dataset

The machine learning model was developed using a labeled news dataset sourced from Kaggle.

The dataset includes two primary categories:

- **Fake.csv:** Articles labeled as fake news.
- **True.csv:** Articles labeled as real news.

These datasets were combined and prepared for supervised classification.

### NLP Preprocessing

News text is preprocessed before feature extraction.

The preprocessing workflow includes text normalization and cleaning to reduce unnecessary variation in the input.

### TF-IDF Vectorization

Term Frequency–Inverse Document Frequency (TF-IDF) transforms text into numerical feature vectors.

It helps represent the importance of terms within a document relative to their occurrence across the training corpus.

### Logistic Regression

Logistic Regression is a supervised classification algorithm used to estimate the probability of a news article belonging to a particular class.

The trained model is saved for reuse during inference.

**Model artifacts:**

- `model.pkl` — Trained classification model.
- `vectorizer.pkl` — Fitted TF-IDF vectorizer.

The application loads these artifacts to classify submitted articles without retraining the model for each prediction.

## 6. Streamlit Web Application

The Streamlit interface provides:

- News article text input.
- REAL or FAKE prediction.
- Prediction confidence score.
- PostgreSQL-backed prediction history.
- A browser-based interface for demonstrating the model.

**Local application URL:**

http://localhost:8501

The application is accessible on the local machine while the Docker services are running.

## 7. PostgreSQL Database Integration

The application uses PostgreSQL to persist prediction records.

The database runs in a separate Docker container and communicates with the Streamlit application over the Docker Compose network.

### Database Configuration

| Setting | Value |
|---|---|
| Database Engine | PostgreSQL 16 |
| Database Name | `news` |
| Database User | `newsapp` |
| Database Service | `db` |
| Persistent Storage | Docker named volume `pgdata` |

The database password is supplied through environment configuration rather than hard-coded into the application.

### Verify Stored Predictions

Run the following command from the project directory:

```bash
docker compose exec -T db psql -U newsapp -d news -c "SELECT id, label, confidence, created_at FROM predictions ORDER BY id DESC LIMIT 5;"
```

This displays the latest prediction records stored in PostgreSQL.

### Data Persistence

The database uses a Docker named volume to preserve records across normal container restarts and recreation.

**Important:** Avoid using `docker compose down -v` unless you intentionally want to remove the database volume and its stored data.

## 8. Docker Containerization

Docker packages the application and its dependencies into a reproducible environment.

Docker Compose manages two services:

| Service | Purpose |
|---|---|
| `web` | Runs the Streamlit application |
| `db` | Runs the PostgreSQL database |

The application service depends on the database health check.

### Prerequisites

Install:

- Docker Desktop
- Git
- Jenkins (for CI/CD demonstrations)

### Clone the Repository

```bash
git clone https://github.com/Shaurya9260/fake-news-detector.git
cd fake-news-detector
```

### Configure Environment Variables

Create a `.env` file in the project root.

Example:

```env
POSTGRES_PASSWORD=replace_with_a_strong_password
```

Do not commit your actual `.env` file or database credentials to GitHub.

### Build and Start the Application

```bash
docker compose up -d --build --wait
```

### Verify Container Health

```bash
docker compose ps
```

Both the `web` and `db` services should report a healthy status.

### Open the Application

Visit:

http://localhost:8501

### Stop Containers

```bash
docker compose stop
```

### Restart Existing Containers

```bash
docker compose start
```

## 9. Jenkins CI/CD Pipeline

The project uses Jenkins to automate application integration, testing, and local deployment.

### Pipeline Configuration

| Property | Configuration |
|---|---|
| Jenkins Job | `Fake-News-Detection-CICD` |
| Pipeline Type | Pipeline from SCM |
| Source Control | GitHub |
| Branch | `main` |
| Pipeline Definition | `Jenkinsfile` |
| Deployment Platform | Local Docker Desktop |

### Pipeline Stages

**Stage 1 — Checkout**

Jenkins retrieves the latest project source code from GitHub.

**Stage 2 — Build**

Jenkins builds the application Docker image using the project Dockerfile.

**Stage 3 — Automated Testing**

The pipeline runs Python automated tests against an isolated PostgreSQL test database.

**Stage 4 — Deployment**

After successful testing, Jenkins deploys the application using Docker Compose.

**Stage 5 — Verification and Cleanup**

The pipeline verifies application and database connectivity and cleans up temporary CI resources.

### Automated Build Trigger

The Jenkins pipeline uses SCM polling to check the repository for changes:

```groovy
pollSCM('H/2 * * * *')
```

When changes are detected, Jenkins can automatically start a new pipeline run.

The actual polling schedule is determined by Jenkins.

### CI/CD Verification

A successful local pipeline execution confirmed:

- GitHub source-code checkout.
- Docker image build.
- Automated Python tests.
- PostgreSQL test integration.
- Docker Compose deployment.
- Application and database health verification.

**Verified Jenkins Result:** Build #3 — SUCCESS.

## 10. Automated Testing

The project includes automated tests in:

```text
tests/test_practical.py
```

The tests verify important application behaviors, including:

1. Text preprocessing.
2. Handling of blank input without saving an unwanted prediction.
3. Persistence and retrieval of REAL/FAKE prediction records across database sessions.

### Test Results

| Metric | Result |
|---|---|
| Automated Tests Executed | 3 |
| Tests Passed | 3 |
| Tests Failed | 0 |
| Jenkins Build #3 | SUCCESS |

The pipeline prevents deployment when the automated testing stage fails.

## 11. Functional Verification

The application was also tested manually through the Streamlit interface.

The verification included:

- Submitting sample news articles.
- Receiving REAL and FAKE classifications.
- Displaying prediction confidence.
- Viewing saved prediction history.
- Querying prediction records directly from PostgreSQL.
- Confirming both Docker services were healthy.

These tests confirm that the application's main workflow operates locally.

They do not establish the model's accuracy on an independent evaluation dataset.

## 12. Project Screenshots

Screenshots of the completed practical are organized in the `Project_Screenshots` directory.

| Screenshot | Description |
|---|---|
| `01_Streamlit_Homepage.png` | Streamlit application interface |
| `02_Fake_News_Prediction.png` | FAKE prediction demonstration |
| `03_Real_News_Prediction.png` | REAL prediction demonstration |
| `04_Prediction_History.png` | Stored prediction history |
| `05_PostgreSQL_Database.png` | PostgreSQL prediction records |
| `06_Docker_Containers.png` | Healthy Docker services |
| `07_Jenkins_Pipeline_SUCCESS.png` | Successful Jenkins pipeline |
| `08_Jenkins_Automated_Tests.png` | Automated test execution |
| `09_GitHub_Repository.png` | GitHub source-code repository |

The screenshots can be included in the project report and published in the repository after confirming that no credentials or sensitive information are visible.

## 13. Project Structure

```text
fake-news-detector/
|
|-- app.py
|-- database.py
|-- utils.py
|
|-- model.pkl
|-- vectorizer.pkl
|
|-- prepare_dataset.py
|-- train_new_model.py
|-- evaluate_model.py
|
|-- Dockerfile
|-- compose.yaml
|-- Jenkinsfile
|-- .dockerignore
|-- .env.example
|-- .gitignore
|
|-- requirements.txt
|-- README.md
|
|-- tests/
|   |-- test_practical.py
|
|-- Project_Screenshots/
|   |-- 01_Streamlit_Homepage.png
|   |-- 02_Fake_News_Prediction.png
|   |-- 03_Real_News_Prediction.png
|   |-- 04_Prediction_History.png
|   |-- 05_PostgreSQL_Database.png
|   |-- 06_Docker_Containers.png
|   |-- 07_Jenkins_Pipeline_SUCCESS.png
|   |-- 08_Jenkins_Automated_Tests.png
|   |-- 09_GitHub_Repository.png
|
|-- README_backup.md
```

The screenshot directory and backup file are local working files unless explicitly committed to GitHub.

## 14. Security and Best Practices

The project follows several basic development and deployment practices:

- Environment variables are used for database credentials.
- Private `.env` files should remain excluded from Git.
- The PostgreSQL database is not published directly to the host network.
- Streamlit is bound to the local loopback interface.
- Jenkins credentials should be stored securely.
- Automated tests run before deployment.
- PostgreSQL data is stored in a persistent Docker volume.

This configuration is intended for local development and practical demonstration. Production deployment would require additional security, monitoring, access control, and infrastructure configuration.

## 15. Future Enhancements

Potential improvements include:

- Comparing additional classification algorithms.
- Training on newer and more diverse datasets.
- Improving model explainability.
- Adding authenticated user accounts.
- Developing a REST API.
- Adding monitoring and logging dashboards.
- Expanding automated tests.
- Deploying the application to a managed cloud platform.
- Integrating external fact-checking sources.

## 16. Limitations

- The model classifies text based on patterns learned from its training data.
- A REAL prediction does not guarantee that every claim is factually correct.
- Model performance may vary across topics, languages, and newer events.
- Confidence scores are model estimates and should not be treated as factual certainty.
- The demonstrated deployment runs locally, not on a public production server.

## 17. Repository

**GitHub:** https://github.com/Shaurya9260/fake-news-detector

**Project:** Fake News Detection Using NLP

**Category:** Machine Learning, NLP, Data Science, DevOps and CI/CD

---

**Project Summary**

This project demonstrates an end-to-end machine learning application combining NLP-based classification, a Streamlit web interface, PostgreSQL persistence, Docker containerization, automated Python testing, and Jenkins CI/CD deployment.

It showcases the integration of machine learning development with practical database engineering and DevOps workflows.