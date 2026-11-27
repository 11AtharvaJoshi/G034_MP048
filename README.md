# MP048 - PG Room Rent Prediction (Indore)

## Project Details

- Group: G034
- Problem ID: MP048
- Faculty: Dr. Amit Soni
- Project: PG Room Rent Prediction

## Team Members

| Enrollment Number | Name |
|---|---|
| S24CSEU2722 | Animesh Sahu |
| S24CSEU2720 | Ashima Singh |
| S24CSEU2701 | Joy Mandal |
| S24CSEU2716 | Atharva Joshi |

## Problem Statement

The objective is to develop a machine learning
system to predict PG room rent in Indore.

This is a supervised regression problem.

Target variable: rent_rs

## Technologies

- Python
- Pandas
- Scikit-learn
- DVC
- MLflow
- FastAPI
- Docker
- Kubernetes
- GitHub Actions
- Pytest

## Project Workflow

1. Data exploration
2. Data cleaning
3. Model training
4. MLflow experiment tracking
5. FastAPI development
6. Docker containerization
7. CI/CD using GitHub Actions
8. Kubernetes deployment
9. Drift monitoring
10. Model retraining
11. Holdout predictions

## Evaluation Metric

Mean Absolute Error (MAE).

## Repository Structure

- data/: Dataset files
- src/: Source code
- app/: FastAPI application
- tests/: Automated tests
- models/: Trained models
- predictions/: Final predictions
- reports/: Project findings
- k8s/: Kubernetes configurations

### MLflow Model Registry Notes

The registered model `PG_Rent_Prediction` contains three versions:

| Version | Description |
|---|---|
| 1 | Original v1 model |
| 2 | Duplicate registration of the original v1 model |
| 3 | Retrained v2 candidate |

The retrained v2 candidate is registered as version 3 because version 2 was already occupied by a duplicate registration of v1.

The `champion` alias points to version 3, which is the actual retrained model.

The v2 candidate achieved an MAE of 1665.75 compared with 1896.73 for v1 on the same evaluation dataset, representing a 12.18% improvement. This exceeds the required 10% promotion threshold.

The existing registry history has been preserved rather than deleting or modifying previous model versions.