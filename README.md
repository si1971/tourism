# Tourism Package Prediction — MLOps Project

This project predicts whether a customer is likely to purchase the Wellness
Tourism Package before sales outreach.

## MLOps workflow

1. Register raw `tourism.csv` under `turisum/data/`
2. Clean and split the data
3. Tune Random Forest and Gradient Boosting models
4. Select the best model using cross-validated average precision
5. Select an outreach threshold from training out-of-fold predictions
6. Save the production model and metadata
7. Run the workflow automatically with GitHub Actions on pushes to `main`
8. Deploy `turisum/deployment/app.py` using Streamlit Community Cloud

## Streamlit entrypoint

`turisum/deployment/app.py`

## GitHub Actions workflow

`.github/workflows/pipeline.yml`
