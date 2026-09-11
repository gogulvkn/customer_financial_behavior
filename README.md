# Customer Financial Behavior Analysis

A machine learning project that analyzes customer financial behavior and assigns customers to behavioral segments using **K-Means clustering**.

## Project Overview

Customer financial data contains useful signals about spending, saving, borrowing, credit usage, and investment behavior. This project uses unsupervised machine learning to identify meaningful customer segments based on their financial characteristics.

The trained clustering model is integrated into a **Flask web application**, where users can enter financial information and receive a predicted customer segment.

## Key Features

- Customer financial behavior analysis
- Exploratory data analysis using Python and Jupyter Notebook
- Data preprocessing and feature scaling
- K-Means clustering for customer segmentation
- Saved machine learning model and scaler for inference
- Flask-based prediction web application
- Interactive form for entering customer financial details
- Customer segmentation into:
  - **Normal**
  - **Premium**

## Financial Features Used

The prediction application accepts the following inputs:

| Feature | Description |
|---|---|
| `Annual_Income` | Customer's annual income |
| `Monthly_Spending` | Average monthly spending |
| `Savings` | Customer savings amount |
| `Checking_Balance` | Checking account balance |
| `Credit_Score` | Customer credit score |
| `Credit_Card_Spending` | Credit card spending amount |
| `Loan_Amount` | Outstanding or associated loan amount |
| `Investment_Amount` | Amount invested |
| `Debt_to_Income_Ratio` | Debt relative to income |
| `Transaction_Frequency` | Frequency of financial transactions |

## Machine Learning Workflow

The project follows a typical machine learning pipeline:

1. Load the customer financial dataset.
2. Explore and understand the data.
3. Select relevant financial features.
4. Preprocess and scale the numerical features.
5. Apply **K-Means clustering**.
6. Analyze the resulting customer clusters.
7. Save the trained model and preprocessing artifacts.
8. Build a Flask application for real-time predictions.
9. Assign a customer to a behavioral segment based on their financial profile.

## Model

### K-Means Clustering

K-Means is an unsupervised learning algorithm that groups observations according to similarity.

In this project, the trained K-Means model is loaded from:

```text
kmeans_model.pkl
```

Input features are transformed using the saved scaler:

```text
scaler.pkl
```

The Flask application scales the submitted financial data before passing it to the clustering model.

### Segment Interpretation

The current application maps the model's cluster output as follows:

```text
Cluster 0 → Normal
Other cluster → Premium
```

> Note: Cluster labels are model-specific. If the underlying clustering model is retrained, the numerical cluster IDs should be revalidated before using the same business labels.

## Project Structure

```text
customer_financial_behavior/
│
├── app.py
├── customer_financial_behavior.ipynb
├── customer_financial_behavior_10000.csv
├── cluster.csv
├── form_val.pkl
├── kmeans_model.pkl
├── scaler.pkl
└── README.md
```

### File Description

- **`app.py`** — Flask application used for customer-segment prediction.
- **`customer_financial_behavior.ipynb`** — Jupyter Notebook containing the data analysis and machine learning workflow.
- **`customer_financial_behavior_10000.csv`** — Customer financial behavior dataset.
- **`cluster.csv`** — Cluster-related output/data generated during the analysis.
- **`form_val.pkl`** — Stored data used by the Flask application for form-related values.
- **`kmeans_model.pkl`** — Trained K-Means clustering model.
- **`scaler.pkl`** — Fitted feature scaler used before prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib
- Jupyter Notebook
- K-Means Clustering

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/gogulvkn/customer_financial_behavior.git
cd customer_financial_behavior
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas scikit-learn flask joblib jupyter
```

## Run the Application

Start the Flask application:

```bash
python app.py
```

The application will normally be available at:

```text
http://127.0.0.1:5000/
```

Open the address in a web browser and enter the customer's financial information.

## How Prediction Works

The Flask application receives the values submitted through the web form and processes them in the required feature order.

The prediction flow is:

```text
User Input
    ↓
Financial Features
    ↓
NumPy Array
    ↓
Feature Scaling
    ↓
K-Means Model
    ↓
Cluster Prediction
    ↓
Customer Segment
```

The application uses the saved scaler before calling the K-Means model:

```python
scaled_data = scaler.transform(input_data)
cluster = model.predict(scaled_data)[0]
```

The resulting cluster is then converted into the application's customer segment label.

## Example

A customer profile containing information such as:

- Annual income
- Monthly spending
- Savings
- Checking balance
- Credit score
- Credit-card spending
- Loan amount
- Investment amount
- Debt-to-income ratio
- Transaction frequency

is submitted through the Flask form.

The application processes the values and returns a result such as:

```text
Assigned Segment: Cluster Normal
```

or

```text
Assigned Segment: Cluster Premium
```

## Notebook

The complete analysis and model-development workflow is available in:

```text
customer_financial_behavior.ipynb
```

The notebook can be opened using Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

## Dataset

The repository includes a customer financial behavior dataset with 10,000 records:

```text
customer_financial_behavior_10000.csv
```

The project focuses on numerical financial behavior rather than supervised target prediction, making clustering suitable for discovering customer groups.

## Use Cases

This type of customer segmentation can support:

- Personalized financial products
- Customer profiling
- Marketing segmentation
- Premium customer identification
- Spending behavior analysis
- Financial service personalization
- Customer relationship management

## Important Notes

- The model is a clustering model, not a supervised classification model.
- Cluster numbers do not inherently represent business categories; the meaning assigned to each cluster should be validated against cluster characteristics.
- The application expects the input features in the same order used during model training.
- The scaler used during training should be reused during inference.
- The `.pkl` model artifacts should be kept compatible with the Python and scikit-learn environment used to load them.

## Future Improvements

Possible improvements include:

- Add cluster visualization to the web application
- Display customer-segment explanations
- Add model evaluation metrics appropriate for clustering, such as silhouette score
- Add interactive charts for financial behavior
- Improve UI/UX
- Add input validation and clearer error messages
- Add automated deployment
- Add additional customer segments based on deeper cluster analysis
- Add API endpoints for programmatic predictions

## Author

**kamatchinathan V**

GitHub: https://github.com/gogulvkn

## Repository

https://github.com/gogulvkn/customer_financial_behavior

## License

This project is intended for educational and portfolio purposes. Add an appropriate open-source license to the repository if you plan to distribute or reuse the project publicly.
