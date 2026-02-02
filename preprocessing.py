import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# expected features used in training (must match model.sav)
EXPECTED_COLUMNS = [
    'SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling',
    'MonthlyCharges', 'TotalCharges', 'MultipleLines_No_phone_service',
    'MultipleLines_Yes', 'InternetService_Fiber_optic', 'InternetService_No',
    'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes',
    'OnlineBackup_No_internet_service', 'TechSupport_No_internet_service',
    'TechSupport_Yes', 'StreamingTV_No_internet_service', 'StreamingTV_Yes',
    'StreamingMovies_No_internet_service', 'StreamingMovies_Yes',
    'Contract_One_year', 'Contract_Two_year', 'PaymentMethod_Electronic_check'
]

def preprocess(df, option):
    """
    Preprocess churn dataframe:
    - Handle missing values
    - Encode categorical features
    - Scale numeric features
    - Ensure output matches training columns
    """

    # 🔹 Handle missing values
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace(" ", pd.NA)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill numeric NaNs with median
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical NaNs with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # 🔹 Binary mapping for specific Yes/No fields
    binary_list = ['SeniorCitizen','Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_list:
        if col in df.columns:
            df[col] = df[col].map({'Yes':1, 'No':0}).fillna(df[col])

    # 🔹 One-hot encode categorical features
    df = pd.get_dummies(df)

    # 🔹 Reindex to match training features
    df = df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

    # 🔹 Scale all numeric columns
    scaler = MinMaxScaler()
    df[num_cols.intersection(df.columns)] = scaler.fit_transform(df[num_cols.intersection(df.columns)])

    return df
