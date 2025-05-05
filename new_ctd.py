import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import zscore

def load_and_filter_data(filepath):
    """
    Loads the dataset and filters temperature anomaly data.
    
    Parameters:
        filepath (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Filtered and formatted dataframe.
    """
    df = pd.read_csv(filepath)
    df = df[df["Indicator"].str.contains("Temperature", case=False)]

    id_cols = ["Country", "ISO2", "ISO3"]
    year_cols = [col for col in df.columns if col.startswith("F")]
    df = df[id_cols + year_cols]
    return df

def melt_and_clean(df):
    """
    Converts wide format to long format, handles missing data via interpolation.
    
    Parameters:
        df (pd.DataFrame): Raw dataframe.
    
    Returns:
        pd.DataFrame: Cleaned long-form dataframe.
    """
    id_cols = ["Country", "ISO2", "ISO3"]
    year_cols = [col for col in df.columns if col.startswith("F")]
    
    df_melted = df.melt(id_vars=id_cols, value_vars=year_cols,
                        var_name="Year", value_name="Anomaly")
    df_melted["Year"] = df_melted["Year"].str[1:].astype(int)

    # Mean Imputation
    df_melted['Anomaly'] = df_melted['Anomaly'].fillna(df_melted['Anomaly'].mean())
    df_melted.dropna(inplace=True)

    # Detect and filter out outliers
    df_melted['zscore'] = df_melted.groupby('Country')['Anomaly'].transform(zscore)
    df_melted = df_melted[df_melted['zscore'].abs() < 3]
    df_melted.drop(columns=['zscore'], inplace=True)

    return df_melted

def train_linear_model(df_melted):
    """
    Trains a linear regression model on the global average anomaly over time.
    
    Parameters:
        df_melted (pd.DataFrame): Long-form cleaned dataset.
    
    Returns:
        LinearRegression: Trained model.
    """
    pivot_df = df_melted.pivot(index='Year', columns='Country', values='Anomaly').fillna(0)
    X = np.array(pivot_df.index).reshape(-1, 1)
    y_global = pivot_df.mean(axis=1).values

    model = LinearRegression()
    model.fit(X, y_global)
    print(f"Global temperature anomaly trend: {model.coef_[0]:.5f} per year")

    return model, pivot_df

def train_polynomial_model(df_melted, degree=2):
    """
    Trains a polynomial regression model to capture non-linear trends.
    """
    pivot_df = df_melted.pivot(index='Year', columns='Country', values='Anomaly').fillna(0)
    X = np.array(pivot_df.index).reshape(-1, 1)  # Years as feature
    y_global = pivot_df.mean(axis=1).values  # Global mean anomaly (averaged over countries)

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y_global)

    print(f"Polynomial regression model coefficients: {model.coef_}")
    return model, poly, pivot_df

def interactive_menu():
    """
    Displays a simple CLI for the user to choose which analysis to run.
    """
    print("Welcome to the Climate Change Analysis Tool")
    print("1. Load and Clean Data")
    print("2. Train Linear Regression Model")
    print("3. Train Polynomial Regression Model")
    print("4. Visualize Future Predictions (Linear Model)")
    print("5. Visualize Polynomial Model Predictions")
    print("6. Exit")

    choice = input("Please select an option (1-6): ")

    if choice == '1':
        file_path = input("Enter the file path for the data: ")
        df = load_and_filter_data(file_path)
        df_clean = melt_and_clean(df)
        print("Data loaded and cleaned successfully!")
        return df_clean
    elif choice == '2':
        model, pivot_df = train_linear_model(df_clean)
        print("Linear regression model trained successfully!")
        return model, pivot_df
    elif choice == '3':
        model, poly, pivot_df = train_polynomial_model(df_clean, degree=2)
        print("Polynomial regression model trained successfully!")
        return model, poly, pivot_df
    elif choice == '4':
        # Assuming df_clean is already available
        future_years = np.arange(2023, 2036).reshape(-1, 1)
        future_preds = model.predict(future_years)
        future_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted_Anomaly': future_preds})
        print("Future predictions visualized!")
        plot_future_predictions(future_df)
    elif choice == '5':
        # Assuming df_clean and poly, model are available
        future_df = pd.DataFrame({
            'Year': np.arange(2023, 2036),
            'Predicted_Anomaly': model.predict(poly.transform(np.arange(2023, 2036).reshape(-1, 1)))
        })
        print("Polynomial predictions visualized!")
        plot_polynomial_predictions(model, poly, future_df, np.array(df_clean['Year']))
    elif choice == '6':
        print("Exiting...")
        exit()
    else:
        print("Invalid choice. Please try again.")
        interactive_menu()


def plot_future_predictions(future_df):
    """
    Plots the predicted future temperature anomalies.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Year', y='Predicted_Anomaly', data=future_df)
    plt.title("Predicted Global Temperature Anomalies (2023-2035)")
    plt.xlabel("Year")
    plt.ylabel("Predicted Anomaly (°C)")
    plt.grid(True)
    plt.savefig("future_predictions_2035.png")  # Save plot
    plt.close()

def plot_polynomial_predictions(model, poly, future_df, X):
    """
    Plots the polynomial regression predictions alongside actual data.
    """
    X_range = np.linspace(min(X), max(X) + 10, 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_range_pred = model.predict(X_range_poly)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Year', y='Predicted_Anomaly', data=future_df, label="Predicted Anomalies")
    plt.plot(X_range, y_range_pred, label="Polynomial Regression Prediction", color='red')
    plt.title("Predicted Global Temperature Anomalies (Polynomial Regression)")
    plt.xlabel("Year")
    plt.ylabel("Predicted Anomaly (°C)")
    plt.legend()
    plt.grid(True)
    plt.savefig("polynomial_predictions.png")  # Save plot
    plt.close()

def explain_models():
    """
    Provide justification for the models used.
    """
    print("\nJustification for Model Choices:")
    print("1. **Linear Regression Model**: Chosen for its simplicity and interpretability. It assumes a linear relationship between the year and global temperature anomaly, which is appropriate for detecting long-term trends in data that appears to be linearly growing over time.")
    print("2. **Polynomial Regression Model**: Chosen to capture potential non-linear trends in the temperature anomaly data. Non-linear models can better fit complex patterns that linear models may miss, especially when the relationship between variables is not purely linear.")

# === MAIN EXECUTION BLOCK ===

if __name__ == "__main__":
    df_clean = interactive_menu()
    explain_models()  # Provide model justification
