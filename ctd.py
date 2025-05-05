import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
from sklearn.preprocessing import PolynomialFeatures

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

    # # Handling missing values per country by - Interpolate and drop remaining missing values
    # df_melted['Anomaly'] = df_melted.groupby('Country')['Anomaly'].transform(lambda x: x.interpolate())
    # df_melted.dropna(inplace=True)
    
    # Mean Imputation
    df_melted['Anomaly'] = df_melted['Anomaly'].fillna(df_melted['Anomaly'].mean())
    df_melted.dropna(inplace=True)

    # Detect and filter out outliers
    df_melted['zscore'] = df_melted.groupby('Country')['Anomaly'].transform(zscore)
    df_melted = df_melted[df_melted['zscore'].abs() < 3]
    df_melted.drop(columns=['zscore'], inplace=True)

    return df_melted

def analyze_missing_data(df):
    """
    Prints summary statistics of missing data.
    
    Parameters:
        df (pd.DataFrame): Original wide-format dataframe.
    """
    missing_summary = df.isnull().sum()
    print("Missing values by column:")
    print(missing_summary[missing_summary > 0])
    
    missing_rows = df.isnull().any(axis=1).sum()
    total_rows = df.shape[0]
    missing_percentage = (missing_rows / total_rows) * 100
    
    print(f"\nTotal rows with missing values: {missing_rows} out of {total_rows} "
          f"({missing_percentage:.2f}%)")

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

    # Predict future years
    future_years = np.arange(2023, 2036).reshape(-1, 1)
    future_preds = model.predict(future_years)

    # Save predictions
    future_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted_Anomaly': future_preds})
    future_df.to_csv("future_predictions_2035.csv", index=False)
    print("Saved future predictions to CSV.")
    return model, pivot_df

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
    
def plot_polynomial_predictions(model, poly, future_df, X):
    """
    Plots the polynomial regression predictions alongside actual data.
    """
    X_range = np.linspace(min(X), max(X) + 10, 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_range_pred = model.predict(X_range_poly)

    plt.figure(figsize=(10, 6))
    # Use 'Predicted_Anomaly' from future_df instead of 'Anomaly'
    sns.lineplot(x='Year', y='Predicted_Anomaly', data=future_df, label="Predicted Anomalies")
    plt.plot(X_range, y_range_pred, label="Polynomial Regression Prediction", color='red')
    plt.title("Predicted Global Temperature Anomalies (Polynomial Regression)")
    plt.xlabel("Year")
    plt.ylabel("Predicted Anomaly (°C)")
    plt.legend()
    plt.grid(True)
    plt.savefig("polynomial_predictions.png")  # Save plot
    plt.close()



def plot_line_chart(df_melted, country='United States'):
    """
    Plots a line chart for a specific country's temperature anomaly.
    """
    plt.figure(figsize=(10, 5))
    sns.lineplot(x='Year', y='Anomaly', data=df_melted[df_melted['Country'] == country])
    plt.title(f"Temperature Anomalies in {country}")
    plt.ylabel("Anomaly (°C)")
    plt.grid(True)
    plt.savefig(f"{country}_line_chart.png")  # Save plot
    plt.close()


def plot_heatmap(pivot_df):
    """
    Plots a heatmap of temperature anomalies for all countries over the years.
    """
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_df.T, cmap='coolwarm', center=0)
    plt.title("Anomaly Heatmap by Country")
    plt.xlabel("Year")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig("anomaly_heatmap.png")  # Save plot
    plt.close()


def plot_boxplot_by_decade(df_melted):
    """
    Plots a boxplot showing anomaly distribution by decade.
    """
    df_melted['Decade'] = (df_melted['Year'] // 10) * 10
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Decade', y='Anomaly', data=df_melted)
    plt.title("Temperature Anomaly Distribution by Decade")
    plt.ylabel("Anomaly (°C)")
    plt.xlabel("Decade")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("boxplot_by_decade.png")  # Save plot
    plt.close()

def business_insights(model):
    """
    Prints a business intelligence interpretation of the trend.
    """
    yearly_increase = model.coef_[0]
    if yearly_increase > 0:
        print(f"BI Insight: The global average temperature anomaly is increasing by approx. {yearly_increase:.3f}°C per year.")
        print("Implication: Policy makers and businesses should consider proactive strategies for climate adaptation.")
    else:
        print("No significant upward trend detected — continue monitoring.")

def reflect_on_ethics():
    """
    Reflects on ethical considerations of data handling and communication.
    """
    print("\nEthical Reflection:")
    print("- Data was cleaned using interpolation, which must be disclosed as it affects accuracy.")
    print("- Ensure visualisations are not misleading by proper axis scaling and color schemes.")
    print("- Anomalies are sensitive indicators—misuse or misinterpretation could fuel climate misinformation.")

def generate_summary(df_melted, model):
    """
    Saves a CSV summary and prints key statistics.
    """
    summary = df_melted.groupby('Year')['Anomaly'].mean().reset_index(name='Global_Mean_Anomaly')
    summary.to_csv("global_anomaly_summary.csv", index=False)
    print("\nSaved global anomaly summary to CSV.")

def plot_correlation_heatmap(pivot_df):
    """
    Plots a correlation heatmap across countries.
    """
    corr = pivot_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title("Correlation of Temperature Anomalies Between Countries")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

def plot_country_avg_bar(df_melted):
    """
    Plots a horizontal bar graph of average anomalies by country.
    """
    avg_by_country = df_melted.groupby('Country')['Anomaly'].mean().sort_values()
    plt.figure(figsize=(12, 8))
    avg_by_country.plot(kind='barh')
    plt.title("Average Temperature Anomaly by Country")
    plt.xlabel("Anomaly (°C)")
    plt.tight_layout()
    plt.savefig("country_avg_bar.png")
    plt.close()


# === MAIN EXECUTION BLOCK ===

if __name__ == "__main__":
    file_path = "climate_change_indicators.csv"

    # Load and prepare data
    raw_df = load_and_filter_data(file_path)
    analyze_missing_data(raw_df)
    df_clean = melt_and_clean(raw_df)

    # Train model and analyze
    model, pivot_df = train_linear_model(df_clean)

    # Predict future anomalies (2035)
    future_years = np.arange(2023, 2036).reshape(-1, 1)
    future_preds = model.predict(future_years)

    # Save future predictions to CSV
    future_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted_Anomaly': future_preds})
    future_df.to_csv("future_predictions_2035.csv", index=False)
    print("Saved future predictions to CSV.")

    # Plot future predictions
    plot_future_predictions(future_df)

    # Train polynomial model and analyze
    model, poly, pivot_df = train_polynomial_model(df_clean, degree=2)

        # Plot Polynomial Predictions
    future_df = pd.DataFrame({
        'Year': np.arange(2023, 2036),
        'Predicted_Anomaly': model.predict(poly.transform(np.arange(2023, 2036).reshape(-1, 1)))
    })
    plot_polynomial_predictions(model, poly, future_df, np.array(df_clean['Year']))

    # Visualizations
    plot_line_chart(df_clean, country="United States")
    plot_heatmap(pivot_df)
    plot_boxplot_by_decade(df_clean)

    # Business Intelligence Interpretation
    business_insights(model)

    # Ethical Consideration Reflections
    reflect_on_ethics()

    # Summary Report
    generate_summary(df_clean, model)

    # Correlation Heat Map
    plot_correlation_heatmap(pivot_df)
    # Bar Graph for Country Comparison
    plot_country_avg_bar(df_clean)

    print(future_df.head())
    plt.show()


