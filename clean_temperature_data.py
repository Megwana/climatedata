import pandas as pd

# Load dataset
df = pd.read_csv("climate_change_indicators.csv")

# Filter only rows related to temperature anomalies (if dataset includes other indicators)
df = df[df["Indicator"].str.contains("Temperature", case=False)]

# Keep only relevant columns
id_cols = ["Country", "ISO2", "ISO3"]
year_cols = [col for col in df.columns if col.startswith("F")]
df = df[id_cols + year_cols]

# Melt the dataframe to a long format: Country, Year, Anomaly
df_melted = df.melt(id_vars=id_cols, value_vars=year_cols,
                    var_name="Year", value_name="Anomaly")

# Convert year format from 'F1961' to '1961'
df_melted["Year"] = df_melted["Year"].str[1:].astype(int)

# Drop rows with missing values
df_melted.dropna(inplace=True)

# Preview
print(df_melted.head())
