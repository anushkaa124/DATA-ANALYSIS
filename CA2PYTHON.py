import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df2 = pd.read_csv("/Users/anushka./Downloads/imdb_top_1000 2.csv")
print("Original columns:", df2.columns.tolist())

# Include all required columns
required_columns = ['Gross', 'Runtime', 'Meta_score', 'IMDB_Rating', 'Genre', 'Certificate']
df2 = df2[required_columns].dropna()

# Convert 'Gross' from string to float (remove commas, scale to crores)
df2['Gross'] = df2['Gross'].str.replace(',', '').astype(float) / 10000000  # in crores

# Clean and convert 'Runtime' (e.g., "142 min" → 142.0)
df2['Runtime'] = df2['Runtime'].str.extract('(\d+)').astype(float)

# Normalize Meta_score to 0–10 scale
df2['Meta_score'] = df2['Meta_score'].astype(float) / 10

# Extract first genre only
df2['Genre'] = df2['Genre'].str.split(',').str[0]

# Check for missing values
print("Missing values after cleaning:\n", df2.isnull().sum())
print("Cleaned Data Shape:", df2.shape)
print("Sample Cleaned Data:\n", df2.head())
print("Basic Statistics:")
print(df2.describe())

# Histogram of Gross Revenue
plt.figure(figsize=(8, 5))
sns.histplot(df2['Gross'], bins=20, kde=True, color='skyblue')
plt.title('Gross Revenue Distribution (in Crores)')
plt.xlabel('Gross Revenue (₹ Crores)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar plot of top 10 genres by revenue
plt.figure(figsize=(8, 6))
genre_revenue = df2.groupby('Genre')['Gross'].mean().sort_values(ascending=False).head(10)
genre_revenue.plot(kind='bar')
plt.title('Top 10 Genres by Average Gross Revenue')
plt.xlabel('Genre')
plt.ylabel('Average Revenue (₹ Crores)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Improved Pie chart of Certificate distribution
plt.figure(figsize=(10, 10))

# Get value counts and limit to top 5, group the rest as "Other"
certificate_counts = df2['Certificate'].value_counts()
top_n = 5
top_certificates = certificate_counts.head(top_n)
others_count = certificate_counts[top_n:].sum()
top_certificates['Other'] = others_count

# Define colors
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#CCCCFF']

# Create pie chart
plt.pie(top_certificates, labels=top_certificates.index, autopct='%1.1f%%', 
        colors=colors, startangle=90, pctdistance=0.85, labeldistance=None)
plt.title('Distribution of Certificate Ratings', pad=20)
plt.axis('equal')

# Add legend and move percentage labels outside
plt.legend(top_certificates.index, loc="center left", bbox_to_anchor=(1, 0.5), 
           fontsize=10, title="Certificates")
for i, patch in enumerate(plt.gcf().get_axes()[0].patches):
    plt.gcf().get_axes()[0].annotate(f'{top_certificates.iloc[i]/top_certificates.sum()*100:.1f}%',
                                    xy=(patch.center[0] * 1.4, patch.center[1] * 1.4),
                                    ha='center', va='center')
plt.tight_layout()
plt.show()

# Scatter plot of Gross Revenue vs Runtime
plt.figure(figsize=(8, 6))
plt.plot(df2['Runtime'], df2['Gross'], 'o', alpha=0.5)
plt.title('Gross Revenue vs Runtime')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Gross Revenue (₹ Crores)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlation Heatmap
numeric_cols = ['Gross', 'Runtime', 'Meta_score', 'IMDB_Rating']
correlation_matrix = df2[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.show()

# Bar plot of average Gross Revenue by Genre
plt.figure(figsize=(10, 6))
genre_revenue = df2.groupby('Genre')['Gross'].mean().sort_values()
genre_revenue.plot(kind='bar')
plt.title('Average Gross Revenue by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Gross Revenue (₹ Crores)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Box plot for Gross with outliers
plt.figure(figsize=(10, 6))
sns.boxplot(y=df2['Gross'], palette='Set2', fliersize=8)
plt.title('Box Plot of Gross Revenue (with Outliers)')
plt.ylabel('Gross Revenue (₹ Crores)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Filter outliers using IQR and create box plot
Q1 = df2['Gross'].quantile(0.25)
Q3 = df2['Gross'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = df2[(df2['Gross'] >= lower_bound) & (df2['Gross'] <= upper_bound)]

plt.figure(figsize=(10, 6))
sns.boxplot(y=df_no_outliers['Gross'], palette='Set2')
plt.title('Box Plot of Gross Revenue (Outliers Removed)')
plt.ylabel('Gross Revenue (₹ Crores)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()