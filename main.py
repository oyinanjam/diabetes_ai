import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Baca dataset
data = pd.read_csv('diabetes.csv')

#tampilin info dari dataset
print("Informasi Dataset:")
print(data.info())
print("\nStatistik Deskriptif:")
print(data.describe())

#tampilin beberapa baris pertama
print("\nData pertama:")
print(data.head())

# visualisasi fitur
sns.pairplot(data, hue="Outcome")
plt.show()

# Pisahkan fitur dan target
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Split data menjadi data latih dan data uji (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi fitur (standarisasi)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Bangun model regresi logistik
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediksi dengan data uji
y_pred = model.predict(X_test)

# Evaluasi hasil model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
