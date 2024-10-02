import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from joblib import dump, load
import seaborn as sns

# 1. Veri Setini Yükleme ve İnceleme
df_housing = pd.read_csv('housing.csv')

# 2. Eğitim ve Test Setlerini Oluşturma (tek özellik: housing_median_age)
X = df_housing[['housing_median_age']]  # Tek özellik olarak medyan konut yaşı
y = (df_housing['median_house_value'] > df_housing['median_house_value'].median()).astype(int)  # Hedef değişken: medyan ev değeri yüksek mi/düşük mü?

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Logistic Regresyon Modeli Oluşturma ve Eğitme (tek özellik ile)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# Eğitim setinde tahmin yapma
prediction_train = model.predict(X_train)

# 4. Tahmin Sonuçlarını Görselleştirme (tek özellik ile)
plt.figure(figsize=(10, 6))
plt.plot(X_train, prediction_train, 'ro', label='Tahminler', alpha=0.2)
plt.xlabel('Housing Median Age (Konut Medyan Yaşı)')
plt.ylabel('Prediction (1=Yüksek Ev Değeri)')
plt.title('Logistic Regression Prediction for House Values')
plt.legend(loc='lower right')
plt.show()

# 5. Modeli Geliştirme: İki özellik kullanma (housing_median_age + price_per_sqft)
# price_per_sqft özellik sütunu oluşturma
df_housing['price_per_sqft'] = df_housing['median_house_value'] / df_housing['total_rooms']

# Eğitim ve Test Setlerini Oluşturma (iki özellik: housing_median_age + price_per_sqft)
X_train = df_housing[['housing_median_age', 'price_per_sqft']].dropna()
y_train = (df_housing['median_house_value'] > df_housing['median_house_value'].median()).astype(int)

X_test = df_housing[['housing_median_age', 'price_per_sqft']].dropna()
y_test = (df_housing['median_house_value'] > df_housing['median_house_value'].median()).astype(int)

# Özellikleri Ölçekleme
std_scale = StandardScaler()
X_train_scaled = std_scale.fit_transform(X_train)
X_test_scaled = std_scale.transform(X_test)

# Logistic Regresyon Modeli Oluşturma ve Eğitme (iki özellik ile)
lm2 = LogisticRegression(solver='liblinear')
lm2.fit(X_train_scaled, y_train)

# Eğitim ve test doğruluk skorları
print("Training Accuracy:", round(lm2.score(X_train_scaled, y_train), 3))
print("Testing Accuracy:", round(lm2.score(X_test_scaled, y_test), 3))

# 6. Modeli Pickle ile Kaydetme ve Yeni Bir Tahmin Yapma
# Modeli pickle ile kaydetme
dump(lm2, 'logreg_model.pkl')

# Modeli yükleme ve yeni bir tahmin yapma
lm_loaded = load('logreg_model.pkl')

# Gerçek hayat senaryosu: Yeni bir tahmin yapma
example_df = pd.DataFrame({
    'housing_median_age': [10],
    'price_per_sqft': [937]
})

example_scaled = std_scale.transform(example_df)
prediction = lm_loaded.predict(example_scaled)

print(f"New prediction for the example: {prediction}")

# 7. Pairplot ile Özellikleri İnceleme
# Pairplot ile konut fiyatlarını etkileyen özelliklerin ilişkilerini görselleştirme

sns.pairplot(df_housing[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                         'households', 'median_income', 'latitude', 'longitude', 'price_per_sqft',
                         'median_house_value']])
plt.show()
