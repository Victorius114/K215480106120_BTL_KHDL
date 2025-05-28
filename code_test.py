import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load('house_price_model.pkl')
df = pd.read_csv('data/test.csv')

features = [
    'GrLivArea', #Diện tích sử dụng
    'LotArea', #Diện tích đất
    'OverallQual', #Chất lượng tổng thể
    'YearBuilt', #Năm xây dựng
    'YearRemodAdd', #Năm gần nhất được sửa chữa
    'GarageCars', #Sức chứa xe trong gara
    'FullBath', #Số phòng tắm
    'TotalBsmtSF' #Diện tích tầng hầm
]
df = df[features].dropna()
predicted_prices = model.predict(df)

df['SalePrice'] = predicted_prices

print(df.head())

# Chia giá nhà thành các khoảng
bins = [0, 100000, 200000, 300000, 400000, 500000, float('inf')]
labels = ['<100k', '100k–200k', '200k–300k', '300k–400k', '400k–500k', '>500k']
df['PriceRange'] = pd.cut(df['SalePrice'], bins=bins, labels=labels)

# Đếm số lượng theo từng khoảng
range_counts = df['PriceRange'].value_counts().sort_index()
total = range_counts.sum()
legend_labels = [
    f"{label}: {count} ({count/total:.1%})"
    for label, count in zip(range_counts.index, range_counts)
]

# Vẽ biểu đồ tròn
colors = plt.cm.Set3.colors[:len(range_counts)]
plt.figure(figsize=(10, 6))
patches, texts, autotexts = plt.pie(range_counts, autopct='', startangle=90, colors=colors)
plt.legend(patches, legend_labels, title="Khoảng giá", loc="center left", bbox_to_anchor=(1, 0.5))
plt.title('Tỷ lệ số lượng nhà theo từng khoảng giá dự đoán')
plt.tight_layout()
plt.show()