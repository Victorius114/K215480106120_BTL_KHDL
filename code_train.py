import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Đọc dữ liệu và xử lý
df = pd.read_csv('data/train.csv')
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
target = 'SalePrice'
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Huấn luyện
model = LinearRegression()
model.fit(X_train, y_train)

# Lưu mô hình vào file
joblib.dump(model, 'house_price_model.pkl')
print("Đã lưu mô hình")