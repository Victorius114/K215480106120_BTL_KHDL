from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load mô hình
model = joblib.load('house_price_model.pkl')

# Tỷ giá USD sang VND
EXCHANGE_RATE = 25900

# Các feature cần thiết
features = [
    'GrLivArea',  # Diện tích sử dụng
    'LotArea',  # Diện tích đất
    'OverallQual',  # Chất lượng tổng thể
    'YearBuilt',  # Năm xây dựng
    'YearRemodAdd',  # Năm gần nhất được sửa chữa
    'GarageCars',  # Sức chứa xe trong gara
    'FullBath',  # Số phòng tắm
    'TotalBsmtSF'  # Diện tích tầng hầm
]


@app.route('/')
def home():
    # Load model và dữ liệu
    model = joblib.load('house_price_model.pkl')
    df = pd.read_csv('data/test.csv')
    df2 = pd.read_csv('data/train.csv')

    features = [
        'GrLivArea', 'LotArea', 'OverallQual',
        'YearBuilt', 'YearRemodAdd', 'GarageCars',
        'FullBath', 'TotalBsmtSF'
    ]
    df = df[features].dropna()
    predicted_prices = model.predict(df)
    df['SalePrice'] = predicted_prices

    # Vẽ biểu đồ đường - Giá nhà theo năm xây dựng
    avg_price_by_year = df2.groupby('YearBuilt')['SalePrice'].mean()
    plt.figure(figsize=(10, 5))
    plt.plot(avg_price_by_year.index, avg_price_by_year.values, marker='o', color='teal')
    plt.title('Giá nhà trung bình theo năm xây dựng')
    plt.xlabel('Năm xây dựng')
    plt.ylabel('Giá trung bình (USD)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Lưu biểu đồ
    line_chart_path = os.path.join('static', 'price_by_year.png')
    plt.savefig(line_chart_path)
    plt.close()

    return render_template("index.html", features=features)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form
        data = {feature: float(request.form[feature]) for feature in features}

        # Tạo DataFrame
        df = pd.DataFrame([data])

        # Dự đoán
        prediction = model.predict(df)
        price_usd = prediction[0]
        price_vnd = price_usd * EXCHANGE_RATE

        # Định dạng kết quả
        predicted_price_usd = "${:,.2f}".format(price_usd)
        predicted_price_vnd = "{:,.0f} VND".format(price_vnd)

        return jsonify({
            'success': True,
            'prediction_usd': predicted_price_usd,
            'prediction_vnd': predicted_price_vnd,
            'message': 'Dự đoán giá nhà thành công!'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi: {str(e)}'
        })


if __name__ == '__main__':
    app.run(debug=True)