import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and data
model = joblib.load('house_price_model.pkl')
df_test = pd.read_csv('data/test.csv')
df_train = pd.read_csv('data/train.csv')

# Define input features
features = [
    'GrLivArea',      # Diện tích sử dụng
    'LotArea',        # Diện tích đất
    'OverallQual',    # Chất lượng tổng thể
    'YearBuilt',      # Năm xây dựng
    'YearRemodAdd',   # Năm gần nhất được sửa chữa
    'GarageCars',     # Sức chứa xe trong gara
    'FullBath',       # Số phòng tắm
    'TotalBsmtSF'     # Diện tích tầng hầm
]

st.set_page_config(page_title="Dự đoán giá nhà", layout="wide")
st.title("🏠 DỰ ĐOÁN GIÁ NHÀ")

# --- User Input ---
st.header("📋 Nhập thông tin nhà")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    inputs = {}
    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            label = {
                'GrLivArea': 'Diện tích sử dụng (sqft)',
                'LotArea': 'Diện tích đất (sqft)',
                'OverallQual': 'Chất lượng tổng thể (1-10)',
                'YearBuilt': 'Năm xây dựng',
                'YearRemodAdd': 'Năm sửa chữa gần nhất',
                'GarageCars': 'Sức chứa xe trong gara',
                'FullBath': 'Số phòng tắm',
                'TotalBsmtSF': 'Diện tích tầng hầm (sqft)'
            }.get(feature, feature)

            value = st.number_input(label, min_value=0, step=1, key=feature)
            inputs[feature] = value

    submitted = st.form_submit_button("🚀 Dự đoán")

if submitted:
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]
    prediction_vnd = prediction * 25900

    st.success(f"💲 Giá nhà dự đoán: **{prediction:,.0f} USD**")
    st.success(f"🇻🇳 Tương đương khoảng **{prediction_vnd:,.0f} VND**")

# --- Biểu đồ ---
st.header("📈 Phân tích dữ liệu nhà")

# Biểu đồ đường: giá trung bình theo năm xây dựng
avg_price_by_year = df_train.groupby('YearBuilt')['SalePrice'].mean()

fig_line, ax = plt.subplots(figsize=(12, 6))
ax.plot(avg_price_by_year.index, avg_price_by_year.values, marker='o', color='teal')
ax.set_title("Giá nhà trung bình theo năm xây dựng", fontsize=14)
ax.set_xlabel("Năm xây dựng")
ax.set_ylabel("Giá trung bình (USD)")
ax.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig_line)

# Dự đoán toàn bộ test set để vẽ biểu đồ tròn
df_test_clean = df_test[features].dropna()
predicted_prices = model.predict(df_test_clean)
df_test_clean['SalePrice'] = predicted_prices

bins = [0, 100000, 200000, 300000, 400000, 500000, float('inf')]
labels = ['<100k', '100k–200k', '200k–300k', '300k–400k', '400k–500k', '>500k']
df_test_clean['PriceRange'] = pd.cut(df_test_clean['SalePrice'], bins=bins, labels=labels)

range_counts = df_test_clean['PriceRange'].value_counts().sort_index()
total = range_counts.sum()
legend_labels = [f"{label}: {count} ({count/total:.1%})"
                 for label, count in zip(range_counts.index, range_counts)]

fig_pie, ax2 = plt.subplots(figsize=(8, 6))
colors = plt.cm.Set3.colors[:len(range_counts)]
patches, texts, autotexts = ax2.pie(range_counts, autopct='', startangle=90, colors=colors)
ax2.legend(patches, legend_labels, title="Khoảng giá", loc="center left", bbox_to_anchor=(1, 0.5))
ax2.set_title("Tỷ lệ số lượng nhà theo khoảng giá")
st.pyplot(fig_pie)
