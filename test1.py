import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np

# Load mô hình đã huấn luyện
model = joblib.load('house_price_model.pkl')

# Các đặc trưng đã dùng khi huấn luyện mô hình
features = [
    'GrLivArea',      # Diện tích sử dụng
    'LotArea',        # Diện tích đất
    'OverallQual',    # Chất lượng tổng thể
    'YearBuilt',      # Năm xây dựng
    'YearRemodAdd',   # Năm sửa chữa gần nhất
    'GarageCars',     # Sức chứa xe trong gara
    'FullBath',       # Số phòng tắm
    'TotalBsmtSF'     # Diện tích tầng hầm
]

# Tỷ giá USD -> VND
usd_to_vnd = 25940

# Hàm dự đoán
def predict_price():
    try:
        input_values = []
        for feature in features:
            value = entries[feature].get()
            if value == '':
                raise ValueError("Vui lòng nhập đầy đủ thông tin.")
            input_values.append(float(value))

        input_array = np.array(input_values).reshape(1, -1)
        predicted_price_usd = model.predict(input_array)[0]
        predicted_price_vnd = predicted_price_usd * usd_to_vnd

        usd_result.set(f"{predicted_price_usd:,.2f} USD")
        vnd_result.set(f"{predicted_price_vnd:,.0f} VND")
    except Exception as e:
        messagebox.showerror("Lỗi", str(e))

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Dự đoán giá nhà")
root.geometry("700x600")
root.configure(bg="#f0f4f7")

# Phông chữ tiêu đề
title_font = ("Segoe UI", 20, "bold")
label_font = ("Segoe UI", 11)
entry_font = ("Segoe UI", 11)

# Header
header = tk.Label(root, text="🏠 DỰ ĐOÁN GIÁ NHÀ", font=title_font, bg="#007bff", fg="white", pady=10)
header.pack(fill="x")

# Frame nhập liệu
form_frame = ttk.Frame(root, padding=20)
form_frame.pack(fill="both", expand=True)

# Các nhãn và ô nhập
entries = {}
row = 0
for feature in features:
    label_text = {
        'GrLivArea': "Diện tích sử dụng (sqft)",
        'LotArea': "Diện tích đất (sqft)",
        'OverallQual': "Chất lượng tổng thể (1-10)",
        'YearBuilt': "Năm xây dựng",
        'YearRemodAdd': "Năm sửa chữa gần nhất",
        'GarageCars': "Sức chứa xe trong gara",
        'FullBath': "Số phòng tắm",
        'TotalBsmtSF': "Diện tích tầng hầm (sqft)"
    }.get(feature, feature)

    ttk.Label(form_frame, text=label_text, font=label_font).grid(row=row, column=0, sticky="w", pady=5)
    entry = ttk.Entry(form_frame, font=entry_font)
    entry.grid(row=row, column=1, pady=5, padx=10, sticky="ew")
    entries[feature] = entry
    row += 1

form_frame.columnconfigure(1, weight=1)

# Nút dự đoán
predict_button = ttk.Button(root, text="🔍 Dự đoán giá", command=predict_price)
predict_button.pack(pady=10)

# Kết quả
result_frame = ttk.Frame(root, padding=20)
result_frame.pack(fill="x")

usd_result = tk.StringVar()
vnd_result = tk.StringVar()

ttk.Label(result_frame, text="Giá USD:", font=label_font).grid(row=0, column=0, sticky="e", padx=5)
ttk.Label(result_frame, textvariable=usd_result, font=("Segoe UI", 13, "bold"), foreground="#007bff").grid(row=0, column=1, sticky="w")

ttk.Label(result_frame, text="Giá VND (1 USD = 25,940 VND):", font=label_font).grid(row=1, column=0, sticky="e", padx=5)
ttk.Label(result_frame, textvariable=vnd_result, font=("Segoe UI", 13, "bold"), foreground="#28a745").grid(row=1, column=1, sticky="w")

# Footer
footer = tk.Label(root, text="Ứng dụng Machine Learning - Dự đoán giá nhà", bg="#f8f9fa", font=("Segoe UI", 9), pady=10)
footer.pack(fill="x", side="bottom")

root.mainloop()
