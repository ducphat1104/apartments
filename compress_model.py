import joblib
import os

# Đường dẫn tới file gốc
input_file = r"C:\Users\Nguyen duc phat\PycharmProjects\Dự đoán US\stacking_model.pkl"
# File sau khi nén (lưu cùng thư mục)
output_file = r"C:\Users\Nguyen duc phat\PycharmProjects\Dự đoán US\stacking_model_compressed.pkl"

# Load model gốc
print("🔄 Đang load model, chờ chút...")
model = joblib.load(input_file)

# Save lại với compress
print("💾 Đang nén model...")
joblib.dump(model, output_file, compress=3)

# In dung lượng trước / sau
print("Trước khi nén:", os.path.getsize(input_file) / (1024**2), "MB")
print("Sau khi nén:", os.path.getsize(output_file) / (1024**2), "MB")
