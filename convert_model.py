import joblib
import xgboost as xgb

# Load model từ file .pkl đã nén
model = joblib.load("xgb_high_model.pkl")

# Save sang JSON (nhẹ hơn, deploy dễ hơn)
model.save_model("xgb_high_model.json")

print("✅ Model đã được lưu sang JSON thành công!")
