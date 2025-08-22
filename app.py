import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# =========================
# 📌 Google Drive file IDs
# =========================
drive_files = {
    "stacking_model_compressed.pkl": "https://drive.google.com/file/d/1ADhSieG5HC2ftI3xRAeeko1lGw5T5hsH/view?usp=drive_link",
    "xgb_high_model.pkl": "https://drive.google.com/file/d/1kFdlAz42Lr0tW9ZauowVmpAb-Bjw3-Xi/view?usp=drive_link",
    "preprocessor.pkl": "https://drive.google.com/file/d/1eNTL0lfM4sgTUTEV8c_3FgTyEqGi71Sb/view?usp=drive_link",
    "feature_cols.pkl": "https://drive.google.com/file/d/1modXbe-boRu-znzHdwu9HHKmITCV_IZZ/view?usp=sharing"
}

# =========================
# 📌 Download nếu chưa có
# =========================
for fname, fid in drive_files.items():
    if not os.path.exists(fname):
        url = f"https://drive.google.com/uc?id={fid}"
        st.write(f"⬇️ Đang tải {fname} từ Google Drive...")
        gdown.download(url, fname, quiet=False)

# =========================
# 📌 Load model
# =========================
try:
    stacking_model = joblib.load('stacking_model_compressed.pkl')
    xgb_high_model = joblib.load('xgb_high_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {str(e)}")
    st.stop()


# =========================
# 📌 Hàm dự đoán
# =========================
def predict_price(input_data):
    try:
        input_df = pd.DataFrame([input_data], columns=feature_cols)

        pred_log = stacking_model.predict(input_df)
        pred = np.expm1(pred_log)[0]

        if pred > 1795:  # switch sang model high
            pred_log_high = xgb_high_model.predict(input_df)
            pred = np.expm1(pred_log_high)[0]

        return pred
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")
        return None


# =========================
# 📌 UI Streamlit
# =========================
st.set_page_config(page_title="Dự đoán Giá Thuê Nhà", layout="wide")
st.title("🏡 Dự đoán Giá Thuê Nhà")

st.markdown("""
Ứng dụng dự đoán giá thuê nhà dựa trên mô hình học máy (Stacking XGBoost + Random Forest).
""")

# Sidebar input
st.sidebar.header("Nhập thông tin căn hộ")

city_options = ['Dallas', 'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio']
state_options = ['TX', 'NY', 'CA', 'IL', 'AZ', 'PA']
source_options = ['RentDigs.com', 'ApartmentGuide', 'Other']

cityname = st.sidebar.selectbox("Thành phố", city_options, index=0)
state = st.sidebar.selectbox("Bang", state_options, index=0)
source = st.sidebar.selectbox("Nguồn dữ liệu", source_options, index=0)

square_feet = st.sidebar.slider("Diện tích (ft²)", 396, 2260, 900, 10)
bathrooms = st.sidebar.slider("Số phòng tắm", 1, 3, 1, 1)
bedrooms = st.sidebar.slider("Số phòng ngủ", 1, 4, 2, 1)
latitude = st.sidebar.number_input("Vĩ độ", 26.36, 47.72, 37.23, 0.01)
longitude = st.sidebar.number_input("Kinh độ", -122.31, -70.98, -84.56, 0.01)
month = st.sidebar.slider("Tháng đăng tin", 1, 12, 9, 1)
year = st.sidebar.slider("Năm đăng tin", 2018, 2019, 2019, 1)
day = st.sidebar.slider("Ngày đăng tin", 1, 31, 18, 1)

# Amenities
st.sidebar.subheader("Tiện ích")
amenities = {
    'amenities_Parking': st.sidebar.checkbox("Bãi đỗ xe", True),
    'amenities_Pool': st.sidebar.checkbox("Hồ bơi", True),
    'amenities_Gym': st.sidebar.checkbox("Phòng gym", True),
    'amenities_Patio/Deck': st.sidebar.checkbox("Sân hiên", False),
    'amenities_Washer Dryer': st.sidebar.checkbox("Máy giặt/sấy", False),
    'amenities_AC': st.sidebar.checkbox("Điều hòa", True),
    'amenities_Refrigerator': st.sidebar.checkbox("Tủ lạnh", True),
}

# Pets
pets_allowed = st.sidebar.radio("Thú cưng", ['None', 'Cats', 'Dogs', 'Cats and Dogs'], index=0)
pets_allowed_Cats = 1 if pets_allowed in ['Cats', 'Cats and Dogs'] else 0
pets_allowed_Dogs = 1 if pets_allowed in ['Dogs', 'Cats and Dogs'] else 0
pets_allowed_None = 1 if pets_allowed == 'None' else 0

fee = st.sidebar.selectbox("Phí", ['No', 'Yes'], index=0)
has_photo = st.sidebar.selectbox("Ảnh", ['No', 'Thumbnail', 'Yes'], index=2)
price_type = st.sidebar.selectbox("Loại giá", ['Monthly', 'Weekly'], index=0)

# Input dict
input_data = {
    'cityname': cityname, 'state': state, 'source': source,
    'square_feet': square_feet, 'bathrooms': bathrooms, 'bedrooms': bedrooms,
    'latitude': latitude, 'longitude': longitude, 'month': month, 'year': year, 'day': day,
    'fee': fee, 'has_photo': has_photo, 'price_type': price_type,
    **amenities,
    'pets_allowed_Cats': pets_allowed_Cats,
    'pets_allowed_Dogs': pets_allowed_Dogs,
    'pets_allowed_None': pets_allowed_None
}

# Predict button
if st.sidebar.button("Dự đoán"):
    predicted_price = predict_price(input_data)
    if predicted_price is not None:
        st.success(f"**Giá thuê dự đoán: ${predicted_price:,.2f}**")
    else:
        st.error("Không thể dự đoán. Vui lòng kiểm tra input.")

# Info
st.markdown("### Thông tin mô hình")
st.write("""
- **Mô hình**: Stacking (XGBoost + Random Forest) + XGBoost high
- **MAPE**: ~8.98%
""")
