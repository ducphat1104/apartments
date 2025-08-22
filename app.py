import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pickle")
# Kiểm tra sự tồn tại của các file .pkl
required_files = ['stacking_model_compressed.pkl', 'xgb_high_model.pkl', 'preprocessor.pkl', 'feature_cols.pkl']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(
        f"Thiếu các file: {', '.join(missing_files)}. Vui lòng đảm bảo các file này nằm trong cùng thư mục với app.py.")
    st.stop()

# Tải các mô hình và preprocessor
try:
    stacking_model = joblib.load('stacking_model_compressed.pkl')
    xgb_high_model = joblib.load('xgb_high_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {str(e)}")
    st.stop()


# Hàm dự đoán giá
def predict_price(input_data):
    try:
        # Chuyển input_data thành DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_cols)

        # Dự đoán bằng Stacking (log scale)
        pred_log = stacking_model.predict(input_df)
        pred = np.expm1(pred_log)[0]

        # Nếu giá dự đoán > 1795, dùng mô hình XGBoost giá cao
        if pred > 1795:
            pred_log_high = xgb_high_model.predict(input_df)
            pred = np.expm1(pred_log_high)[0]

        return pred
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")
        return None


# Giao diện Streamlit
st.set_page_config(page_title="Dự đoán Giá Thuê Nhà", layout="wide")

# Tiêu đề và mô tả
st.title("🏡 Dự đoán Giá Thuê Nhà")
st.markdown("""
Ứng dụng này dự đoán giá thuê nhà dựa trên mô hình học máy tối ưu (Stacking XGBoost + Random Forest).
Nhập thông tin căn hộ và nhấn **Dự đoán** để xem kết quả.
Độ tin cậy: MAPE ~8.98% (lỗi trung bình khoảng ±8.98% so với giá thực tế).
""")

# Sidebar cho input
st.sidebar.header("Nhập thông tin căn hộ")

# Danh sách thủ công cho cityname, state, source
city_options = ['Dallas', 'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio']
state_options = ['TX', 'NY', 'CA', 'IL', 'AZ', 'PA']
source_options = ['RentDigs.com', 'ApartmentGuide', 'Other']

cityname = st.sidebar.selectbox("Thành phố", options=city_options,
                                index=city_options.index('Dallas') if 'Dallas' in city_options else 0)
state = st.sidebar.selectbox("Bang", options=state_options,
                             index=state_options.index('TX') if 'TX' in state_options else 0)
source = st.sidebar.selectbox("Nguồn dữ liệu", options=source_options,
                              index=source_options.index('RentDigs.com') if 'RentDigs.com' in source_options else 0)

# Numeric features
square_feet = st.sidebar.slider("Diện tích (ft²)", min_value=396, max_value=2260, value=900, step=10)
bathrooms = st.sidebar.slider("Số phòng tắm", min_value=1, max_value=3, value=1, step=1)
bedrooms = st.sidebar.slider("Số phòng ngủ", min_value=1, max_value=4, value=2, step=1)
latitude = st.sidebar.number_input("Vĩ độ", min_value=26.36, max_value=47.72, value=37.23, step=0.01)
longitude = st.sidebar.number_input("Kinh độ", min_value=-122.31, max_value=-70.98, value=-84.56, step=0.01)
month = st.sidebar.slider("Tháng đăng tin", min_value=1, max_value=12, value=9, step=1)
year = st.sidebar.slider("Năm đăng tin", min_value=2018, max_value=2019, value=2019, step=1)
day = st.sidebar.slider("Ngày đăng tin", min_value=1, max_value=31, value=18, step=1)

# Amenities (checkbox cho top features quan trọng)
st.sidebar.subheader("Tiện ích")
amenities = {
    'amenities_Parking': st.sidebar.checkbox("Bãi đỗ xe", value=True),
    'amenities_Pool': st.sidebar.checkbox("Hồ bơi", value=True),
    'amenities_Gym': st.sidebar.checkbox("Phòng gym", value=True),
    'amenities_Patio/Deck': st.sidebar.checkbox("Sân hiên", value=False),
    'amenities_Washer Dryer': st.sidebar.checkbox("Máy giặt/sấy", value=False),
    'amenities_Storage': st.sidebar.checkbox("Kho chứa", value=False),
    'amenities_Clubhouse': st.sidebar.checkbox("Nhà câu lạc bộ", value=False),
    'amenities_Dishwasher': st.sidebar.checkbox("Máy rửa bát", value=False),
    'amenities_AC': st.sidebar.checkbox("Điều hòa", value=True),
    'amenities_Refrigerator': st.sidebar.checkbox("Tủ lạnh", value=True),
    'amenities_Fireplace': st.sidebar.checkbox("Lò sưởi", value=False),
    'amenities_Cable or Satellite': st.sidebar.checkbox("Cáp/Internet", value=False),
    'amenities_Playground': st.sidebar.checkbox("Sân chơi", value=False),
    'amenities_Internet Access': st.sidebar.checkbox("Truy cập Internet", value=False),
    'amenities_Wood Floors': st.sidebar.checkbox("Sàn gỗ", value=False),
    'amenities_Gated': st.sidebar.checkbox("Cổng bảo vệ", value=False),
    'amenities_Tennis': st.sidebar.checkbox("Sân tennis", value=False),
    'amenities_TV': st.sidebar.checkbox("TV", value=False),
    'amenities_Elevator': st.sidebar.checkbox("Thang máy", value=False),
    'amenities_Basketball': st.sidebar.checkbox("Sân bóng rổ", value=False),
    'amenities_Hot Tub': st.sidebar.checkbox("Bồn nước nóng", value=False),
    'amenities_Garbage Disposal': st.sidebar.checkbox("Máy xử lý rác", value=False),
    'amenities_View': st.sidebar.checkbox("Tầm nhìn đẹp", value=False),
    'amenities_Alarm': st.sidebar.checkbox("Hệ thống báo động", value=False),
    'amenities_Doorman': st.sidebar.checkbox("Nhân viên gác cửa", value=False),
    'amenities_Luxury': st.sidebar.checkbox("Cao cấp", value=False)
}

# Pets allowed
st.sidebar.subheader("Chính sách thú cưng")
pets_allowed = st.sidebar.radio("Cho phép thú cưng", options=['None', 'Cats', 'Dogs', 'Cats and Dogs'], index=0)
pets_allowed_Cats = 1 if pets_allowed in ['Cats', 'Cats and Dogs'] else 0
pets_allowed_Dogs = 1 if pets_allowed in ['Dogs', 'Cats and Dogs'] else 0
pets_allowed_None = 1 if pets_allowed == 'None' else 0

# Categorical features
fee = st.sidebar.selectbox("Phí", options=['No', 'Yes'], index=0)
has_photo = st.sidebar.selectbox("Ảnh", options=['No', 'Thumbnail', 'Yes'], index=2)
price_type = st.sidebar.selectbox("Loại giá", options=['Monthly', 'Weekly'], index=0)

# Tạo dictionary input
input_data = {
    'cityname': cityname,
    'state': state,
    'source': source,
    'square_feet': square_feet,
    'bathrooms': bathrooms,
    'bedrooms': bedrooms,
    'latitude': latitude,
    'longitude': longitude,
    'month': month,
    'year': year,
    'day': day,
    'fee': fee,
    'has_photo': has_photo,
    'price_type': price_type,
    **amenities,
    'pets_allowed_Cats': pets_allowed_Cats,
    'pets_allowed_Dogs': pets_allowed_Dogs,
    'pets_allowed_None': pets_allowed_None
}

# Nút dự đoán
if st.sidebar.button("Dự đoán"):
    predicted_price = predict_price(input_data)
    if predicted_price is not None:
        # Định dạng số chuẩn với dấu chấm và ký tự USD
        st.success(f"**Giá thuê dự đoán: ${predicted_price:,.2f}**")
        lower_bound = predicted_price * 0.9102  # 100% - 8.98%
        upper_bound = predicted_price * 1.0898  # 100% + 8.98%
        st.info(
            f"Độ tin cậy: ±8.98% (dựa trên MAPE của mô hình). Giá thực tế có thể dao động từ ${lower_bound:,.2f} đến ${upper_bound:,.2f}.")
    else:
        st.error("Không thể dự đoán. Vui lòng kiểm tra dữ liệu đầu vào.")

# Hiển thị thông tin mô hình
st.markdown("### Thông tin mô hình")
st.write("""
- **Mô hình**: Kết hợp Stacking (XGBoost + Random Forest) và XGBoost riêng cho giá cao (>1795 USD).
- **Hiệu suất**:
  - RMSE: 213.27
  - MAE: 131.06
  - R²: 0.9080
  - MAPE: 8.98%
- **Feature quan trọng nhất**: Thành phố (`cityname`), diện tích (`square_feet`), bang (`state`), số phòng tắm (`bathrooms`).
""")

# CSS để cải thiện giao diện
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSidebar {
        background-color: #f8f9fa;
    }
    .stSuccess {
        font-size: 1.2em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)
