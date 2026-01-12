# ==========================================
# 1️⃣ Import Libraries
# ==========================================
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# ==========================================
# 2️⃣ Forecasting: Total Customers
# ==========================================
df = pd.read_csv('month wise customer join.csv', header=None, names=['y', 'month'])
df['year'] = 2024

# Proper date banana
df['ds'] = pd.to_datetime(
    df['year'].astype(str) + '-' +
    df['month'].astype(str) + '-01'
)

df_prophet = df[['ds', 'y']]

model = Prophet(
    growth='linear',
    yearly_seasonality=False
)
model.fit(df_prophet)

future = model.make_future_dataframe(periods=12, freq='MS')
forecast = model.predict(future)
forecast.to_csv('total_customer_prediction.csv', index=False)

# ==========================================
# 3️⃣ Forecasting: Total Products
# ==========================================
df1 = pd.read_csv('month wise products join.csv', header=None, names=['y', 'month'])
df1['year'] = 2024
df1['ds'] = pd.to_datetime(
    df1['year'].astype(str) + '-' +
    df1['month'].astype(str) + '-01'
)

df1_prophet = df1[['ds', 'y']]

model1 = Prophet(yearly_seasonality=False, growth='linear')
model1.fit(df1_prophet)

future1 = model1.make_future_dataframe(periods=12, freq='MS')
forecast1 = model1.predict(future1)
forecast1.to_csv('total_products_prediction.csv', index=False)

# ==========================================
# 4️⃣ Forecasting: Total Orders
# ==========================================
df2 = pd.read_csv('month wise orders join.csv', header=None, names=['y', 'month'])
df2['year'] = 2024
df2['ds'] = pd.to_datetime(
    df2['year'].astype(str) + '-' +
    df2['month'].astype(str) + '-01'
)

df2_prophet = df2[['ds', 'y']]

model2 = Prophet(yearly_seasonality=False, growth='linear')
model2.fit(df2_prophet)

future2 = model2.make_future_dataframe(periods=12, freq='MS')
forecast2 = model2.predict(future2)
forecast2.to_csv('total_orders_prediction.csv', index=False)

# ==========================================
# 5️⃣ Forecasting: Gross Revenue
# ==========================================
df3 = pd.read_csv('month wise gross or net profit join.csv', header=None, names=['month', 'y', 'net'])
df3['year'] = 2024
df3['ds'] = pd.to_datetime(
    df3['year'].astype(str) + '-' +
    df3['month'].astype(str) + '-01'
)

df3_prophet = df3[['ds', 'y']]

model3 = Prophet(yearly_seasonality=False, growth='linear', changepoint_prior_scale=0.9)
model3.fit(df3_prophet)

future3 = model3.make_future_dataframe(periods=12, freq='MS')
forecast3 = model3.predict(future3)
forecast3[['yhat', 'yhat_upper', 'yhat_lower', 'trend']] = forecast3[['yhat', 'yhat_upper', 'yhat_lower', 'trend']].apply(lambda x: x.round(0).astype(int))
forecast3.to_csv('total_gross_prediction.csv', index=False)

# ==========================================
# 6️⃣ Forecasting: Net Revenue
# ==========================================
df4 = pd.read_csv('month wise gross or net profit join.csv', header=None, names=['month', 'gross', 'y'])
df4['year'] = 2024
df4['ds'] = pd.to_datetime(
    df4['year'].astype(str) + '-' +
    df4['month'].astype(str) + '-01'
)

df4_prophet = df4[['ds', 'y']]

model4 = Prophet(yearly_seasonality=False, growth='linear', changepoint_prior_scale=0.2)
model4.fit(df4_prophet)

future4 = model4.make_future_dataframe(periods=12, freq='MS')
forecast4 = model4.predict(future4)
forecast4[['yhat', 'yhat_upper', 'yhat_lower', 'trend']] = forecast4[['yhat', 'yhat_upper', 'yhat_lower', 'trend']].apply(lambda x: x.round(0).astype(int))
forecast4.to_csv('total_net_prediction.csv', index=False)

# ==========================================
# 7️⃣ Data Cleaning and Preprocessing: Customers, Orders, Products, Returns, Delivery
# ==========================================
df = pd.read_csv('customers_3yr.csv')
df1 = pd.read_csv('delivery_3yr.csv')
df2 = pd.read_csv('order_items_3yr.csv')
df3 = pd.read_csv('orders_3yr.csv')
df4 = pd.read_csv('products_3yr.csv')
df5 = pd.read_csv('returns_3yr.csv')

# Customers
df.drop_duplicates(inplace=True)
df.rename(columns={'age of customers': 'age_customers'}, inplace=True)
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['age_customers'] = df['age_customers'].astype('Int64')
df['city'] = df['city'].astype(str).str.strip().str.capitalize()
df['state'] = df['state'].astype(str).str.strip().str.capitalize()
df['tier'] = df['tier'].astype(str).str.strip().str.capitalize()
df['gender'] = df['gender'].astype(str).str.strip().str.capitalize()
df['acquisition_channel'] = df['acquisition_channel'].astype(str).str.strip().str.capitalize()

# Returns
df5.drop_duplicates(inplace=True)
df5['return_reason'] = df5['return_reason'].astype(str).str.strip().str.capitalize()
df5['return_date'] = pd.to_datetime(df5['return_date'])

# Products
df4.drop_duplicates(inplace=True)
df4['product_name'] = df4['product_name'].astype(str).str.strip().str.capitalize()
df4['category'] = df4['category'].astype(str).str.strip().str.capitalize()
df4['sub_category'] = df4['sub_category'].astype(str).str.strip().str.capitalize()
df4['brand'] = df4['brand'].astype(str).str.strip().str.capitalize()
df4['size_category'] = df4['size_category'].astype(str).str.strip().str.capitalize()
df4['launch_date'] = pd.to_datetime(df4['launch_date'])

# Orders
df3.drop_duplicates(inplace=True)
df3['order_date'] = pd.to_datetime(df3['order_date'])
df3['payment_mode'] = df3['payment_mode'].astype(str).str.strip().str.capitalize()
df3['order_status'] = df3['order_status'].astype(str).str.strip().str.capitalize()
df3['order_channel'] = df3['order_channel'].astype(str).str.strip().str.capitalize()

# Order Items
df2.drop_duplicates(inplace=True)
df2['item_status'] = df2['item_status'].astype(str).str.strip().str.capitalize()

# Delivery
df1.drop_duplicates(inplace=True)
df1['promised_date'] = pd.to_datetime(df1['promised_date'])
df1['delivered_date'] = pd.to_datetime(df1['delivered_date'])
df1['delivery_status'] = df1['delivery_status'].astype(str).str.strip().str.capitalize()

# ==========================================
# 8️⃣ Export Cleaned Data
# ==========================================
# df.to_csv('customers_data.csv', index=False)
# df1.to_csv('delivery_data.csv', index=False)
# df2.to_csv('order_items_data.csv', index=False)
# df3.to_csv('orders_data.csv', index=False)
# df4.to_csv('product_data.csv', index=False)
# df5.to_csv('returns_data.csv', index=False)
