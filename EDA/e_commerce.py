import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import seaborn as sns
import os


def path_join(dir_path, file_name):
    return os.path.join(dir_path, file_name)


def create_time_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayname'] = df.index.day_name()
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    df['timeofday'] = pd.cut(df['hour'], bins=[-1, 5, 11, 17, 23],
                             labels=['Night', 'Morning', 'Afternoon', 'Evening'])

    return df


@st.cache_resource
def load_data():
    raw_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    olist_customer = pd.read_csv(path_join(raw_path, 'olist_customers_dataset.csv'))
    olist_geolocation = pd.read_csv(path_join(raw_path, 'olist_geolocation_dataset.csv'))
    olist_orders = pd.read_csv(path_join(raw_path, 'olist_orders_dataset.csv'))
    olist_order_items = pd.read_csv(path_join(raw_path, 'olist_order_items_dataset.csv'))
    olist_order_payments = pd.read_csv(path_join(raw_path, 'olist_order_payments_dataset.csv'))
    olist_order_reviews = pd.read_csv(path_join(raw_path, 'olist_order_reviews_dataset.csv'))
    olist_products = pd.read_csv(path_join(raw_path, 'olist_products_dataset.csv'))
    olist_sellers = pd.read_csv(path_join(raw_path, 'olist_sellers_dataset.csv'))
    return {
        "customers": olist_customer,
        "geolocation": olist_geolocation,
        "orders": olist_orders,
        "order_items": olist_order_items,
        "order_payments": olist_order_payments,
        "order_reviews": olist_order_reviews,
        "products": olist_products,
        "sellers": olist_sellers
    }


def show_olist_page():
    st.title("Explore Olist E-commerce Data")

    # Prepare data with time features
    data_dict = load_data()

    olist_orders = data_dict["orders"]
    olist_orders = olist_orders.set_index('order_purchase_timestamp')
    olist_orders.index = pd.to_datetime(olist_orders.index)
    olist_orders = olist_orders.sort_index()
    olist_orders = create_time_features(olist_orders)

    ### Show total orders over time ###
    st.write("### Total orders over time")

    # Line plot of total orders over time
    counts = olist_orders['order_status'].value_counts()
    fig = plt.figure(figsize=(8, 4))
    sns.barplot(counts)
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.tight_layout()
    st.pyplot(fig)
