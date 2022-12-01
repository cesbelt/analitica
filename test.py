import mysql.connector
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cufflinks as cf
import streamlit as st
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from config import *
import warnings

st.set_page_config(
    page_title = "Test de DataFrames Aimbridge",
    page_icon = "",
    layout="wide",
)   

config = {
    'user': USERNAME,
    'password': PASSWORD,
    'host': HOST,
    'database': DATABASE,
    'raise_on_warnings': True
}

cnx = mysql.connector.connect(**config)

tables = ['DM_Brands', 'DM_Business_Dates', 'DM_Chains', 'DM_Cities',
          'DM_Countries', 'DM_Currencies', 'DM_Regions', 'DM_Regions',
          'DM_States', 'DM_Venues', 'DM_Verticals', 'FT_Nights']


table_queries = {}

for table in tables:
    table_queries[table] = f'SELECT * FROM {table}'


def create_dfs(names):
    dfs = []
    for x in names:
        df = pd.read_sql(table_queries[x], cnx)
        dfs.append(df)
    return dfs
    
dfs = create_dfs(tables)

DM_Brands_df = pd.read_sql(table_queries['DM_Brands'], cnx)
DM_Business_Dates_df = pd.read_sql(table_queries['DM_Business_Dates'], cnx)
DM_Chains_df = pd.read_sql(table_queries['DM_Chains'], cnx)
DM_Cities_df = pd.read_sql(table_queries['DM_Cities'], cnx)
DM_Countries_df = pd.read_sql(table_queries['DM_Countries'], cnx)
DM_Currencies_df = pd.read_sql(table_queries['DM_Currencies'], cnx)
DM_Regions_df = pd.read_sql(table_queries['DM_Regions'], cnx)
DM_States_df = pd.read_sql(table_queries['DM_States'], cnx)
DM_Venues_df = pd.read_sql(table_queries['DM_Venues'], cnx)
DM_Verticals_df = pd.read_sql(table_queries['DM_Verticals'], cnx)
FT_Nights_df = pd.read_sql(table_queries['FT_Nights'], cnx)

DM_Venues_df = DM_Venues_df.sort_values(by = ['venue_id'])

merged_df = pd.merge(FT_Nights_df, DM_Business_Dates_df, on="business_date_id", how="left")
merged_df = merged_df.sort_values(by=['venue_id'])

merged_df2 = pd.merge(merged_df, DM_Venues_df, on = 'venue_id', how='left')


active_hotels_df = merged_df2[merged_df2['active'] == 1]
active_hotels_df = active_hotels_df[active_hotels_df['venue_id'] <= 60]
active_hotels_df = active_hotels_df[active_hotels_df['found'] == 1]
active_hotels_df = active_hotels_df.drop(columns=['latitude','longitude'])
active_hotels_df = active_hotels_df.rename(columns={"name": "hotel_name"})

active_hotels_df2 = active_hotels_df.merge(
    DM_Verticals_df,
    on='vertical_id',
    how='left'
)
active_hotels_df2 = active_hotels_df2.drop(columns=['vertical_id'])
active_hotels_df2 = active_hotels_df2.rename(columns={"name": "vertical_id"})


Chains_Brands_df = DM_Brands_df.merge(
    DM_Chains_df,
    on='chain_id',
    how='left'
)
Chains_Brands_df = Chains_Brands_df.drop(columns=['chain_id'])


active_hotels_df3 = active_hotels_df2.merge(
    Chains_Brands_df,
    on='brand_id',
    how='left'
)
active_hotels_df3 = active_hotels_df3.drop(columns=['brand_id'])
active_hotels_df3 = active_hotels_df3.rename(columns={"name_x": "brand_id","name_y": "chain_id" })

active_hotels_df4 = active_hotels_df3.merge(
    DM_Regions_df,
    on='region_id',
    how='left'
)
active_hotels_df4 = active_hotels_df4.drop(columns=['region_id'])
active_hotels_df4 = active_hotels_df4.rename(columns={"name": "region_id"})

active_hotels_df5 = active_hotels_df4.merge(
    DM_Cities_df,
    on='city_id',
    how='left'
)
active_hotels_df5 = active_hotels_df5.drop(columns=['city_id'])
active_hotels_df5 = active_hotels_df5.rename(columns={"name": "city_id"})


final_df_names = active_hotels_df5.drop(columns=['night_id','business_date_id', 'found', 'active','venue_id', 'incode', 'postal_code','day','month','year','week_day','day_of_year','month_name','iso_week_year','hotel_name','currency_id','available_rooms','rooms_revenue','compset_rooms_revenue'])


df_combined_final = active_hotels_df.drop(columns=['postal_code','day','month','year',
'week_day','day_of_year','month_name','iso_week_year','hotel_name', 'week_day_name', 'iso_week_number', 'month_name_short'])
df_combined_final.business_date = pd.to_datetime(df_combined_final.business_date)
refinity_df = pd.read_csv("datos_refinity.csv", encoding='latin-1')
refinity_df.business_date = pd.to_datetime(refinity_df.business_date)

combined_df = df_combined_final
combined_df = combined_df.sort_values('business_date')
combined_df = combined_df.merge(refinity_df, how='outer', on='business_date')
combined_df = combined_df.dropna()
#dropeamos NaN por ahora

df_combined_final = combined_df.drop(columns=['business_date','compset_rooms_available','compset_rooms_occupied',
'compset_rooms_revenue', 'quarter', 'currency_id', 'active', 'brand_id', 'region_id', 'city_id', 'vertical_id',
'found', 'semester'])


#--------------------------------------------------------------

def proceso_todo():
    df_combined_final = combined_df.drop(columns=['incode', 'business_date','compset_rooms_available','compset_rooms_occupied',
    'compset_rooms_revenue', 'quarter', 'currency_id', 'active', 'region_id', 'city_id', 'vertical_id',
    'found', 'semester'])

    pca = PCA()
    pca.fit_transform(df_combined_final)
    #Creamos df con la proporcion de la varianza y proporción acumulada
    pca_summary_df = pd.DataFrame({'Proporción de la varianza': pca.explained_variance_ratio_,
                                'Proporción acumulada': np.cumsum(pca.explained_variance_ratio_)})
    pca_summary_df = pca_summary_df.transpose()
    components = np.arange(pca.n_components_) + 1
    variance = pca.explained_variance_ratio_
    pca_components_df = pd.DataFrame(pca.components_.transpose(), columns = pca_summary_df.columns, index = df_combined_final.columns)

    pca = PCA()
    pca.fit(preprocessing.scale(df_combined_final))
    pca_summary_df = pd.DataFrame({'Proporción de la varianza': pca.explained_variance_ratio_,
                                'Proporción acumulada': np.cumsum(pca.explained_variance_ratio_)})
    pca_summary_df = pca_summary_df.transpose()
    pca_summary_df.columns = ['PC' + str(pc) for pc in range(1, len(pca_summary_df.columns) + 1)]
    pca_components_df = pd.DataFrame(pca.components_.transpose(), columns = pca_summary_df.columns, index = df_combined_final.columns)

    pca = PCA(n_components=8)
    componentes_pca = pca.fit_transform(df_combined_final)
    pca_summary_df = pd.DataFrame({'Proporción de la varianza': pca.explained_variance_ratio_,
                                'Proporción acumulada': np.cumsum(pca.explained_variance_ratio_)})
    pca_summary_df = pca_summary_df.transpose()
    pca_summary_df.columns = ['PC' + str(pc) for pc in range(1, len(pca_summary_df.columns) + 1)]
    components = np.arange(pca.n_components_) + 1   
    variance = pca.explained_variance_ratio_
    pca_components_df = pd.DataFrame(pca.components_.transpose(), columns = pca_summary_df.columns, index = df_combined_final.columns)
    df_pca = pd.DataFrame(data = componentes_pca, columns = ["PC1", "PC2", 'PC3', 'PC4', 'PC5', 'PC6', 'PC7','PC8']) 

    #--------------------------------------------

    columns = pca_components_df.columns
    X = df_pca[columns]
    y = df_combined_final['rooms_occupied']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy = round((regressor.score(X_test, y_test) * 100), 2)

    df_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_test = df_test.sort_index()
    df_test = pd.merge(df_test, combined_df, left_index=True, right_index=True)
    df_test = df_test.drop(columns=['night_id', 'venue_id', 'business_date_id',
       'found', 'rooms_available', 'rooms_occupied', 'rooms_revenue',
       'compset_rooms_available', 'compset_rooms_occupied',
       'compset_rooms_revenue', 'currency_id', 'quarter', 'semester', 'incode', 'brand_id', 'active', 'available_rooms',
       'vertical_id', 'region_id', 'city_id', 'Tipo de cambio', 'BMV',
       'Vacunados totales', 'Desempleo', 'Tasa de interés', 'Inflación',
       'Gasto en turismo(Total)', 'Variación del Pib'])
    df_test = df_test.set_index('business_date')
    df_test.index.name = ''
    
    df1 = df_test
    fig = df1.iplot(asFigure=True, kind='bar')
    
    return fig, accuracy


def proceso_por_hotel(hotel_name):
    
    df_temp = df_combined_final[df_combined_final['incode'] == hotel_name]
    df_temp = df_temp.drop(columns=['incode'])

    pca = PCA()
    pca.fit_transform(df_temp)
    #Creamos df con la proporcion de la varianza y proporción acumulada
    pca_summary_df = pd.DataFrame({'Proporción de la varianza': pca.explained_variance_ratio_,
                                'Proporción acumulada': np.cumsum(pca.explained_variance_ratio_)})
    pca_summary_df = pca_summary_df.transpose()
    components = np.arange(pca.n_components_) + 1
    variance = pca.explained_variance_ratio_
    pca_components_df = pd.DataFrame(pca.components_.transpose(), columns = pca_summary_df.columns, index = df_temp.columns)

    pca = PCA()
    pca.fit(preprocessing.scale(df_temp))
    pca_summary_df = pd.DataFrame({'Proporción de la varianza': pca.explained_variance_ratio_,
                                'Proporción acumulada': np.cumsum(pca.explained_variance_ratio_)})
    pca_summary_df = pca_summary_df.transpose()
    pca_summary_df.columns = ['PC' + str(pc) for pc in range(1, len(pca_summary_df.columns) + 1)]
    pca_components_df = pd.DataFrame(pca.components_.transpose(), columns = pca_summary_df.columns, index = df_temp.columns)

    pca = PCA(n_components=8)
    componentes_pca = pca.fit_transform(df_temp)
    pca_summary_df = pd.DataFrame({'Proporción de la varianza': pca.explained_variance_ratio_,
                                'Proporción acumulada': np.cumsum(pca.explained_variance_ratio_)})
    pca_summary_df = pca_summary_df.transpose()
    pca_summary_df.columns = ['PC' + str(pc) for pc in range(1, len(pca_summary_df.columns) + 1)]
    components = np.arange(pca.n_components_) + 1   
    variance = pca.explained_variance_ratio_
    pca_components_df = pd.DataFrame(pca.components_.transpose(), columns = pca_summary_df.columns, index = df_temp.columns)
    df_pca = pd.DataFrame(data = componentes_pca, columns = ["PC1", "PC2", 'PC3', 'PC4', 'PC5', 'PC6', 'PC7','PC8']) 

    #--------------------------------------------

    regressor = LinearRegression()

    columns = pca_components_df.columns
    X = df_pca[columns]
    y = df_temp.rooms_occupied

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)


    df_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_test = df_test.sort_index()
    df_test = pd.merge(df_test, combined_df, left_index=True, right_index=True)
    df_test = df_test.drop(columns=['night_id', 'venue_id', 'business_date_id',
       'found', 'rooms_available', 'rooms_occupied', 'rooms_revenue',
       'compset_rooms_available', 'compset_rooms_occupied',
       'compset_rooms_revenue', 'currency_id', 'quarter', 'semester', 'incode', 'brand_id', 'active', 'available_rooms',
       'vertical_id', 'region_id', 'city_id', 'Tipo de cambio', 'BMV',
       'Vacunados totales', 'Desempleo', 'Tasa de interés', 'Inflación',
       'Gasto en turismo(Total)', 'Variación del Pib'])
    df_test = df_test.set_index('business_date')
    df_test.index.name = ''
    
    df1 = df_test
    fig = df1.iplot(asFigure=True, kind='bar')
    
    
    final_df = combined_df.drop(columns=['night_id','business_date_id', 'found', 'active','venue_id',
    'currency_id','available_rooms','compset_rooms_revenue', 'compset_rooms_occupied', 'compset_rooms_available',
    'quarter', 'semester', 'brand_id', 'region_id', 'city_id', 'vertical_id'])

    final_df = final_df[final_df['incode'] == hotel_incode]

    final_df = final_df.sort_values(by=['business_date'])
    final_df = final_df.reset_index(drop=True)

    final_df['Occupancy_tag'] = round(final_df['rooms_occupied']*10/final_df['rooms_available'])

    ocupacion = round(final_df["Occupancy_tag"].mean(), 2)
    ingreso = round(final_df["rooms_revenue"].sum() / final_df["rooms_occupied"].sum(), 2)

    return fig, ocupacion, ingreso

#--------------------------------------------
cf.set_config_file(sharing='public', theme='ggplot', offline=True)
st.title("Modelo de Predicción - Aimbridge Prisma")
placeholders = st.empty()


with placeholders.container():


    hotel_incode = st.selectbox('Seleccione el hotel deseado:',
    active_hotels_df5['incode'].unique())

    fig1, accuracy = proceso_todo()
    fig2, ocupacion, ingreso = proceso_por_hotel(hotel_incode)

    kpi1, kpi2, kpi3 = st.columns(3)
    
    kpi1.metric(
        label = "Calificación de ocupación para el hotel seleccionado",
        value = f"{ocupacion}/10"
    )

    kpi2.metric(
        label = "Ingreso por habitación para el hotel seleccionado",
        value = f"${ingreso} pesos"
    )

    kpi3.metric(
        label = 'Accuracy del Modelo',
        value = f"{accuracy}%"
    )

    st.markdown("### Regresión con datos actuales vs Predecidos (Todos los hoteles)")
    st.plotly_chart(fig1)

    st.markdown("### Regresión con datos actuales vs Predecidos (Por Hotel)")
    st.plotly_chart(fig2)

        

    


    


    

