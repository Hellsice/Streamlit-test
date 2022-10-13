#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install xlrd
# !pip install openpyxl
# !pip install geopy
# !pip install streamlit-folium

import requests
import pandas as pd
from pandas import json_normalize
import plotly.express as px
import streamlit as st 
import numpy as np
import folium
from folium import plugins
import streamlit_folium as st_folium
from streamlit_folium import folium_static
import calendar
from datetime import timedelta
import plotly.graph_objects as go
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# !pip install session.info
# import session_info
# session_info.show()


# In[3]:


st.title('2022-2023 sem-1 Case 3: Dashboard')
st.header('Elektrisch mobiliteit en laadpalen')
st.subheader(' Team 7: Olger, Bart, Annika, Estelle') 
##### Inleidende tekst 
st.markdown('In dit dashboard hebben wij gebruik gemaakt van de datasets OpenChargeMap, laadpaalgebruik en aantallen elektrische auto’s.')
st.markdown('Hiermee hebben wij verbanden kunnen leggen over de toename van elektrische autos, tijd dat elektrische auto’s aan de laadpaal zitten.')


# In[4]:


st.subheader('Aantallen elektrische autos')
df = pd.read_csv('voertuigen.csv')
df = df.drop(columns=['Unnamed: 0'])


st.markdown('De data over het onderwerp aantal elektrische autos is afkomstig van de website opendata.rdw')
st.markdown('Hieruit zijn er twee datasets gedownload voor gebruik. De ene over elektrische autos en ander over brandstof gebruik.')
st.markdown('Deze twee datasets zijn samengevoegd tot de nieuwe opgeschoonde dataset: "Open bezine"')
st.markdown('Doormiddel van functie zijn er extra kolomen aan toegevoegd.')
st.markdown('Zie hier dataset hieronder')

st.dataframe(df)


# In[5]:


st.markdown('Vanuit deze "Open bezine" dataset zijn er visualisaties gemaakt.')
st.markdown('Figuur hieronder bevat een cumulatieve lijndiagram van het aantal voertuigen per maand per brandstof categorie.')
st.markdown('Zoals te zien blijkt het aantal elektische voertuigen in de loop van de jaren meer toeneemt dan over brandstog categorien')


### grafiek 1 maken:
fig1 = px.ecdf(df, x="maand_tenaamstelling", 
              color="Brandstof omschrijving", 
              ecdfnorm=None)

fig1.update_layout(
    title="Cumalatief aantal voertuigen per maand per brandstof categorie. ", 
    xaxis_title="Per maand door de jaren heen",
    yaxis_title="Aantal")

st.plotly_chart(fig1)


# In[6]:


##Dataset met alleen elektrische auto's : df_elek
df_elek = df[(df['Brandstof omschrijving']=='Elektriciteit')]

st.markdown('Figuur hieronder bevat gegevens over de aanschaf elekrische autos over de jaren heen weer')
st.markdown('Door onder aan het figuur de slider te gebruiken kan er gezien worden welke auto merk in dat jaar erbij is gekomen')
st.markdown('Weer valt er op te merken dat er veel meer elektrische autos in de laatste jaren toeneemt')
## Figuur 2: Aanschaf elektishce autos over de tijd
fig2 = px.bar(data_frame=df_elek,   
                 x='Merk',  
                 animation_frame="jaar_tenaamstelling"
            )

fig2.update_layout(
    {'yaxis': {'range': [0, 10000]},'xaxis': {'range': [-1, 80]}}
)
fig2['layout'].pop('updatemenus')
fig2['layout']['sliders'][0]['pad']=dict(r= 1, t= 150,)

fig2.update_layout(
    title="Aanschaf elektrische auto’s over tijd",
    xaxis_title="Jaar",
    yaxis_title="Aantal")

st.plotly_chart(fig2)


# In[7]:


st.subheader('Chargemap') 

# maxresults=10000
# url_chargemap = ("https://api.openchargemap.io/v3/poi?key=a386a50f-1e5d-4021-baaf-868394bc33e9/?output=json&countrycode=NL&maxresults="+str(maxresults))
# response_chargemap = requests.get(url_chargemap)
# json_chargemap = response_chargemap.json()
# df_chargemap = pd.json_normalize(json_chargemap)

# columns = ['Operator', 'Comments', 'DataProvider', 'PercentageSimilarity', 'MediaItems']
# for i in columns:
#     df_chargemap = df_chargemap[df_chargemap.columns.drop(list(df_chargemap.filter(regex=i)))]
# df_chargemap = df_chargemap.dropna(axis=1, how='all')

# df_chargemap['AddressInfo.Postcode'] = df_chargemap['AddressInfo.Postcode'].str.replace(r'\D', '', regex=True)

# df_chargemap = df_chargemap[df_chargemap['AddressInfo.Postcode'].notna()]
merged = pd.read_csv('merged.csv')
df_chargemap = pd.read_csv('Chargemap data.csv')
st.dataframe(merged)
st.dataframe(df_chargemap)

# df_chargemap = df_chargemap.merge(merged, left_on='AddressInfo.Postcode', right_on='PC6', how='left')

# unique_chargemap = df_chargemap['AddressInfo.Town'].unique()
# gemeenten = gemeente_codes['Gemeentenaam'].unique()
# unique_gemeenten = [x for x in unique_chargemap if x in gemeenten]


map = folium.Map(location = [52.2129919, 5.2793703], zoom_start=7, tiles=None)
base_map = folium.FeatureGroup(name='Basemap', overlay=True, control=False)
folium.TileLayer(tiles='OpenStreetMap').add_to(base_map)
base_map.add_to(map)
All_markers = folium.FeatureGroup(name='all', overlay=False, control=True)
#for index, row in df_chargemap.iterrows():
#    All_markers.add_child(folium.Marker(location=[row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
#                                                       popup=row['AddressInfo.AddressLine1'])).add_to(map)
for i in merged['Gemeentenaam'].unique():
    globals()['%s' %i] = folium.FeatureGroup(name=i, overlay = False, control = True)
    for index, row in df_chargemap.iterrows():
        if row['Gemeentenaam'] == i:
            globals()['%s' %i].add_child(folium.Marker(location=[row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
                                                       popup=row['AddressInfo.AddressLine1'])).add_to(map)
folium.LayerControl(position='bottomleft', collapsed=False).add_to(map)
st_data = folium_static(map)


st.markdown('Cluster map')
mapcluster = folium.Map(location = [52.2129919, 5.2793703], zoom_start=7, tiles=None)
base_map = folium.FeatureGroup(name='Basemap', overlay=True, control=False)
folium.TileLayer(tiles='OpenStreetMap').add_to(base_map)
base_map.add_to(mapcluster)

cluster = folium.plugins.MarkerCluster(name='Clusters', overlay=False, control=True).add_to(mapcluster)
all_clusters = folium.plugins.FeatureGroupSubGroup(group=cluster, name='All', show=False)
for index, row in df_chargemap.iterrows():
    all_clusters.add_child(folium.Marker(location=[row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
    popup=row['AddressInfo.AddressLine1'])).add_to(mapcluster)


for i in merged['Gemeentenaam'].unique():
    globals()['%s' %i] = folium.plugins.FeatureGroupSubGroup(group=cluster, name=i, show=False)
    mapcluster.add_child(globals()['%s' %i])
    for index, row in df_chargemap.iterrows():
        if row['Gemeentenaam'] == i:
            globals()['%s' %i].add_child(folium.Marker(location=[row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
                                                       popup=row['AddressInfo.AddressLine1'])).add_to(mapcluster)

folium.LayerControl(position='bottomleft', collapsed=False).add_to(mapcluster)
st_data = folium_static(mapcluster)


st.markdown('Cluster and marker map')
map2 = folium.Map(location = [52.2129919, 5.2793703], zoom_start=7, tiles=None)
base_map = folium.FeatureGroup(name='Basemap', overlay=True, control=False)
folium.TileLayer(tiles='OpenStreetMap').add_to(base_map)
base_map.add_to(map2)
marker_cluster = folium.plugins.MarkerCluster(name='Clusters', overlay=False, control=True).add_to(map2)
for index, row in df_chargemap.iterrows():
    folium.Marker(location=[row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
                                                       popup=row['AddressInfo.AddressLine1']).add_to(marker_cluster)
    
all_markers = folium.FeatureGroup(name='All markers', overlay=False, control=True)    
map2.add_child(all_markers)

for i in merged['Gemeentenaam'].unique():
    globals()['%s' %i] = folium.plugins.FeatureGroupSubGroup(group=all_markers, name=i, show=False)
    map2.add_child(globals()['%s' %i])
    for index, row in df_chargemap.iterrows():
        if row['Gemeentenaam'] == i:
            globals()['%s' %i].add_child(folium.Marker(location=[row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
                                                       popup=row['AddressInfo.AddressLine1'])).add_to(map2)
folium.LayerControl(position='bottomleft', collapsed=False).add_to(map2)
st_data = folium_static(map2)
# In[10]:


st.subheader('Laadpaaldata') 
st.markdown("Om de date op te schonen en overzichtelijk te maken hebben we de volgende bewerkingen uitgevoerd:")
st.markdown("Allereerst hebben we alle NA-waardes verwijderd.")
st.markdown("Vervolgens hebben we alle datums omgezet naar datetime format.")
st.markdown("Toen hebben we gegekeken of de aangesloten uren volgens de Ended-Started time wel overeen kwamen,")
st.markdown("hierbij waren er 5 erg afwijkend, dus hebben we deze verwijderd.")
st.markdown("Hevige 'uitschieters' hebben we verwijderd om de meest voorkomende oplaadtijden te kunnen plotten voor een beter overzicht")
st.markdown("De gemiddelde en mediaan zijn berekend om te implementeren in de volgende figuren.")
st.markdown("")

laadpaaldata = pd.read_csv('laadpaaldata.csv')
# print(laadpaaldata.isna().sum().sum()) # Geen NaN waardes te vinden in deze dataset

# De waarnemingen waarbij het opladen later begint dan eindigt verwijderen, dit is immers onmogelijk en wij hebben er geen goede verklaring voor
laadpaaldata = laadpaaldata[laadpaaldata['Ended']>=laadpaaldata['Started']]

# De tijden omzetten naar een datetime waarde
laadpaaldata['Started'] =  pd.to_datetime(laadpaaldata['Started'], format='%Y-%m-%d  %H:%M:%S', errors='coerce') # Day is out of range for month, dus errors='coerce'
laadpaaldata['Ended'] =  pd.to_datetime(laadpaaldata['Ended'], format='%Y-%m-%d  %H:%M:%S', errors='coerce') # Day is out of range for month, dus errors='coerce'
laadpaaldata['Maand'] = pd.DatetimeIndex(laadpaaldata['Started']).month

# Tijd berekenen dat de auto aangesloten staat aan de laadpaal en het verschil met de daadwerkelijke oplaadtijd
laadpaaldata['LaadsessieAangesloten'] = laadpaaldata['Ended'] - laadpaaldata['Started']
laadpaaldata['LaadsessieAangeslotenUren'] = laadpaaldata['LaadsessieAangesloten'].dt.components['days']*24 + laadpaaldata['LaadsessieAangesloten'].dt.components['hours'] + laadpaaldata['LaadsessieAangesloten'].dt.components['minutes']/60 + laadpaaldata['LaadsessieAangesloten'].dt.components['seconds']/(60*60)
laadpaaldata['ConnectedAangeslotenDif'] = laadpaaldata['ConnectedTime'] - laadpaaldata['LaadsessieAangeslotenUren']

# De 5 waardes waarbij het verschil in ChargeTime en AangeslotenTime daadwerkelijk meet dan 0,1 uur verschilt
laadpaaldata = laadpaaldata[((laadpaaldata['ConnectedAangeslotenDif'] < 0.1) & (laadpaaldata['ConnectedAangeslotenDif'] > 0)) | ((laadpaaldata['ConnectedAangeslotenDif'] > -0.1) & (laadpaaldata['ConnectedAangeslotenDif'] < 0))]

# Hele lange tijden van laadsessie, de aansluiting, eruit halen
q_laad = laadpaaldata['LaadsessieAangeslotenUren'].quantile(0.965)
laadpaaldata[laadpaaldata['LaadsessieAangeslotenUren']<q_laad]
q_low_laad = laadpaaldata['LaadsessieAangeslotenUren'].quantile(0.035)
q_hi_laad = laadpaaldata['LaadsessieAangeslotenUren'].quantile(0.965)
laadpaaldata = laadpaaldata[(laadpaaldata["LaadsessieAangeslotenUren"] < q_hi_laad) & (laadpaaldata["LaadsessieAangeslotenUren"] > q_low_laad)]

# Hele lange tijden van chargetime eruit halen
q_charge = laadpaaldata['ChargeTime'].quantile(0.965)
laadpaaldata[laadpaaldata['ChargeTime']<q_charge]
q_low_charge = laadpaaldata['ChargeTime'].quantile(0.35)
q_hi_charge = laadpaaldata['ChargeTime'].quantile(0.965)
laadpaaldata = laadpaaldata[(laadpaaldata["ChargeTime"] < q_hi_charge) & (laadpaaldata["ChargeTime"] > q_low_charge)]

# print(laadpaaldata.head())
st.dataframe(laadpaaldata)

# Dubbel check of de uitschieters weg zijn
# fig = px.box(laadpaaldata, x="ChargeTime")
# fig.show()
# fig = px.box(laadpaaldata, x="LaadsessieAangeslotenUren")
# fig.show()

gemiddelde = round(laadpaaldata['ChargeTime'].mean(), 4)
mediaan = round(laadpaaldata['ChargeTime'].median(), 4)
# print('Gemiddelde: ' +str(int(gemiddelde))+ " uur en " +str(round((gemiddelde-int(gemiddelde))*60)) +" minuten")
# print('Mediaan: ' +str(int(mediaan))+ " uur en " +str(round((mediaan-int(mediaan))*60)) +" minuten")

st.markdown('In onderstaand figuur is de verdeling te zien van de tijd dat een auto oplaadt.')
st.markdown('Het gemiddelde en de mediaan verschillen weinig. Wel is er duidelijk te zien dat zich twee toppen vormen,')
st.markdown("een bimodale normale verdeling waarschijnlijk. De meeste auto's laden iets meer dan 2 uur op.")

fig3 = px.histogram(laadpaaldata, x="ChargeTime", nbins=40, text_auto=True,
            title='Aantal laadpalen per oplaadtijd',
            labels={"ChargeTime":"Oplaadtijd (in uren)"})

fig3.add_annotation(x=6, y=750,
            text='Gemiddelde: ' +str(int(gemiddelde))+ " uur en " +str(round((gemiddelde-int(gemiddelde))*60)) +" minuten",
            showarrow=False,
            arrowhead=1,
            align="center",
            font=dict(color="#ffffff"),
            ax=20,
            ay=-30,
            bordercolor="#ffffff",
            borderwidth=2,
            borderpad=4,
            bgcolor="#27285C",
            opacity=0.8)

fig3.add_annotation(x=5.97, y=680,
            text='Mediaan: ' +str(int(mediaan))+ " uur en " +str(round((mediaan-int(mediaan))*60)) +" minuten",
            showarrow=False,
            arrowhead=1,
            align="center",
            font=dict(color="#ffffff"),
            ax=20,
            ay=-30,
            bordercolor="#ffffff",
            borderwidth=2,
            borderpad=4,
            bgcolor="#27285C",
            opacity=0.8)

fig3.add_annotation(x=5.93, y=610,
            text='Kansdichtheid: Bimodale verdeling / tweetoppige normale verdeling',
            showarrow=False,
            arrowhead=1,
            align="center",
            font=dict(color="#ffffff"),
            ax=20,
            ay=-30,
            bordercolor="#ffffff",
            borderwidth=2,
            borderpad=4,
            bgcolor="#27285C",
            opacity=0.8)

fig3.update_layout(yaxis_title="Aantal laadpalen")
st.plotly_chart(fig3)

# Groeperen op maand en de totale laad en connected time berekenen --> verwerken in een barplot
laadtijd_per_maand = laadpaaldata.groupby('Maand')['ChargeTime', "ConnectedTime"].sum().reset_index()
laadtijd_per_maand['Maand'] = laadtijd_per_maand['Maand'].astype(int)
laadtijd_per_maand['Maand'] = laadtijd_per_maand['Maand'].apply(lambda x: calendar.month_abbr[x])

st.markdown('In onderstaand figuur is een duidelijk verschil te zien in de tijd van aansluiten en daadwerkelijk opladen.')
st.markdown("De auto's staan in totaal vaak twee keer zo lang aangesloten als dat ze aan het opladen zijn.")

fig4 = go.Figure()
fig4.add_trace(go.Bar(x=laadtijd_per_maand["Maand"], y=laadtijd_per_maand["ChargeTime"], name="Oplaadtijd"))
fig4.add_trace(go.Bar(x=laadtijd_per_maand["Maand"], y=laadtijd_per_maand["ConnectedTime"], name="Aansluittijd"))

fig4.update_layout(yaxis_title="Totale tijd in uren",
                    xaxis_title="Maand",
                    title="Totale laadtijd vs aansluittijd per maand",
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Beiden",
                     method="update",
                     args=[{"visible": [True, True]}]),
                dict(label="Aansluittijd",
                     method="update",
                     args=[{"visible": [False, True]}]),
                dict(label="Oplaadtijd",
                     method="update",
                     args=[{"visible": [True, False]}])]
))
                  ])

st.plotly_chart(fig4)


# In[13]:


st.subheader("Voorspelling aantal elektrische auto's") 
st.markdown("Doormiddel van de variabelen in de datasets is er een voorspelling gedaan over het aantal elektrische auto's.")
st.markdown('Dit hebben wij kunnen doen door bepaalde gegevens te voorspellen en uit te zetten over de jaren later')
st.markdown("Uit de figuur hieronder blijkt dat het aantal elektrische auto's tot met 2030 zal toenemen'")
st.markdown('De blauwe bolletjes geven de echte waardes aan in de jaren die al gegeven zijn')
st.markdown('De blauwe lijn geeft het gemiddelde aan door de blauwe bolletjes')
st.markdown('Om te voorspellen wat er in de jaren na de blauwe bolletjes gebeurt,')
st.markdown('zijn er "predictions" gemaakt die zijn gebaseerd op de gegeven punten')
st.markdown('De rode bolletjes vormen de voorspelling in de toekomst')
st.markdown("Hieruit zal blijken dat de aanschaf van elektrische auto's toeneemt in de toekomst")

# Create sqrt_dist_to_mrt_m

aantal_per_jaar = df_elek.groupby('jaar_tenaamstelling')['Kenteken'].count().reset_index()

groter_2017jaar = aantal_per_jaar[(aantal_per_jaar['jaar_tenaamstelling'] >= 2017)]

groter_2017jaar['jaar_tenaamstelling'] = groter_2017jaar['jaar_tenaamstelling'].astype(np.int64)
groter_2017jaar.jaar_tenaamstelling.dtype

regressie = ols("Kenteken~ jaar_tenaamstelling", data=groter_2017jaar).fit()

explanatory_data = pd.DataFrame({"jaar_tenaamstelling":np.arange(2021,2030)})
prediction_data = explanatory_data.assign(Kenteken=regressie.predict(explanatory_data))

tips = sns.load_dataset("tips")

fig5 = plt.figure()

ax1 = sns.regplot(x="jaar_tenaamstelling",
            y="Kenteken",            
            ci=None,            
            data=groter_2017jaar,)
ax2 = sns.scatterplot(x="jaar_tenaamstelling",
                y="Kenteken",                
                data=prediction_data,                 
                color="red",                
                marker="s")
ax1.set(xlabel='Jaar', ylabel="Aantal elektrische auto's")
ax1.set(title="Aantal elektrische auto's per jaar en voorspelling aankomende jaren")

st.pyplot(fig5)


# In[ ]:




