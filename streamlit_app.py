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

df = pd.read_csv('voertuigen.csv')
df = df.drop(columns=['Unnamed: 0'])

### grafiek 1 maken:
fig1 = px.ecdf(df, x="maand_tenaamstelling", 
              color="Brandstof omschrijving", 
              ecdfnorm=None)

fig1.update_layout(
    title="Cumalatief aantal voertuigen per maand per brandstof categorie. ", 
    xaxis_title="Per maand door de jaren heen",
    yaxis_title="Aantal")


##Dataset met alleen elektrische auto's : df_elek
df_elek = df[(df['Brandstof omschrijving']=='Elektriciteit')]

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


merged = pd.read_csv('merged.csv')
df_chargemap = pd.read_csv('Chargemap data.csv')

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


mapcluster = folium.Map(location = [52.2129919, 5.2793703], zoom_start=7, tiles=None)
base_map = folium.FeatureGroup(name='Basemap', overlay=True, control=False)
folium.TileLayer(tiles='OpenStreetMap').add_to(base_map)
base_map.add_to(mapcluster)

cluster = folium.plugins.MarkerCluster(name='Clusters', overlay=False, control=True).add_to(mapcluster)
all_clusters = folium.plugins.MarkerCluster(name='Alle gemeenten', overlay=False, control=True).add_to(mapcluster)
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



laadpaaldata = pd.read_csv('laadpaaldata.csv')

gemiddelde = round(laadpaaldata['ChargeTime'].mean(), 4)
mediaan = round(laadpaaldata['ChargeTime'].median(), 4)
# print('Gemiddelde: ' +str(int(gemiddelde))+ " uur en " +str(round((gemiddelde-int(gemiddelde))*60)) +" minuten")
# print('Mediaan: ' +str(int(mediaan))+ " uur en " +str(round((mediaan-int(mediaan))*60)) +" minuten")


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

# Groeperen op maand en de totale laad en connected time berekenen --> verwerken in een barplot
laadtijd_per_maand = laadpaaldata.groupby('Maand')['ChargeTime', "ConnectedTime"].sum().reset_index()
laadtijd_per_maand['Maand'] = laadtijd_per_maand['Maand'].astype(int)
laadtijd_per_maand['Maand'] = laadtijd_per_maand['Maand'].apply(lambda x: calendar.month_abbr[x])


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



# Create sqrt_dist_to_mrt_m

aantal_per_jaar = df_elek.groupby('jaar_tenaamstelling')['Kenteken'].count().reset_index()

groter_2017jaar = aantal_per_jaar[(aantal_per_jaar['jaar_tenaamstelling'] >= 2017)]

groter_2017jaar['jaar_tenaamstelling'] = groter_2017jaar['jaar_tenaamstelling'].astype(np.int64)

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

#Voorpagina
st.title('2022-2023 sem-1 Case 3: Dashboard')
st.header('Elektrisch mobiliteit en laadpalen')
st.subheader(' Team 7: Olger, Bart, Annika, Estelle')
st.markdown('In dit dashboard hebben wij gebruik gemaakt van de datasets OpenChargeMap, laadpaalgebruik en aantallen elektrische auto’s.')
st.markdown('Hiermee hebben wij verbanden gelgd over de toename van elektrische autos, tijd dat elektrische auto’s aan de laadpaal zitten en de laadpalen per gemeente.')

#Elektrische auto's
st.subheader('Aantallen elektrische autos')
st.markdown('De data over het onderwerp aantal elektrische autos is afkomstig van de website opendata.rdw')
st.markdown('Hieruit zijn er twee datasets gedownload voor gebruik. De ene over elektrische autos en ander over brandstof gebruik.')
st.markdown('Deze twee datasets zijn samengevoegd tot de nieuwe opgeschoonde dataset: "Open bezine"')
st.markdown('Doormiddel van functie zijn er extra kolomen aan toegevoegd.')
st.markdown('Zie hier dataset hieronder')
st.dataframe(df)
st.markdown('Vanuit deze "Open bezine" dataset zijn er visualisaties gemaakt.')
st.markdown('Figuur hieronder bevat een cumulatieve lijndiagram van het aantal voertuigen per maand per brandstof categorie.')
st.plotly_chart(fig1)
st.markdown('Zoals te zien blijkt het aantal elektische voertuigen in de loop van de jaren meer toeneemt dan de andere brandstof categorieën')
st.markdown('Figuur hieronder bevat gegevens over de aanschaf elekrische autos over de jaren heen weer')
st.markdown('Door onder aan het figuur de slider te gebruiken kan er gezien worden welk auto merk in dat jaar erbij is gekomen')
st.markdown('Weer valt er op te merken dat er veel meer elektrische autos in de laatste jaren toeneemt')
st.plotly_chart(fig2)

#Chargemap
st.subheader('Chargemap') 
st.markdown('Met chargemap is opgevraagd wat de locaties zijn van alle laadpalen die zij geregistreerd hebben. Met de postcodes die er zijn aangegeven in de dataset en de volgende dataframe is bepaald bij welke gemeente welke laadpaal hoort.')
st.dataframe(merged)
st.markdown('Er zijn meerdere mappen gemaakt, de eerste geeft per gemeente aan waar de laadpalen staan door middel van markers.')
st_data = folium_static(map)
st.markdown('De tweede map geeft dit ook weer, maar dit keer als cluster markers. Hier is ook een optie toegevoegd voor alle gemeenten tegelijk.')
st_data = folium_static(mapcluster)
st.markdown('Deze map is een combinatie van beide. De eerste optie geeft alle laadpalen in alle gemeentes weer als een cluster marker. De tweede optie laat je kiezen welke gemeenten je wilt laten zien.')
st_data = folium_static(map2)
            
            
#Laadpaaldata
st.subheader('Laadpaaldata') 
st.markdown("Om de date op te schonen en overzichtelijk te maken hebben we de volgende bewerkingen uitgevoerd:")
st.markdown("Allereerst hebben we alle NA-waardes verwijderd.")
st.markdown("Vervolgens hebben we alle datums omgezet naar datetime format.")
st.markdown("Toen hebben we gegekeken of de aangesloten uren volgens de Ended-Started time wel overeen kwamen,")
st.markdown("hierbij waren er 5 erg afwijkend, dus hebben we deze verwijderd.")
st.markdown("Hevige 'uitschieters' hebben we verwijderd om de meest voorkomende oplaadtijden te kunnen plotten voor een beter overzicht")
st.markdown("De gemiddelde en mediaan zijn berekend om te implementeren in de volgende figuren.")
st.dataframe(laadpaaldata)
st.markdown('In onderstaand figuur is de verdeling te zien van de tijd dat een auto oplaadt.')
st.markdown('Het gemiddelde en de mediaan verschillen weinig. Wel is er duidelijk te zien dat zich twee toppen vormen,')
st.markdown("een bimodale normale verdeling waarschijnlijk. De meeste auto's laden iets meer dan 2 uur op.")
st.plotly_chart(fig3)
st.markdown('In onderstaand figuur is een duidelijk verschil te zien in de tijd van aansluiten en daadwerkelijk opladen.')
st.markdown("De auto's staan in totaal vaak twee keer zo lang aangesloten als dat ze aan het opladen zijn.")
st.plotly_chart(fig4)
            
            
#Voorspelling
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
st.pyplot(fig5)
