# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 19:17:00 2021

@author: abura
"""

import streamlit as st
import pandas as pd
import altair as alt
import os
import plotS
from plotS import *
from bbgData import *
import urllib
import datetime
from PIL import Image
import os.path
import time

@st.cache
def file_age_in_seconds(pathname):
    return time.time() - os.stat(pathname).st_mtime
def bbgDataApp():
    AWS_BUCKET_URL = "https://streamlit-demo-data.s3-us-west-2.amazonaws.com"
    df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
    return df.set_index("Region")

#def tradeconvert(t):
    
try:
    #b = bbgData()
    #b.populate('20050101')
    curr = ['USD', 'KRW', 'SGD', 'HKD', 'AUD', 'CNY', 'INR', 'JPY', 'EUR', 'TWD',
       'MYR', 'NZD', 'HUF', 'CZK', 'GBP', 'PLN']


    #dtype = st.sidebar.radio("Data: ",('Stored', 'Live'))  
    dtype = 'Stored'
    count = st.sidebar.number_input('No of Data points', value=1000)
    
    if dtype == 'Live':
            countries = st.multiselect("Choose countries", curr, ["KRW", "USD"] )
            if not countries:
                st.error("Please select at least one country.")
            else:
                    
                trade = st.sidebar.text_input("Trade Structure", '1f2s5s')
                sdate = st.sidebar.date_input('Start date', datetime.date(2011,1,1))
                e = bbgData()
                (ttf,tt) = e.analyze(countries[0]+trade,periods=count,ccylist=countries,start=sdate.strftime('%Y%m%d'),folder='plots/')
                #st.beta_set_page_config(layout="wide")
                c1, c2 = st.beta_columns((1, 1))
                with c1:
                    image1 = Image.open('plots/'+countries[0]+trade+'crossMktTradeRD.png')
                    st.image(image1, caption='Cross Market Roll Down')
                with c2:
                    image2 = Image.open('plots/'+countries[0]+trade+'crossMktTrade.png')
                    st.image(image2, caption='Cross Market Range')
                # st.write("### Gross Agricultural Production ($B)", data.sort_index())
                image3 = Image.open('plots/'+countries[0]+trade+'RollDown.png')
                st.image(image3,caption='Historical Range')
    if dtype =='Stored':
        
        b = bbgData()
        b.populate()
        #mkt_num = st.sidebar.radio("", ('Cross Mkt','Single Mkt')) 
        mkt_num = 'Cross Mkt'
        if mkt_num=='Cross Mkt':
            countries = st.multiselect("Choose countries", curr, ["KRW", "USD"] )
            if not countries:
                st.error("Please select at least one country.")
            else:    
                all_options = st.checkbox("Select all markets")
                if all_options:
                    countries = curr
                data = st.sidebar.selectbox('Explore: ',('Trade Structure', 'Spread', 'Data'))                   
                if data=='Spread':
                    availCols = b.getType(countries)
                
                    col1, col2 = st.sidebar.beta_columns(2)
                    yld1 = st.sidebar.selectbox('1st Leg: ',availCols)                   
                    yld2 = st.sidebar.selectbox('2nd Leg: ',availCols)                   
                    if yld1 != yld2:
                            
                        #yld1 =  col1.text_input("1st Leg", 'Policy')
                        #yld2 =  col2.text_input("2st Leg", 'Swap2y')
        
                        (ret,ry)=b.realyld(bond=yld2,cpi=yld1,filename='plots/realyld.png',ccylist=countries)
                        ry = ry[countries]
                        st.line_chart(ry.tail(count))
                        image5 = Image.open('plots/realyld.png')
                        st.image(image5,caption='Spread Z Score')    
                        fig = plotBarBox(ry,count=count,spread_name=yld1+" "+yld2,charts=[1,0,0],labelcount=3)
                        st.pyplot(fig)
                    else :
                        st.error("Leg1 and Leg2 should be different")

                if data =='Trade Structure':
                    instrument = st.sidebar.selectbox('Instrument: ',('Swap', 'Govt', 'FX'))                   
                    if (instrument == 'Swap') or (instrument == 'Govt'):
                        trade = st.sidebar.text_input("Trade Structure", '2y 5y')
                    else:
                        trade = st.sidebar.text_input("Trade Structure", '3M 6M')
                    trade_val = trade.split(" ")
                    changeType = st.sidebar.selectbox('Change Type: ',('Difference','Percentage'))     
                    ct=0    
                    if changeType=='Percentage':
                        ct=1
                    
                    if len(trade_val)==1:
                        spread = b.getbbg(type_=instrument+trade_val[0],fill='ffill',ccylist=countries)
                        spread.columns=b.bbgticker[b.bbgticker.bbgticker.isin(spread.columns.to_list())].Code.to_list()
                        
                    else:
                        spread=b.getSpreadforAll(type_=instrument,tenor=trade_val,ccylist=countries)
                        spread.columns = [c[0:3] for c in spread.columns.to_list()]
                    #if not all_options:
                    #    spread = spread[countries]
                    st.line_chart(spread.tail(count))

                    fig = plotRange(spread.tail(count),count=count,title=trade,filename='plots/range1.png')
                    image4 = Image.open('plots/range1.png')
                    st.image(image4,caption='Historical Range')
                    fig = plotBarBox(spread,count=count,spread_name=trade,charts=[1,0,0],labelcount=3)
                    
                    st.pyplot(fig)
                    plotChange(spread,n=[1,5,22],diff=ct, title_prefix=trade,filename='plots/spreadchange.png',mult=1)
                    image10 = Image.open('plots/spreadchange.png')
                    st.image(image10,caption='Change:')
                if data == 'Data':
                    availCols = b.getType(countries)
                
                    data = st.sidebar.selectbox('Available Fields: ',availCols)                   
                    freq = st.sidebar.selectbox('Frequency: ',('D:Daily','W:Weekly' ,'M:Monthly', 'Q:Quarterly'))     
                    changeType = st.sidebar.selectbox('Change Type: ',('Difference','Percentage'))     
                    

                    spread = b.gettsforCCYlist(type_=data,ccylist=countries,fill='ffill')
                    spread = spread.resample(freq[0]).last()
                    #spread.columns=b.bbgticker[b.bbgticker.bbgticker.isin(spread.columns.to_list())].Code.to_list()
                    st.line_chart(spread.tail(count))

                    fig = plotRange(spread.tail(count),count=count,title=data,filename='plots/range1.png')
                    image4 = Image.open('plots/range1.png')
                    st.image(image4,caption='Historical Range')
                    fig = plotBarBox(spread,count=count,spread_name=data,charts=[1,0,0],labelcount=3)
                    
                    st.pyplot(fig)
                    if freq[0] =='D':
                        changeDur = [1,5,22]
                    elif freq[0] == 'W':
                        changeDur = [1,4,12]
                    elif freq[0]=='M':
                        changeDur = [1,3,12]
                    else:
                        changeDur = [1,2,3]
                    ct=0    
                    if changeType=='Percentage':
                        ct=1
                    plotChange(spread,n=changeDur,diff=ct, title_prefix=data+ " "+freq+" ",filename='plots/spreadchange.png',mult=100)
                    image10 = Image.open('plots/spreadchange.png')
                    st.image(image10,caption='Change:')
                    
                    
        if mkt_num=='Single Mkt':
            country = st.selectbox("Choose countries", curr )
            t = bbgData()
            if (os.path.exists(country+'.bbgData')) and (file_age_in_seconds(country+'.bbgData')<600):
                t = t.read(country+'.bbgData')
            else:
                    t.genSwaptickers(country,tenor_set=2)
                    t.populate(start='20050101')
                    t.dump(country+'.bbgData')
            t.genpca(count=count,filename='plots/app')
            plotYldCurve(t,filename='plots/yc.png')
            
            image7 = Image.open('plots/appresidual.png')
            image8 = Image.open('plots/yc.png')
            st.image(image8,caption='Yield Curve')
            st.image(image7,caption='PCA Residuals')
            col1, col2 = st.sidebar.beta_columns(2)
            rate1 =  col1.text_input("1st Leg", '5y')
            rate2 =  col2.text_input("2st Leg", '2y 5y 10y')
            tenor1=rate1.split(" ")
            tenor2=rate2.split(" ")
            t.dfbbg = t.dfbbg.tail(count)                                
            c1 = t.getSpread(tenor=tenor1)
            c2 = t.getSpread(tenor=tenor2)
            plotSwap(c1,c2,filename='plots/plotSwap')
            image9 = Image.open('plots/plotSwapps.png')
            st.image(image9,caption='Curve Move')

        # data = data.T.reset_index()
        # data = pd.melt(data, id_vars=["index"]).rename(
        #     columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
        # )
        # chart = (
        #     alt.Chart(data)
        #     .mark_area(opacity=0.3)
        #     .encode(
        #         x="year:T",
        #         y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
        #         color="Region:N",
        #     )
        # )
        # st.altair_chart(chart, use_container_width=True)
except urllib.error.URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )