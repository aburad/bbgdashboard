# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 15:51:23 2020

@author: Ajit
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import matplotlib.dates as mdates
import re
import talib
import pandas_ta as ta
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
import matplotlib.dates as mdates
from scipy import stats
from bbgData import *
global spreadPlot
global plotRange
#os.chdir('P:\\Work\\Python');

def plotSwap(x,y,x_label=None,y_label=None,filename=''):
    if isinstance(x, pd.DataFrame):
        x=x.iloc[:,0]
    if isinstance(y, pd.DataFrame):
        y=y.iloc[:,0]
        
    ctitle = "Period: "+ x.index[0].strftime('%d/%m/%Y') +" to " + x.index[-1].strftime('%d/%m/%Y')
    x=x.sort_index(ascending=False);
    y=y.sort_index(ascending=False);
    if not x_label:
        x_label = x.name;
    if not y_label:
        y_label = y.name;        
    x0=x[0];
    y0=y[0];
    x1=x[1:20];
    y1=y[1:20];
    x2=x[21:120];
    y2=y[21:120];
    x3=x[121:];
    y3=y[121:];

    xchart1 = x[1:120];
    ychart1 = y[1:120];
    fig1, ax1 = plt.subplots();
    plt.scatter(x=x3, y=y3, color="LightGreen", s=10,label="T-120 and before");
    plt.ion() ; #interactive mode
    plt.scatter(x=x2, y=y2, color="DarkGreen", s=10,label="T-20 to T-120");
    plt.scatter(x=x1, y=y1, color='Blue', s=10,label="Latest 20 Obs");
    plt.scatter(x=x0, y=y0, color='Red', s=25, label="Now");
    plt.plot(x[0:20], y[0:20], color='Blue');
    plt.title(ctitle);
    plt.plot(np.unique(xchart1), np.poly1d(np.polyfit(xchart1, ychart1, 1))(np.unique(xchart1)),'--', color='Purple',label="Regression for latest 120");
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),'-', color='Black',label="Regression for all");
    plt.legend(loc=2 ,fontsize = 'x-small');
    ax1.set_xlabel(x_label, color='g');
    ax1.set_ylabel(y_label, color='b');
    fig1.show(); 
    plt.savefig(filename+'ps.png', bbox_inches='tight',dpi = 300)

    fig2, ax2 = plt.subplots();
    ax2_1 = ax2.twinx();
    ax2_1.plot(x.index,y,'g-');
    ax2.plot(x.index,x,'b-');
    ax2.set_xlabel("date");
    #ax2.set_xticklabels(ax2.xaxis.get_majorticklabels(), rotation=90);
    fig2.autofmt_xdate();
    ax2_1.set_ylabel(y_label, color='g')
    ax2.set_ylabel(x_label, color='b')      
    fig2.show();

    fig, ax = plt.subplots();
    x=pd.DataFrame(x)
    y=pd.DataFrame(y)
    data=x.join(y)
    data['year']=data.index.year.values.tolist()
    sn=sns.scatterplot(data=data, y=data.columns[1], x=data.columns[0], hue="year", palette="deep")
    
    return fig
#    sns.lmplot(x=x, y=y)

def rollingcorr(df,window=50):
    correl=[];
    num = len(df)
    for i in range(num-window):
        correl.append(df.iloc[i:i+window].corr().values);
    return correl;

def rollingstd(df,window_=50):
    dfstd=df.rolling(window=window_).std();
    dfstd=dfstd.dropna();
    return dfstd;

def plotCorr(df):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    data_corr = df.corr();
    fig, ax = plt.subplots();
    ax.autoscale(False);
    labels = [i.split(" ", 1)[0] for i in list(df)];
    #ax = sns.heatmap(data_corr,color="b")
    #ax.set_xticklabels(labels,rotation=30)
    #ax.set_yticklabels(labels,rotation=30) 
    fig, ax = plt.subplots(figsize=(9,6));
    
    im=plt.pcolor(data_corr,cmap='winter',figure=fig)
    
    for y in range(data_corr.shape[0]):
        for x in range(data_corr.shape[1]):
            plt.text(x + 0.5, y + 0.35, '%.2f' % data_corr.values[x, y],
                 horizontalalignment='center',
                 verticalalignment='center',color='w',fontsize=13); 
    plt.xticks(rotation=90);
    divider = make_axes_locatable(plt.gca());
    cax = divider.append_axes("right", "5%", pad="3%");
    plt.colorbar(im, cax=cax);
    ax.set_xticklabels(labels );
    ax.set_yticklabels(labels );
    locs = np.arange(len(labels));
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_ticks(locs + 0.75, minor=True);
        axis.set(ticks=locs + 0.25, ticklabels=labels);

    plt.tight_layout();
    plt.show();
    return data_corr;

def calcDiff(df):
    if isinstance(df, pd.Series):
        if df.name.split(" ", 1)[0]==3:
                df_change=df.pct_change();
        else:
            df_change = df.diff();
        df_change=df_change.iloc[1:];
    else:
        df_change = df.sort_index(ascending=[True]);
        cols = df.columns;
        curr_cols = [col for col in cols.tolist() if len(col.split(" ", 1)[0])==3];
        rate_cols = [col for col in cols.tolist() if len(col.split(" ", 1)[0])>3]; 
        df_change[curr_cols]=df_change[curr_cols].pct_change();
        df_change[rate_cols]=df_change[rate_cols].diff();
        df_change=df_change.iloc[1:,:];
    return df_change;

def cumReturn(df,flag=1):
    return (df+flag).cumprod();

def plotStat(ts):
    name=str(ts.name)
    plt.ion()
    fig, ax = plt.subplots()
    #Determing rolling statistics
    rolmean25 = ts.rolling(window=25).mean()
    rolmean50 = ts.rolling(window=50).mean()    ;
    rolstd25 = ts.rolling(window=25).std()
    upper = ts + rolstd25;
    lower = ts - rolstd25; 
    #ts_s = pd.concat([ts, rolmean25, rolmean50, rolstd25],axis=1) ;
    #ts_s.columns = [name, name+"mean25",name+"mean50",name+"std25"];
    #Plot rolling statistics:
    orig = plt.plot(ts.index,ts, color='blue',label='Original')
    mean1 = plt.plot(ts.index,rolmean25, color='red', label='Rolling 25d Mean')
    mean2 = plt.plot(ts.index,rolmean50,color='brown',label='Rolling 50d Mean');
    plt.plot(ts.index,upper,'--',color='g');
    plt.plot(ts.index,lower,'--',color='g');
    #ax2 = ax.twinx();
    #std = ax2.plot(ts.index, rolstd25, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    fig.autofmt_xdate();
    plt.show()
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput);
    return;

def df_write(df,file):
    df=df.to_pickle(file); 
    return file;

def df_read(file):
    df = pd.read_pickle(file);
    return df;

def plotYldCurve(df: bbgData, filename='korea/yc.png',tenor_suffix='y',mult=100):
    t1=df.dfbbg.index[-1];
    t2=df.dfbbg.index[-6];
    t3=df.dfbbg.index[-22];
    t4=df.dfbbg.index[-120];
    country=df.currency
    [tenor, yc1]=df.yldcurve(date=t1);
    yc2=df.yldcurve(date=t2)[1];
    yc3=df.yldcurve(date=t3)[1];
    yc4=df.yldcurve(date=t4)[1];
    wk_chg = (yc1-yc2)*mult;
    mth_chg = (yc1-yc3)*mult;
    ycurves=np.stack([yc1,yc2,yc3,yc4]);
    ycurves_txt=[["{:.3f}".format(y) for y in x] for x in ycurves]
    ycchange = np.stack([wk_chg,mth_chg]);
    ycchange_txt=[["{:.3f}".format(y) for y in x] for x in ycchange]
    num_lines=4;
    tenor_y = [str(i)+tenor_suffix for i in tenor];
    colours = ['red', 'pink', 'g','lightblue'] ;
    markers = ['o', 'v', '*', '+'];
    labels = ['Now','1w ago','1mth ago','6mth ago'];
 
    fig, ax = plt.subplots(figsize=(9,6));
    fig.subplots_adjust(bottom=0.35) 
    plt.ion();
    for i in range(num_lines):
        # Scatter plot with point_size^2 = 75, and with respective colors
        plt.scatter(tenor, ycurves[i], marker=markers[i], s=25, c=colours[i],label=labels[i]);
        # Connects points with lines, and with respective colours
        plt.plot(tenor, ycurves[i], c=colours[i]);
    plt.xticks(tenor,tenor_y);
    ticksize=1/(plt.xlim()[1]-plt.xlim()[0]);
    ticksizes = [ticksize for x in range(len(tenor))];
    the_table = plt.table(cellText=ycurves_txt,rowLabels=labels, colLabels=tenor_y,
                          rowColours=colours,colWidths=ticksizes,
                          cellLoc='center',loc='bottom',
                          bbox=[ticksize/2, -0.38,ticksize*10, 0.3]);
    the_table1 = plt.table(cellText=ycchange_txt,rowLabels=['1w change','1mth change'], 
                      colWidths=ticksizes, cellLoc='center',loc='bottom',
                      bbox=[ticksize/2, -0.58,ticksize*10, 0.15]);
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(13)
    the_table1.auto_set_font_size(False)

    the_table1.set_fontsize(13)
    
#   plt.grid();
    plt.legend(loc=2 ,fontsize = 'small');
    plt.title("Yield Curve Change "+ country);
    plt.show();
    plt.savefig(filename, bbox_inches='tight',dpi = 300)

    plt.ioff();

def twoAxisPlot(x,y1,y2,xlabel,y1label,y2label,title,filename='a.png',inverty1=0,inverty2=0,y1hline=[],y2hline=[]):
    fig, ax = plt.subplots(figsize=(9,6));
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    ax2 = ax.twinx();
    ax.plot(x,y1,'g-');
    ax2.plot(x,y2,'b-');
    ax.set_xlabel(xlabel);
    plt.title(title,fontsize=18);
    ax.set_ylabel(y1label, color='g',fontsize=13)
    ax2.set_ylabel(y2label, color='b',fontsize=13)  
    if inverty1 == 1:
        ax.invert_yaxis()
    if inverty2 == 1:
        ax2.invert_yaxis()    
    plt.yticks(fontsize=14)
    for y1h in y1hline:
        ax.axhline(y=y1h,color='r')    
    for y2h in y2hline:
        ax2.axhline(y=y2h,color='r')    

    fig.autofmt_xdate();
    fig.show();
    plt.savefig(filename, bbox_inches='tight',dpi = 300)

    return;


def spreadPlot(x,y1,y2,ylabel,xlabel='',y1label='',y2label='',title='',spreadlabel='',beta=1,filename='tmp.png',g2type='line',mult=100,hline_f=0,spreadtype='diff'):
    if isinstance(y1,pd.DataFrame):
        y1=y1.iloc[:,0]
    if isinstance(y2,pd.DataFrame):
        y2=y2.iloc[:,0]    
    if y1label=='':
        y1label=y1.name
    if y2label=='':
        y2label=y2.name
    if title=='':
        title="Spread of "+y2label +" vs "+y1label
    if xlabel=='':
        xlabel='Date'
    if spreadlabel=='':
        spreadlabel='Spread (bp)'
    
    fig, (ax1, ax2) = plt.subplots(2, sharex=True,figsize=(9,6))
    ax1.plot(x,y1)
    ax1.plot(x,y2)
    if spreadtype == 'diff':
        spread = (y2-beta*y1)
    elif spreadtype=='ratio':
        spread = y2/y1
    if g2type == 'line':
        ax2.plot(x,spread*mult,'b-')
    if g2type == 'bar':
        ax2.bar(x,spread*mult,color='b',width=.4)
    if hline_f == 1:    
        ax2.axhline(y=0)    
    fig.suptitle(title,size=17)
    ax1.set_ylabel(ylabel, color='g',size=13)
    ax1.legend([y1label,y2label])
    ax2.set_ylabel(spreadlabel, color='b',size=13)  
    ax1.set_xlabel(xlabel,fontsize = 15);
    ax1.xaxis.grid(True) 
    ax1.yaxis.grid(True) 
    #plt.locator_params(axis='x', nbins=14) #Show five dates

    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m:%Y"))
    ax2.xaxis.grid(True) 
    plt.xticks(rotation=45,size=13)
    #fig.autofmt_xdate()
    fig.show()
    plt.savefig(filename, bbox_inches='tight',dpi = 300)

    return fig

def spreadPlot1(x,y1,y2,y3,ylabel,xlabel='',y1label='',y2label='',y3label='',title='',spreadlabel='',beta=1,filename='tmp.png',g2type='line',spreadtype='diff',mult=100,h0line=1):
    if isinstance(y1,pd.DataFrame):
        y1=y1.iloc[:,0]
    if isinstance(y2,pd.DataFrame):
        y2=y2.iloc[:,0]    
    if isinstance(y3,pd.DataFrame):
        y3=y3.iloc[:,0]    
    
    if y1label=='':
        y1label=y1.name
    if y2label=='':
        y2label=y2.name
    if y3label=='':
        y3label=y3.name
    myFmt = mdates.DateFormatter('%d%m%Y')
    
    if title=='':
        title="Spread of "+y2label +" vs "+y1label
    if xlabel=='':
        xlabel='Date'
    if spreadlabel=='':
        spreadlabel='Spread (bp)'
    
    fig, (ax1, ax2,ax3) = plt.subplots(3, sharex=True,figsize=(9,9))
    ax3.plot(x,y3)
    ax1.plot(x,y1)
    ax1.plot(x,y2)
    if spreadtype == 'diff':
        spread = (y2-beta*y1)
    elif spreadtype=='ratio':
        spread = y2/y1
    if g2type == 'line':
        ax2.plot(x,spread*mult,'b-')
    if g2type == 'bar':
        ax2.bar(x,spread*mult,color='b')
    if h0line==1:
        ax2.axhline(y=0)    
    fig.suptitle(title,size=17)
    ax1.set_ylabel(ylabel, color='g',size=13)
    ax1.legend([y1label,y2label])
    ax2.set_ylabel(spreadlabel, color='b',size=13)  
    ax1.set_xlabel(xlabel,fontsize = 15);
    ax1.xaxis.grid(True) 
    ax1.yaxis.grid(True) 
    #plt.locator_params(axis='x', nbins=14) #Show five dates

    ax3.legend([y3label])
    ax3.set_ylabel(y3label, color='r',size=13)  
    ax3.set_xlabel(xlabel,fontsize = 15);
    ax3.xaxis.grid(True) 
    ax3.yaxis.grid(True) 


    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m:%Y"))
    ax2.xaxis.grid(True) 
    plt.xticks(rotation=90,size=13)
    ax2.xaxis.set_major_formatter(myFmt)

    #fig.autofmt_xdate()
    fig.show()
    plt.savefig(filename, bbox_inches='tight',dpi = 300)
    return fig

def spreadPlotn(x,y,xlabel='',ylabels='',title='',filename='tmp.png'):
    n = len(y)
    for i in range(n):
       if isinstance(y[i],pd.DataFrame):
        y[i]=y[i].iloc[:,0]   
       if ylabels[i]=='':
        ylabels[i]=y[i].name
    myFmt = mdates.DateFormatter('%d-%m-%Y')
    
    if xlabel=='':
        xlabel='Date'

    fig, ax = plt.subplots(n, sharex=True,figsize=(3*n,9))
    for i in range(n):
        ax[i].plot(x,y[i])
        ax[i].set_ylabel(ylabels[i], size=13)
        ax[i].legend([ylabels[i]])
        ax[i].yaxis.grid(True) 

    fig.suptitle(title,size=17)
    ax[n-1].set_xlabel(xlabel,fontsize = 15);
    ax[n-1].xaxis.grid(True) 
    ax[n-1].xaxis.set_major_formatter(myFmt)
    plt.xticks(rotation=45,size=13)
    fig.show()
    plt.savefig(filename, bbox_inches='tight',dpi = 300)
    return fig

def calcTrends(data,window=100,margin=10,title=" "):
    ts_ = data;
    ts=ts_.tolist();
    rows=len(ts);
    if window>rows:
        window=rows;
    ts=ts[-window:];
    fig,ax=plt.subplots(figsize=(8,6));
    plt.plot(ts_.index,ts_,zorder=10);     
    max0=np.argmax(ts);
    min0=np.argmin(ts);
                   
    min0_l=np.argmin(ts[0:min0-margin]);
    support_l_m,support_l_c = np.polyfit([min0_l, min0], [ts[min0_l],ts[min0]],1);
    support_l=[support_l_m*x+support_l_c for x in range(window)];
    plt.plot(ts_.index[rows-window:rows],support_l,linestyle='--',c='r');
    
    if min0+margin<window:
        min0_r=min0+margin+np.argmin(ts[min0+margin:window]);
        support_r_m,support_r_c = np.polyfit([min0, min0_r], [ts[min0],ts[min0_r]], 1);
        support_r=[support_r_m*x+support_r_c for x in range(window)];
        plt.plot(ts_.index[rows-window:rows],support_r,linestyle='--',c='r');
            
    max0_l=np.argmax(ts[0:max0-margin]);
    resist_l_m,resist_l_c = np.polyfit([max0_l,max0], [ts[max0_l],ts[max0]], 1);
    resist_l=[resist_l_m*x+resist_l_c for x in range(window)];
    plt.plot(ts_.index[rows-window:rows],resist_l,linestyle='--',c='g');
    
    if max0+margin<window:
        max0_r=max0+margin+np.argmax(ts[max0+margin:window]);
        resist_r_m,resist_r_c = np.polyfit([max0,max0_r], [ts[max0],ts[max0_r]], 1);
        resist_r=[resist_r_m*x+resist_r_c for x in range(window)];
        plt.plot(ts_.index[rows-window:rows],resist_r,linestyle='--',c='g');
    fig.autofmt_xdate();    
    plt.title('Technicals '+title,fontsize=16)
    plt.show();
    plt.savefig('korea/technicals.png', bbox_inches='tight',dpi = 300)

    return;

def bbands(df, win=30,numsd=2,title=''):
    ave = df.rolling(center=False,window=win).mean();
    sd = df.rolling(center=False,window=win).std();
    upband = ave + (sd*numsd);
    dnband = ave - (sd*numsd);
    fig,ax=plt.subplots(figsize=(8,5));
    plt.plot(df.index,df,c='b');
    plt.plot(df.index,np.round(upband,3),c='r',linestyle='--');
    plt.plot(df.index,np.round(dnband,3),c='r',linestyle='--');
    plt.plot(df.index,np.round(ave,3),c='g');
    ans = pd.concat([df, np.round(ave,3), np.round(upband,3), np.round(dnband,3)], axis=1) ;
    
    fig.autofmt_xdate();    
    plt.title('Bollinger Bands '+title,fontsize=16)
    plt.show();  

    plt.savefig('korea/bband.png', bbox_inches='tight',dpi = 300)

    return ans;

def plotmacd(df, count=200,title='',fname=''):
    exp1 = df.iloc[:,0].ewm(span=12, adjust=False).mean()
    exp2 = df.iloc[:,0].ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
    exp3 = macd.ewm(span=9, adjust=False).mean()
    plotS.spreadPlot1(exp3.tail(count).index,exp3.tail(count),macd.tail(count),df.tail(count),xlabel='Date',ylabel='MACD & Singal',y2label='MACD ',y1label='Signal',y3label=title, title=' MACD vs Signal: '+title  , filename=fname,g2type='line')
    return (macd, exp3)

def plotrsi(df, count=200,title='',fname=''):
    return pd.DataFrame(data=talib.RSI(df.iloc[:,0].to_numpy()),index=df.index, columns=['rsi'])

def genmacdsignal(macd, signal):
    return
    
def cleandf_curr(df,flip=1,freq='w',diff=1):
    if flip==1:
        inv_curr=['AUD','EUR','GBP'];
        for c in inv_curr:
            df[c+' Curncy']=1/df[c+' Curncy'];
    if freq=='w':
        df=df[df.index.weekday==4];
    if diff==1:
        df=calcDiff(df);
    return df;

def seriestest(ts):
    lag_correlations = acf(ts.tail(len(ts-1)));
    lag_partial_correlations = pacf(ts.tail(len(ts-1)));
    plt.subplot(2, 1, 1)                               
    plt.plot(lag_correlations, marker='o', linestyle='--');
    plt.title("ACF");
    plt.subplot(2, 1, 2) ;
    plt.plot(lag_partial_correlations, marker='+', linestyle='-');
    plt.title("PACF");
    decomposition = seasonal_decompose(ts);
    fig = decomposition.plot();  
    plt.show();
            
def seasonality(df,freq='M'):
    df_r=df.resample(freq).last();
    df_r=calcDiff(df_r);
    name=df.columns[0]
    df_r['Year']=df_r.index.year;
    df_r['Month']=df_r.index.month
    tmp=df_r[[name, 'Year', 'Month']]
    aa=tmp.pivot(index='Month',columns='Year') 
    return aa;

def yc_svm(preds,y,margin=0.05):
    n=y.shape[0];
    n_sd = int(n*0.75);
    #n_vd = int((n-n_sd)/2);
    #n_td = n-n_sd-n_vd;
    yy=y.copy();
    yy[yy>0.1]=1;
    yy[yy<-0.1]=-1;
    yy[(yy>-0.1) & (yy<0.1) ]=0  
    X=preds[0:n_sd,:];
    ret=yy[0:n_sd];
    clf = svm.SVC(kernel='linear', C = 1.0);
    clf.fit(X,ret);
    print("Sample score: {:.2f}" ,clf.score(X,ret));
    print("Validation score: {:.2f}", clf.score(preds[n_sd+1:],yy[n_sd+1:]));  


def uscnplot(ccy1='USD',ccy2='CNY',tenor='2',yr='2008',dfbbg=''):
    currlist=['GT'+ccy1+tenor+'Y GOVT', 'GT'+ccy2+tenor+'Y CORP',ccy2+' CURNCY']
    # if not isinstance(dfbbg,pd.DataFrame):
    #     datas=load_BBGdata(currlist,start='20080101',end='20200804',freq='DAILY',maxDataPoints=5000)
    #     dfbbg,secs=processBBGData(datas)
    df_cnh=dfbbg[currlist]
    df_cnh=df_cnh.rename(columns={ccy2+' Curncy':ccy1+ccy2})
    df_cnh[ccy1+'_'+ccy2]=(df_cnh[currlist[1]]-df_cnh[currlist[0]])*-100
    df_cnh=df_cnh[df_cnh.index>yr]
    twoAxisPlot(df_cnh.index,df_cnh[ccy1+'_'+ccy2],df_cnh[ccy2+" CURNCY"],"Date",ccy1[0:2]+' ' +ccy2[0:2] +' ' +tenor+"y Govt yield spread",ccy1+ccy2,ccy1[0:2]+' ' +ccy2[0:2]+'  '+tenor+'y Govt yield spread and '+ccy1+ccy2)
    
    
def plotBarBox(df,count=30,spread_name='',charts=[1,1,1],labelcount=6):
    df=df.tail(count)
    #%matplotlib qt
    if spread_name=='':
        spread_name=df.columns[0][3:]
    last=df.tail(1).values[0]
    cols=df.columns.values
    
    Z = [x for _,x in sorted(zip(last,cols))]
    df = df[Z]        
    cols=df.columns.values
    last=df.tail(1)

    xlabel=[c[0:labelcount] for c in Z]
    # make barplot and sort bars in descending order

    if charts[0]==1:
        fig, ax = plt.subplots();
        sn=sns.barplot(x=cols, y=last.values[0],zorder=0)
        fig.canvas.draw()
        #yticks1=sn.get_yticklabels()
        #labels = [item.get_text() for item in yticks1]
        sns.boxplot(x="variable", y="value", data=pd.melt(df),width=.3)
        fig.canvas.draw()
        sn.set_xticklabels(xlabel, rotation=45,size=13)
        sn.set_xlabel("Currencies",fontsize=20)
        sn.set_ylabel(spread_name+" ",fontsize=15)
        #sn.set_yticklabels(sn.get_yticklabels(), size=15)
        fig.suptitle(spread_name+" across currencies with past "+str(count)+" days boxplot", fontsize=20)
        plt.show()
    if charts[1]==1:    
        fig, ax = plt.subplots();
        sn1=sns.scatterplot(x=last.columns.values,y=last.values[0],s=100,marker="s",color='.2',legend='full')
        fig.canvas.draw()
        sn1=sns.swarmplot(x="variable", y="value", data=pd.melt(df),zorder=0)
        ax.legend(["Latest Value"],loc="lower left")
        fig.canvas.draw()
        sn1.set_xticklabels(xlabel, rotation=45,size=13)
        sn1.set_xlabel("Currencies",fontsize=15)
        sn1.set_ylabel(spread_name+" swap spread",fontsize=15)
        #sn1.set_yticklabels(sn1.get_yticklabels(), size=15)
        fig.suptitle(spread_name+" swap spread across currencies with past "+str(count)+" days boxplot", fontsize=20)
        plt.show()

    if charts[2]==1:    
        fig, ax = plt.subplots();
        sn2=sns.scatterplot(x=last.columns.values,y=last.values[0],s=200,marker="s",color='.2',zorder=10,legend='full')
        ax.legend(["Latest Value"],loc="lower left")
        fig.canvas.draw()
        sn2=sns.violinplot(x="variable", y="value", data=pd.melt(df),zorder=0,palette=sns.color_palette("husl", 8))
        fig.canvas.draw()
        sn2.set_xticklabels(xlabel, rotation=45,size=13)
        sn2.set_xlabel("Currencies",fontsize=15)
        sn2.set_ylabel(spread_name+" swap spread",fontsize=15)
        sn2.set_yticklabels(sn2.get_yticklabels(), size=15)
        fig.suptitle(spread_name+" swap spread across currencies with past "+str(count)+" days boxplot", fontsize=20)
        plt.show()
    return fig

def plotRange(df,count=30,title='',filename='a.png',type_='',labelcount=3,sort='y'):
    if sort=='y':
        df= df.tail(count)
        last=df.tail(1).values[0]
        cols=df.columns.values
        Z = [x for _,x in sorted(zip(last,cols))]
        df = df[Z]        
    cols=df.columns.values
    last=df.tail(1)
    
    min_=df.min()
    max_=df.max()
    xlabel=[c[0:labelcount] for c in df.columns.values.tolist()]
    
    if type_=='Swap':
        xlabel=[re.findall('\d+', c)[0] +'y' for c in df.columns.values.tolist()]
    last = df.iloc[-1,:]
    mthago = df.iloc[-25,:]
    wkago = df.iloc[-7,:]
    
    fig, ax = plt.subplots(figsize=(9,6))
    sns.scatterplot(x=max_.index.values,y=max_.values,s=200,marker="_",color='green')
    sns.scatterplot(x=min_.index.values,y=min_.values,s=200,marker="_",color='green')
    sns.scatterplot(x=last.index.values,y=last.values,s=100,marker="o",color='blue',label='Latest Value')
    sns.scatterplot(x=mthago.index.values,y=mthago.values,s=100,marker="X",color='red',label='Value month ago')
    sns.scatterplot(x=wkago.index.values,y=wkago.values,s=100,marker="p",color='brown',label='Value week ago')
    
    plt.vlines(x=last.index.values, ymin=min_, ymax=max_, color='grey', alpha=1)
    plt.yticks(fontsize=16)
    ax.set_xticklabels(xlabel ,rotation=45,size=14)
    plt.title(str(count)+" period Range plot "+title,fontsize=20)
    plt.legend(fontsize='large')
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight',dpi = 300)

    return df

def regress(x,y,alpha=.05,xlabel='',ylabel='',title=''):
    if x.name=='':
        x.name='x'
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()

    st, data, ss2 = summary_table(res, alpha=0.05)
    fittedvalues = data[:,2]
    #predict_mean_se  = data[:,3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
    predict_ci_low, predict_ci_upp = data[:,6:8].T
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(x, y, label="data",s=5)
    ax.plot(x, fittedvalues, 'r-', label='OLS: 95% confidence')
    ax.plot(x, predict_ci_low, 'b--')
    ax.plot(x, predict_ci_upp, 'b--')
    ax.plot(x, predict_mean_ci_low, 'g--')
    ax.plot(x, predict_mean_ci_upp, 'g--')
    ax.scatter(x=x.tail(1), y=y.tail(1), color='Red', s=25, label="Now")
    ax.legend(loc='best');
    equation = ylabel+" = %.4f"%res.params[1] +" * " + xlabel +"  + " + "%.4f"%res.params[0]
    ax.set_xlabel(xlabel + "          "+ equation, color='g',fontsize = 15);
    ax.set_ylabel(ylabel, color='b',fontsize = 15);
    plt.show()
    
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(x.index,data[:,8])
    plt.title("Residual plot:  "+equation,fontsize=20)
    ax.set_xlabel("Date ", color='g',fontsize = 15);
    ax.set_ylabel("Residual ", color='g',fontsize = 15);
    plt.show()
    return data

def mvg(df,windows=[5,10,50,100,150,200]):
    for n in windows:
        df['mvg'+str(n)]   = df.iloc[:,0].rolling(window=n).mean()
    return df
    
def plot2Curves(x,y,xlab='',ylab='',title='',count=1000,filename=''):
    x = x.copy()
    y = y.copy()
    if type(x)==pd.DataFrame:
        x = x.iloc[:,0]
    if type(y)==pd.DataFrame:
        y = y.iloc[:,0]
    if x.name == '':
        x.name = xlab
    if y.name == '':
        y.name = ylab
    x=x.tail(count)
    y=y.tail(count)
    if xlab == '':
        xlab=x.name
    if ylab=='':
        ylab=y.name
        
    regress(x,y,xlabel=xlab,ylabel=ylab,title=title)
    plotSwap(x,y,x_label=xlab,y_label=ylab,filename=filename)
    spreadPlot(x.index,x,y,ylabel='Rate',xlabel='Date',y1label=xlab,y2label=ylab,title=title,spreadlabel='Spread ')
    sns.jointplot(x=x,y=y, kind="reg")
    
def ts(trade,start='20150101',plot=1,mult=100):
    if type(trade)==list:
        if len(trade)==2:
            e1 = bbgData()
            e2 = bbgData()
            (curr1, fwd1, tenors1, ts1)=e1.getTradeTs(trade[0],start=start)
            (curr2, fwd2, tenors2, ts2)=e2.getTradeTs(trade[1],start=start)
            if plot==1:
                spreadPlot(ts1.index,ts1,ts2,ylabel='Rate',xlabel='Date',y1label=ts1.columns[0],y2label=ts2.columns[0],title=ts1.columns[0] +' vs '+ ts2.columns[0],spreadlabel='Spread ')
            #ts1.plot()
            #ts2.plot()
            tt= (ts2.iloc[:,0]-ts1.iloc[:,0])*mult
            tt = tt.to_frame()
            tt.columns = [trade[0]+"_"+trade[1]]
            return tt
        if len(trade)==1:
            return ts(trade[0],start=start,plot=plot)
    else:        
        e = bbgData()
        (curr, fwd, tenors, ts0)=e.getTradeTs(trade,start=start)
        if plot==1:
            ts0.plot()
        return ts0

def ts1(trade,start='20150101',plot=1):
    e = bbgData()
    ticker = e.getTicker(trade)
    f_ticker = flatten(ticker)
    f_ticker=list(dict.fromkeys(f_ticker))
    e = bbgData()
    e.freshTS(tickers=f_ticker,start=start,fill='ffill',verbose=0)
    df= pd.DataFrame()
    df=e.getTickerTs(ticker).to_frame()
    df.columns = [trade]
    if plot==1:
        df.plot()
    return df

def plotChange(df,diff=0,n=[1,5,22],mult=100,title_prefix=' ',filename=''):
    df=df.loc[:,df.columns[list(np.invert(df.tail(1).isnull().values[0]))]]

    if diff == 0:
        mtd = df.diff(n[2]).tail(1).transpose()*mult
        wtd = df.diff(n[1]).tail(1).transpose()*mult
        d = df.diff(n[0]).tail(1).transpose()*mult
        change_type = 'Difference: '
    if diff == 1:
        mtd = df.pct_change(n[2]).tail(1).transpose()*mult
        wtd = df.pct_change(n[1]).tail(1).transpose()*mult
        d = df.pct_change(n[0]).tail(1).transpose()*mult
        change_type = 'Pct change: '

    if diff == -1:
        mtd = df.iloc[-n[2],:].to_frame()
        wtd = df.iloc[-n[1],:].to_frame()
        d = df.iloc[-n[0],:].to_frame()
        change_type = "Historical values: "

    mtd.columns = ['MTD']
    wtd.columns = ['WTD']
    d.columns = ['Daily']
    df_chg =mtd.join(wtd).join(d)
    df_chg.index.name = 'Currency'
    ax=df_chg.sort_values(by='MTD').plot.bar()
    ax.set_xlabel("Currency", size=18)
    ax.set_ylabel("Change in bps", size=18)
    ax.tick_params(axis = 'both',  labelsize = 12)
    ax.axes.set_title(title_prefix+change_type +str(n),size=20)
    #ax.xticks(rotation=45,fontsize=12)
    ax.figure.savefig(filename)
    return df_chg

def vstackedPlot(df,title='',hlines=[],filename=''):
    cols = df.columns.to_list()
    num= len(cols)
    fig, ax = plt.subplots(num, sharex=True,figsize=(9,num*2))
    for i in range(num):
        ax[i].plot(df.index,df.iloc[:,i],label=cols[i]+' %.2f' % df.iloc[-1,i])
        for hh in hlines:
            ax[i].axhline(y=hh,color='r')    
        ax[i].legend(loc='best')
        ax[i].set_ylabel(cols[i], size=14)
    
    fig.suptitle(title,size=16)
    ax[i].set_xlabel("Date", size=16)
    plt.savefig(filename, bbox_inches='tight',dpi = 300)
    plt.show()


def plotVolAdjusted(df,period=60,count=1000,title='Vol Adjusted yield '):
    swstd = df/(df.rolling(window=period).std()*np.sqrt(252))
    plotRange(swstd,count=count,title=title)
    plotBarBox(swstd,count=count,spread_name=title,charts=[1,0,0],labelcount=3)


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def dump(b:bbgData,file):
    with open(file, 'wb') as output:
        pickle.dump(b, output, pickle.HIGHEST_PROTOCOL)

