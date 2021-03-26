# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 23:41:00 2020

@author: Ajit
"""
#import blpapi
from optparse import OptionParser
import pandas as pd
import os
from datetime import date, datetime
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pickle
import plotS 
import matplotlib.ticker as ticker 
import re
from statsmodels.regression.rolling import RollingOLS
import copy
from scipy import stats

class bbgData:

    def __init__(self,tickers=[],sdate='20150101',filepath="/data/BBGticker.xls",flds=['PX_LAST'],fill='ffill'):
        self.bbgticker = pd.DataFrame()
        self.fxmult = pd.DataFrame()
        self.dfbbg=pd.DataFrame()
        self.secs=[];
        self.maturity_set = {
            1: [1,2,3,4,5,7,10],
            2: [1,2,3,4,5,7,10,12],
            3: [1,2,3,4,5,7,10,12,15,20],
            4: [1,2,3,4,5,7,10,12,15,20,25,30],
            5: ['1Y','2Y','3Y','4Y','5Y','7Y', '10Y'],
            6: [1,2,3,6,9,12],
            7: ['F',1,2,3,4,5,7,10,12],
            8: [1,2,3,4,5,7,10,12,15],
            9: [1,2,3,4,5,6,7,8,9,10,12],
            10: [2,3,5,7,10,20,30]

        }
        if (isinstance(tickers,list)) & (len(tickers) != 0):
            self.secs=tickers
            self.populate(start=sdate,flds=flds,fill=fill)
        wk_dir = os.path.realpath('');
        self.tickerfile = wk_dir + filepath;
        
    def dump(self,file):
        with open(file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def read(self,file):
        with open(file, 'rb') as inputfile:
            b = bbgData()
            b = pickle.load(inputfile)
            return b

    def readdfbbg(self,file):
        with open(file, 'rb') as inputfile:
            self.dfbbg = pickle.load(inputfile)



    def getbbg(self,type_='',currency='',tenor=[],fill='0'):
        if currency=='':
            ticker=self.bbgticker[self.bbgticker.Type==type_].bbgticker.values.tolist()
        elif type_=='':
            ticker=self.bbgticker[self.bbgticker.Code==currency].bbgticker.values.tolist()
        else:    
            ticker=self.bbgticker[self.bbgticker.index==currency+type_].bbgticker[0]
            #if len(tenor)>0:
            #    ticker=self.bbgticker.bbgticker[[[s[4:] for s in self.bbgticker.Type.values].index(t) for t in tenor]].values.tolist()
                            
        return self.getts(ticker,fill)
    
    def gettsforCCYlist(self,ccylist=['USD','EUR','JPY','SGD','KRW','HKD','AUD','CNY', 'TWD'],type_='',fill='ffill'):
        ind = [c+type_ for c in ccylist]
        ccy = self.bbgticker[self.bbgticker.index.isin(ind)].Code.to_list()
        ts = self.getts(self.bbgticker[self.bbgticker.index.isin(ind)].bbgticker.to_list(),fill=fill)
        ts.columns=ccy
        return ts
    
    def genIMMPrefix(self,start='M1',end='M5'):
        immprefix = [imm+str(y) for y  in list(range(1,6)) for imm in ['H','M','U','Z']]
        return immprefix[immprefix.index(start):immprefix.index(end)+1]
                    
    def getsubset(self,type_='',currency='',tenor=[],fill='ffill'):
        c = bbgData()
        if currency=='':
            c.bbgticker=self.bbgticker[self.bbgticker.Type==type_]
        elif type_=='':
            c.bbgticker=self.bbgticker[self.bbgticker.Code==currency]
        else:    
            c.bbgticker=self.bbgticker[self.bbgticker.index==currency+type_]
        c.dfbbg = self.getbbg(type_,currency,tenor,fill=fill)
        c.secs = c.dfbbg.columns.values.tolist()
        return c
        
    def fill_na(self,df,fill='0'):
        if fill == '0':
            return df.fillna(0)
        if fill == 'ffill':
            return df.fillna(method='ffill')
        if fill ==   'bfill':
            return df.fillna(method='bfill')

    def getts(self,ticker,fill='0'):
        if isinstance(ticker,list):
            ticker = [x.upper() for x in ticker]
            return self.fill_na(self.dfbbg[ticker].copy(),fill)
        else:
            ticker = ticker.upper()
            return self.fill_na(self.dfbbg[[ticker]].copy(),fill);

    def getSpread(self,type_='Swap',curr='',tenor=['1y', '5y', '10y'],w=[]):
        if curr=='':
            curr=self.currency
        spread=pd.DataFrame()
        if len(tenor)==1:
            if len(w)==0:
                w=1
                spread[curr+tenor[0]]=(self.dfbbg[self.bbgticker['bbgticker'][curr+type_+tenor[0]]])*w
        if len(tenor)==2:
            if len(w)==0:
                w=[-1,1]
            spread[curr+tenor[0]+tenor[1]]=(self.dfbbg[self.bbgticker['bbgticker'][curr+type_+tenor[1]]]*w[1]+self.dfbbg[self.bbgticker['bbgticker'][curr+type_+tenor[0]]]*w[0])*100
        if len(tenor)==3:
            if len(w)==0:
                w=[-1,2,-1]
            spread[curr+tenor[0]+tenor[1]+tenor[2]]=(w[1]*self.dfbbg[self.bbgticker['bbgticker'][curr+type_+tenor[1]]]+w[0]*self.dfbbg[self.bbgticker['bbgticker'][curr+type_+tenor[0]]] +w[2]*self.dfbbg[self.bbgticker['bbgticker'][curr+type_+tenor[2]]])*100
        return self.fill_na(spread,'ffill')

    def getCode(self,ticker):
        if isinstance(ticker,list):
            return [self.bbgticker[self.bbgticker.bbgticker==t].Code[0] for t in ticker]
        return self.bbgticker[self.bbgticker.bbgticker==ticker].Code[0]
    
    
    def yldcurve(self,curr='',date=''):
        if curr=='':
            curr=self.currency
        cols = self.dfbbg.columns; 
        cols_v=self.bbgticker[self.bbgticker.Code==curr].bbgticker.values
        tenor_v=self.bbgticker[self.bbgticker.Code==curr].Maturity.values
        col_tenor=list(zip(cols_v,tenor_v))
        cols_f=[c for (c,t) in col_tenor if c in cols]
        tenor_f= [t for (c,t) in col_tenor if c in cols]
        if date=='':
            yc= (self.dfbbg[cols_f].tail(1).T).unstack().values;
        else:        
            yc = self.dfbbg[cols_f].loc[date].values;
        return [tenor_f, yc];    

        
    def genSwaptickers(self,curr,prefix='',suffix='',type_='Swap',tenor_set=1,fwdflag='0',start=0,method='int'):
        if self.fxmult.empty:
            self.bbgticker,self.fxmult=self.readTickers1()
        if method=='':    
            method = self.fxmult[self.fxmult.index==curr]['method'][0]
        if (start != 0) and (fwdflag=='0'):
            fwdflag = self.fxmult[self.fxmult.index==curr]['fwdflag'][0]

        if prefix == '':
            if start ==0:
                prefix = self.fxmult[self.fxmult.index==curr]['spotprefix'][0]
            else:
                prefix = self.fxmult[self.fxmult.index==curr]['fwdprefix'][0]
        if suffix == '':
            if start ==0:
                suffix = self.fxmult[self.fxmult.index==curr]['spotsuffix'][0]
            else:
                suffix = self.fxmult[self.fxmult.index==curr]['fwdsuffix'][0]

        if type(tenor_set)==list:
            Maturity = tenor_set
        else:
            Maturity=self.maturity_set.get(tenor_set)
        tenors = [str(c) for c in Maturity]
        if method == 'int':
            if fwdflag == 1:
                tenors = [str(c) if len(str(c)) == 2 else '0' + str(c) for c in Maturity]
            if start != 0:
                prefix = prefix+str(start)
        if method == 'str':
              tenors = [str(c)+'Y' for c in Maturity]
              if start == '0C':
                  prefix = prefix + '3M'
              else:
                  if start != 0:
                      prefix = prefix + str(int(start))+'Y'
                  elif start==0:
                      tenors = [str(c) for c in Maturity]

        cols = [prefix+tenor+suffix for tenor in tenors]
        bbgt = pd.DataFrame()
        bbgt['bbgticker']=cols
        bbgt['Code']=curr
        tenors = [str(c) for c in Maturity]
        
        Type= [type_+tenor+'y' for tenor in tenors]
        bbgt['Type']=Type
        bbgt['newindex']=bbgt['Code']+bbgt['Type']
        bbgt.index=bbgt.newindex
        bbgt['Maturity']=Maturity
        bbgt=bbgt.drop(columns=['newindex'])
        self.bbgticker=bbgt
        self.prefix=prefix
        self.suffix=suffix
        self.currency = curr
        self.tenor_set = tenor_set
        self.start = start
        
        return bbgt
    
    def genFXtickers(self,curr,prefix,suffix=' CURNCY',tenor_set=6,tenor_suffix='M'):
        Maturity=self.maturity_set.get(tenor_set)
        tenors = [str(t)+tenor_suffix for t in Maturity]
        cols = [prefix+tenor+suffix for tenor in tenors]
        bbgt = pd.DataFrame()
        bbgt['bbgticker']=cols
        bbgt['Code']=curr
        Type= ['FX'+tenor for tenor in tenors]
        bbgt['Type']=Type
        bbgt['newindex']=bbgt['Code']+bbgt['Type']
        bbgt.index=bbgt.newindex
        bbgt['Maturity']=Maturity
        bbgt=bbgt.drop(columns=['newindex'])
        self.bbgticker=bbgt
        self.prefix=prefix
        self.suffix=suffix
        self.currency = curr
        return bbgt

    def genCustomTickers(self,curr,secs,type_):
        bbgt = pd.DataFrame()
        bbgt['bbgticker']=secs
        bbgt['Code']=curr
        bbgt['Type']=type_
        bbgt['newindex']=bbgt['Code']+bbgt['Type']
        bbgt.index=bbgt.newindex
        bbgt=bbgt.drop(columns=['newindex'])
        self.bbgticker=bbgt
        self.currency = curr
        return bbgt

    def genFXSpreads(self,curr,type_='FX1M'):
        tenor_type = self.bbgticker.Type.values.tolist()
        fxspread = pd.DataFrame()
        type_fx = self.getbbg(type_=type_,currency=curr)
        fxspread[type_]=type_fx.iloc[:,0]
        for c in tenor_type:
            if c != type_:
                fxspread[type_+c[2:]]= self.getbbg(type_=c,currency=curr).iloc[:,0]-type_fx.iloc[:,0]
        return fxspread

    def genFXSpreads1(self,curr,type_='FX1M'):
        tenor_type = self.bbgticker.Type.values.tolist()
        fxspread = pd.DataFrame()
        type_fx = self.getbbg(type_=type_,currency=curr)
        fxspread[type_]=type_fx.iloc[:,0]
        c1=type_
        for c in tenor_type:
            if c != type_:
                fxspread[c1+c[2:]]= self.getbbg(type_=c,currency=curr).iloc[:,0]-self.getbbg(type_=c1,currency=curr).iloc[:,0]
            c1 = c    
        return fxspread

    
    def genSwaptiontickers(self,curr='KRW',prefix='KWSN',suffix=' CURNCY',broker='GIRO',type_='Swaption'):
        
        maturity=['1','2','3','5','7','10']
        expiry=['0C','0F','01','02','03','05','07','10']
        grid=[[prefix+i+j+' '+broker+suffix for i in expiry] for j in maturity]
        cols= [prefix+i+j+' '+broker+suffix for i in expiry for j in maturity]
        
        bbgt = pd.DataFrame()
        bbgt['bbgticker']=cols
        bbgt['Code']=curr
        Type= [type_+i+j for i in expiry for j in maturity]
        bbgt['Type']=Type
        bbgt['newindex']=bbgt['Code']+bbgt['Type']
        bbgt.index=bbgt.newindex
        bbgt=bbgt.drop(columns=['newindex'])
        self.bbgticker=bbgt
        self.prefix=prefix
        self.suffix=suffix
        self.broker=broker
        self.currency = curr
        self.grid=grid
        return bbgt
    
    
    
    def genTickers(self,type_):
        self.bbgticker,self.fxmult=self.readTickers1()
        self.bbgticker=self.bbgticker[self.bbgticker.Type==type_]

    def readTickers1(self):
        xl = pd.ExcelFile(self.tickerfile);
        #xl = pd.read_excel(self.tickerfile, engine='openpyxl')
        bbgticker = xl.parse(xl.sheet_names[0],convert_float=False,header=None);
        bbgticker.columns = bbgticker.iloc[0]  ;  
        bbgticker=bbgticker.drop(bbgticker.index[[0]]);
        bbgticker = bbgticker.set_index('Code');
        bbgticker=bbgticker.stack(dropna=True)
        bbgticker=bbgticker.to_frame()
        bbgticker.index = bbgticker.index.set_names(['Code', 'Type'])
        bbgticker=bbgticker.reset_index()
        bbgticker['newindex']=bbgticker['Code']+bbgticker['Type']
        bbgticker.index=bbgticker.newindex
        bbgticker=bbgticker.drop(columns=['newindex'])
        bbgticker=bbgticker.rename(columns={0:'bbgticker'})
        bbgticker.bbgticker=bbgticker['bbgticker'].str.upper() 
        
        fxmult = xl.parse('Mult',convert_float=False,header=None);
        fxmult.columns = fxmult.iloc[0]  ;
        fxmult=fxmult.drop(fxmult.index[[0]]);
        fxmult = fxmult.set_index('Code');
        
        return bbgticker,fxmult;
    
    def populate(self,curr='',flds=['PX_LAST'],start='20150101',end='',freq='DAILY',maxDataPoints=5000,fill='ffill',verbose=0):
        self.dfbbg=self.readdfbbg('b.bbgData')
        self.bbgticker,self.fxmult=self.readTickers1()
        self.secs = self.dfbbg.columns.to_list()

    def dropCurr(self,curr):
        if isinstance(curr,list):
            for c in curr:
                tickersd=self.bbgticker.bbgticker[self.bbgticker.Code==c].values.tolist()
                self.dfbbg=self.dfbbg.drop(columns=tickersd)
                self.bbgticker=self.bbgticker[self.bbgticker.Code!=c]
        else:
            tickersd=self.bbgticker.bbgticker[self.bbgticker.Code==curr].values.tolist()
            self.dfbbg=self.dfbbg.drop(columns=tickersd) 
            self.bbgticker=self.bbgticker[self.bbgticker.Code!=curr]
        return tickersd    
    
    def realyld(self,bond='Govt10y',cpi='CPIYOY',filename='korea/realyld.png'):
        retdata=pd.DataFrame();
        real_yld=pd.DataFrame();
        tickers=self.bbgticker[ ( self.bbgticker['Type']==bond) | ( self.bbgticker['Type']==cpi)]
        ticker_curr=tickers.pivot(index='Code',columns='Type')
        ticker_curr.columns = [ticker_curr.columns[0][1],ticker_curr.columns[1][1]]
        ticker_curr=ticker_curr[(ticker_curr[bond].isnull()==False)]
        cur=set(ticker_curr.index[(ticker_curr.iloc[:,0]!='') & (ticker_curr.iloc[:,0]!='')].values.tolist())
        for curr in cur:
            real_yld[curr]=self.fill_na(self.dfbbg[self.bbgticker.loc[curr+bond].bbgticker],fill='ffill') - self.fill_na(self.dfbbg[self.bbgticker.loc[curr+cpi].bbgticker],fill='ffill')
        real_yld_z=self.zscore1(real_yld.tail(780))
        self.real_yld=real_yld.tail(780)
        for curr in cur:
            retdata.loc[curr,bond]=self.fill_na(self.dfbbg[ticker_curr.loc[curr,bond]],fill='ffill').tail(1)[0]
            retdata.loc[curr,cpi]=self.fill_na(self.dfbbg[ticker_curr.loc[curr,cpi]],fill='ffill').tail(1)[0]
            retdata.loc[curr,'RealYield']=real_yld[curr].tail(1)[0]
            retdata.loc[curr,'RealYield ZScore']=real_yld_z[curr].tail(1)[0]
        fig, ax = plt.subplots(figsize=(8,8))         # Sample figsize in inches
        ax = plt.axes()
        g=sns.heatmap(retdata.sort_values(by=['RealYield']), annot=True,cmap="YlGnBu",annot_kws={"size": 15},yticklabels=True)
        ax.set_title("Real Yield = "+ bond +" - " + cpi)
        g.axes.set_xticklabels(g.axes.get_xmajorticklabels(), fontsize = 14)
        g.axes.set_yticklabels(g.axes.get_ymajorticklabels(), fontsize = 10)
        plt.show()
        plt.savefig(filename, bbox_inches='tight',dpi = 300)

        return (retdata.sort_values(by=['RealYield']),real_yld)

    
    def policy(self):
        tickers=self.bbgticker[(self.bbgticker['Type']=='Float') | ( self.bbgticker['Type']=='Policy') | ( self.bbgticker['Type']=='Swap1y')]
        tmpdata=self.fill_na(self.dfbbg[tickers['bbgticker']],fill='ffill').tail(1)
        rettickers=tickers.pivot(index='Code',columns='Type')
        rettickers.columns=[c[1] for c in rettickers.columns.values]
        retdata=pd.DataFrame()
        for curr in rettickers.index.values:
            for col in ['Policy', 'Float', 'Swap1y']:
                if not(pd.isnull(rettickers.loc[curr,col])):
                    retdata.loc[curr,col]=tmpdata[rettickers.loc[curr,col]][0]
        retdata['Next1Y (bp)']=(retdata['Swap1y'] - retdata['Float'])*100            
        fig, ax = plt.subplots(figsize=(8,8))         # Sample figsize in inches
        g=sns.heatmap(retdata, annot=True,cmap="YlGnBu",annot_kws={"size": 15},yticklabels=True)
        g.axes.set_xticklabels(g.axes.get_xmajorticklabels(), fontsize = 15)
        g.axes.set_yticklabels(g.axes.get_ymajorticklabels(), fontsize = 13)
        ax.set_title('Central Banks Policies ')
        
        plt.show()
        plt.savefig('korea/policy.png', bbox_inches='tight',dpi = 300)

        return retdata
    
    def getSpreadforAll(self,type_='Govt',tenor=['2y','10y'],w=[],ccylist=[]):
        cur=[]
        spread=pd.DataFrame();
        #govt2s10s.index=dfbbg.index
        if len(ccylist)==0:
            ccylist = list(set(self.bbgticker[self.bbgticker.Type.str[0:4]==type_].Code))
        for curr in ccylist:
            if (curr+type_+tenor[0] in self.bbgticker.index)&(curr+type_+tenor[1] in self.bbgticker.index):
                cur.append(curr+tenor[0]+tenor[1])
                if len(tenor)==2:
                    if len(w)==0:
                        w=[-1,1]
                    spread[curr+tenor[0]+tenor[1]]=(self.dfbbg[self.bbgticker['bbgticker'][curr+type_+tenor[1]]]*w[1]+self.dfbbg[self.bbgticker['bbgticker'][curr+type_+tenor[0]]]*w[0])*100
                if len(tenor)==3:
                    if len(w)==0:
                        w=[-1,2,-1]
                    spread[curr+tenor[0]+tenor[1]+tenor[2]]=(w[1]*self.dfbbg[self.bbgticker['bbgticker'][curr+type_+tenor[1]]]+w[0]*self.dfbbg[self.bbgticker['bbgticker'][curr+type_+tenor[0]]] +w[2]*self.dfbbg[self.bbgticker['bbgticker'][curr+type_+tenor[2]]])*100
        #govt2s10s.plot.scatter()
        return self.fill_na(spread,'ffill')

    
    def residualPlot(self,residual,tenor,tenor_label,title='', w='',n=100,filename=''):
        if isinstance(tenor, pd.Series):
            #tenor = tenor.as_matrix();
            tenor   = tenor.values
            tenor = tenor.tolist()
        fig, ax = plt.subplots(figsize=(8,6))
        # if w=='':
        #     w=.7;
        plt.bar(tenor,residual[-1,:], width=.2, alpha=0.4, align="center",color='green',label='Residuals',zorder=0)
        plt.scatter(tenor,residual[-6,:],marker=">",color='blue',label='week ago',zorder=1)
        plt.scatter(tenor,residual[-22,:],marker="<",color='red',label='month ago',zorder=2 )
        plt.xticks(tenor,tenor_label)
        plt.legend(loc=2 ,fontsize = 'x-small')
        plt.axhline(y=0)
        fig.autofmt_xdate();
        plt.title(title+' PCA residual plot by tenor: '+str(n)+' days')
        plt.savefig(filename+'residual.png', bbox_inches='tight',dpi = 300)
    
        plt.show() 

    def pcaplot(self,ev,labels):
        eigennum,varnum=ev.shape;
        for i in range(eigennum-1):
            fig,ax=plt.subplots();
            plt.scatter(ev[i],ev[i+1]);
            for j in range(len(labels)):
                 ax.annotate(labels[j],(ev[i][j],ev[i+1][j]));
            ax.set_xlabel('EigenVec '+str(i+1), color='g');
            ax.set_ylabel('EigenVec '+str(i+2), color='b');
            plt.show(); 

    def genpca(self,type_='',eigennum=3,title_='',w='',count=1000,filename='',plotpca=0):
        cols = self.dfbbg.columns;
        dates = self.dfbbg.tail(count).index;
        ctitle = "Period: "+ dates[0].strftime('%d/%m/%Y') +" to " + dates[-1].strftime('%d/%m/%Y')
        #orig=df.as_matrix();
        orig=self.dfbbg.tail(count).values;
        len_prefix=len(self.prefix)
        if type_=='FX':
            tenor = range(len(cols));
            tenor_label = [c.split()[0] for c in cols.values]   
        elif (len(type_)>2) & (type_[0:2] =='FX'):
            tenor=[c.split()[0][3:][:-1] for c in cols.values]
            tenor = [float(c) for c in tenor]
            tenor_label = [str(i)+'M' for i in tenor]                 
            
        else:
            tenor=[c.split()[0][len_prefix:] for c in cols.values]
            if 'F' in tenor:
                tenor[tenor.index('F')]='.5'
            tenor = [float(c) for c in tenor]
            tenor_label = [str(i)+'y' for i in tenor]                 
        pca=PCA(n_components=eigennum)
        pca.fit(self.dfbbg.tail(count))
        pca_ratios= pca.explained_variance_ratio_
        #cov=pca.get_covariance()
        pca_ratios_str=["{:.2f}%".format(x*100) for x in pca_ratios]
        pca_labels = ['PCA'+str(i+1) for i in range(eigennum)];
        ratio_labels=[str(pca_labels[i])+" : " + str(pca_ratios_str[i]) for i in range(len(pca_ratios_str))]
        eigenv = pca.components_;
        df_pca = pca.transform(self.dfbbg.tail(count))
        proj = pca.inverse_transform(df_pca)
        
        residuals = orig-proj;
        fig1 = plt.subplots(figsize=(8,6));
        plt.plot(tenor,eigenv.T);
        plt.xticks(tenor,tenor_label,rotation=45,fontsize=15);
        plt.legend(ratio_labels,loc=2 ,fontsize = 'small');
        plt.title(title_+" PCA Factors: "+ctitle,fontsize = 16);
        plt.show();
        plt.savefig(filename+str(count)+'pca.png', bbox_inches='tight',dpi = 300)

        fig1, ax1 = plt.subplots(figsize=(8,6))
        plt.plot(dates,df_pca)
        ax1.set_xlabel("date")
        fig1.autofmt_xdate()
        plt.legend(pca_labels,loc=2 ,fontsize = 'small')
        plt.title(title_+" PCA Factors: "+ctitle,fontsize = 16)
        plt.xticks(rotation=45,fontsize=15)

        plt.show()
        plt.savefig(filename+str(count)+'pcats.png', bbox_inches='tight',dpi = 300)

        self.residualPlot(residuals,tenor,tenor_label,title=title_,n=count,filename=filename)
        if plotpca == 1:
            self.pcaplot(eigenv,tenor_label)
        return (df_pca, residuals, pca,tenor)
    
    
    def zscore(self):
        self.dfbbg_z=pd.DataFrame()
        for col in self.dfbbg.columns:
            self.dfbbg_z[col+'_z'] = (self.dfbbg[col] - self.dfbbg[col].mean())/self.dfbbg[col].std(ddof=0)
            
    def zscore1(self,df):
        df_z=pd.DataFrame()
        for col in df.columns:
            df_z[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
        return df_z
    
    
    def getPolicyAction(self,curr='',retcol='Swap2y',days=[30,90]):
        policyticker=self.bbgticker[(self.bbgticker['Type']=='Policy') & (self.bbgticker['Code']==curr)].bbgticker[0]
        swap=self.bbgticker[(self.bbgticker['Type']==retcol) & (self.bbgticker['Code']==curr)].bbgticker[0]
        policy=self.getts([policyticker,swap])
        movecols=[]
        for day in days:
            policy['Move'+str(day)+'d']=(policy[swap].shift(-day)-policy[swap])*100
            movecols.append('Move'+str(day)+'d')
        swaplast=policy[swap][-1]
        policy['next']=0
        policy['Change']=policy[policyticker].diff()
        policy.loc[policy['Change'] >0 , 'next']=1
        policy.loc[policy['Change'] <0 , 'next']=-1
        policy.drop(policy[policy['next']==0].index,inplace=True)
        policy['date']=policy.index.values
        policy['prev']=policy['date'].shift(-1)
        policy['Pause']=policy['prev']-policy['date']
        policy.at[policy.index[-1],'Pause']=pd.Timedelta((datetime.today()-policy.index[-1].to_pydatetime()).days,unit='d')
        for day in days:
            if np.isnan(policy['Move'+str(day)+'d'][-1]):
                policy.at[policy.index[-1],'Move'+str(day)+'d']=(swaplast-policy[swap][-1])*100
        policy=policy.drop(columns=['date','next','prev'])
        
        return policy[[policyticker,'Change','Pause',swap]+movecols]
    
    def plotHist(self,curr='KRW',type_='Swap1y',sec='',dates=[]):
        if sec == '':
            sec=self.bbgticker[(self.bbgticker['Type']==type_) & (self.bbgticker['Code']==curr)].bbgticker[0]
        swap = self.getts([sec])
        if len(dates)==0:
            swap.plot()
            return swap;
        t0=np.where(swap.index==dates[0])[0][0]
        for d in dates[1:]:
            delta = np.where(swap.index==d)[0][0]-t0
            swap[curr+d]= swap[sec].shift(-delta)
        swap = swap[(swap.index>=pd.to_datetime(dates[0])-pd.Timedelta(days=30)) & (swap.index<=pd.to_datetime(dates[0])+pd.Timedelta(days=365))]   
        swap=swap.rename(columns={sec:curr+dates[0]})
        #fig, ax = plt.subplots(figsize=(8,6))

        swap.plot()
        plt.axvline(x=pd.to_datetime(dates[0]),color='red')
        plt.axvline(x=pd.to_datetime(dates[0])+pd.Timedelta(days=30),linestyle='--')
        plt.axvline(x=pd.to_datetime(dates[0])+pd.Timedelta(days=90),linestyle='--')
        plt.axvline(x=pd.to_datetime(dates[0])+pd.Timedelta(days=180),linestyle='--')
        plt.text(pd.to_datetime(dates[0]),0,'Policy Action Day',rotation=90,fontsize=13)
        plt.text(pd.to_datetime(dates[0])+pd.Timedelta(days=30),0,'30 Days',rotation=90,fontsize=13)
        plt.text(pd.to_datetime(dates[0])+pd.Timedelta(days=90),0,'90 Days',rotation=90,fontsize=13)
        plt.text(pd.to_datetime(dates[0])+pd.Timedelta(days=180),0,'180 Days',rotation=90,fontsize=13)
        for d in dates:
            plt.axhline(y=swap.iloc[np.where(swap.index==dates[0])[0][0]][curr+d])
            plt.text(swap.index[-30],swap.iloc[np.where(swap.index==dates[0])[0][0]][curr+d],d,va='center', ha='center',fontsize=14,backgroundcolor='w')
        plt.legend(fontsize='large',loc='upper left')
        plt.title('History repeats itself? '+ curr+ '  ' + type_, size=17)
        plt.xticks(rotation=45,fontsize=15)
        plt.yticks(fontsize=16)

        return swap
    
    
    def plotHist1(self,curr='KRW',type_='Swap1y',sec='',dates=[]):
        if sec == '':
            sec=self.bbgticker[(self.bbgticker['Type']==type_) & (self.bbgticker['Code']==curr)].bbgticker[0]
        swap = self.getts([sec])
        if len(dates)==0:
            swap.plot()
            return swap;
        t0=np.where(swap.index==dates[0])[0][0]
        for d in dates[1:]:
            delta = np.where(swap.index==d)[0][0]-t0
            swap[curr+d]= swap[sec].shift(-delta)
        swap = swap[(swap.index>=pd.to_datetime(dates[0])-pd.Timedelta(days=30)) & (swap.index<=pd.to_datetime(dates[0])+pd.Timedelta(days=365))]   
        swap=swap.rename(columns={sec:curr+dates[0]})
        #fig, ax = plt.subplots(figsize=(8,6))
        t0=np.where(swap.index==dates[0])[0][0]

        print(t0)
        for i in range(len(swap.columns)-1):
            print(t0)
            swap[swap.columns[i+1]]=swap[swap.columns[i+1]]+swap.iloc[t0,0]-swap.iloc[t0,i+1]
        swap.plot()
        plt.axvline(x=pd.to_datetime(dates[0]),color='red')
        plt.axvline(x=pd.to_datetime(dates[0])+pd.Timedelta(days=30),linestyle='--')
        plt.axvline(x=pd.to_datetime(dates[0])+pd.Timedelta(days=90),linestyle='--')
        plt.axvline(x=pd.to_datetime(dates[0])+pd.Timedelta(days=180),linestyle='--')
        plt.text(pd.to_datetime(dates[0]),0,'Policy Action Day',rotation=90,fontsize=13)
        plt.text(pd.to_datetime(dates[0])+pd.Timedelta(days=30),0,'30 Days',rotation=90,fontsize=13)
        plt.text(pd.to_datetime(dates[0])+pd.Timedelta(days=90),0,'90 Days',rotation=90,fontsize=13)
        plt.text(pd.to_datetime(dates[0])+pd.Timedelta(days=180),0,'180 Days',rotation=90,fontsize=13)
        for d in dates:
            plt.axhline(y=swap.iloc[np.where(swap.index==dates[0])[0][0]][curr+d])
            plt.text(swap.index[-30],swap.iloc[np.where(swap.index==dates[0])[0][0]][curr+d],d,va='center', ha='center',fontsize=14,backgroundcolor='w')
        plt.legend(fontsize='large',loc='upper left')
        plt.title('History repeats itself? '+ curr+ '  ' + type_, size=17)
        plt.xticks(rotation=45,fontsize=15)
        plt.yticks(fontsize=16)

        return swap

    
    def econPlot(self,curr='KRW',cpitarget=2,folder='korea/'):
        title1=curr+' Rate, CPI Inflation, GDP Growth'
        cpi = self.getbbg(type_='CPIYOY',currency=curr,fill='ffill').resample('M').last()
        gdp = self.getbbg(type_='GDPYOY',currency=curr,fill='ffill').resample('M').last()
        policy = self.getbbg(type_='Policy',currency=curr,fill='ffill').resample('M').last()
        cpi_l = cpi.tail(1).values[0][0]
        gdp_l = gdp.tail(1).values[0][0]
        policy_l = policy.tail(1).values[0][0]
        
        fig, ax = plt.subplots(figsize=(8,6))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=17)
        ax2 = ax.twinx()
        ax.plot(gdp.index,gdp,'g-',label='GDP YOY: '+str(gdp_l))
        ax2.plot(cpi.index,cpi,'r-',label='CPI YOY: '+str(cpi_l))
        ax2.axhline(y=cpitarget)
        ax2.text(cpi.index[10],cpitarget,'Inflation Target: '+str(cpitarget)+'%',va='center', ha='center',fontsize=12,backgroundcolor='w')

        ax2.plot(policy.index,policy,'b--',label='Policy Rate: '+str(policy_l))
        ax.set_xlabel('Date',fontsize='x-large');
        plt.title(title1,fontsize=17);
        ax.set_ylabel('GDP YOY (Green)', color='g',fontsize='x-large')
        ax.legend(fontsize='large',loc='upper left')
        ax2.legend(fontsize='large')
        plt.yticks(fontsize=17)
        fig.autofmt_xdate()
        plt.savefig(folder+'econ.png', bbox_inches='tight',dpi = 300)

        fig.show()

    def crossMktEcon(self,econ='CPIYOY',f='M',count=36):
        mktecon = self.getbbg(type_=econ,fill='ffill').resample(f).last()
        newcol=self.bbgticker[self.bbgticker.Type == econ].bbgticker.tolist()
        oldcol=self.bbgticker[self.bbgticker.Type == econ].Code.tolist()
        mktecon = mktecon.rename(columns=dict(zip(newcol, oldcol)))
        plotS.plotRange(mktecon,count=36,title='Cross Market (freq: '+f+' ): '+ econ,filename='korea/mkt'+econ+f+'.png')

    def gdpforecast(self,curr='KRW',t=[21,22],filename='korea/gdpf.png'):
        f21 = self.getbbg(type_='Forecast21',currency=curr,fill='ffill').tail(500)
        f22 = self.getbbg(type_='Forecast22',currency=curr,fill='ffill').tail(500)
        gdp = self.getbbg(type_='GDPYOY',currency=curr,fill='ffill').tail(500)
        f21_l = f21.tail(1).values[0][0]
        f22_l = f22.tail(1).values[0][0]
        gdp_l = gdp.tail(1).values[0][0]
        fig, ax = plt.subplots(figsize=(8,6))
        ax.yaxis.tick_right()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=20)
        ax.plot(f21.index,f21,'g-',label='GDP Forecast for 2021: '+str(f21_l))
        ax.plot(f22.index,f22,'b-',label='GDP Forecast for 2022: '+str(f22_l))
        ax.plot(gdp.index,gdp,'r--',label='Actual GDP YoY: '+str(gdp_l))
        plt.title('GDP forecast (%): '+curr,fontsize=21)
        ax.legend(fontsize='x-large')
        fig.autofmt_xdate()
        fig.show()
        plt.savefig(filename, bbox_inches='tight',dpi = 300)

        
    def hhd(self,curr='KRW'):
        title1=curr+': Household Debt Levels and Growth'
        hhd = self.getbbg(type_='HHD',currency=curr,fill='ffill').resample('Q').last()[:-1]
        hhdyoy = self.getbbg(type_='HHDGrowth',currency=curr,fill='ffill').resample('Q').last()[:-1]
        plotS.twoAxisPlot(hhd.index,hhd,hhdyoy,'Date','Household Debt ('+curr+')','Percent (YoY) : '+str(hhdyoy.tail(1).values[0][0]),title1,filename='korea/hhd.png')
        
    def exports(self,curr='KRW'):
        title1='China, Chips Weigh on South Korea Exports'
        ExportSC = self.getbbg(type_='ExportSC',currency=curr,fill='ffill').resample('M').last()[:-1]
        ExportCH = self.getbbg(type_='ExportCH',currency=curr,fill='ffill').resample('M').last()[:-1]
        ExportYOY = self.getbbg(type_='ExportYOY',currency=curr,fill='ffill').resample('M').last()[:-1]
        ExportSC_l = ExportSC.tail(1).values[0][0]
        ExportCH_l = ExportCH.tail(1).values[0][0]
        ExportYOY_l = ExportYOY.tail(1).values[0][0]
        fig, ax = plt.subplots(figsize=(9,6))
        ax.yaxis.tick_right()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.plot(ExportSC.index,ExportSC,'g-',label=curr+' Semiconductor Export: '+str(ExportSC_l))
        ax.plot(ExportCH.index,ExportCH,'b-',label=curr+' Exports to China: '+ str(ExportCH_l))
        ax.plot(ExportYOY.index,ExportYOY,'r--',label=curr+ ' Total Exports: '+str(ExportYOY_l))
        plt.title(title1,fontsize=17)
        ax.legend(fontsize='large')
        fig.autofmt_xdate()
        fig.show()
        plt.savefig('korea/exports.png', bbox_inches='tight',dpi = 300)

    def trade(self,curr='KRW'):
        title1=curr+' : Trade Data Exports/Imports (YoY) and TradeBalance'
        TradeBal = self.getbbg(type_='TrdBal',currency=curr,fill='ffill').resample('M').last()[:-1].tail(60)
        ImportYOY = self.getbbg(type_='ImportYOY',currency=curr,fill='ffill').resample('M').last()[:-1].tail(60)
        ExportYOY = self.getbbg(type_='ExportYOY',currency=curr,fill='ffill').resample('M').last()[:-1].tail(60)
        TradeBal_l = TradeBal.tail(1).values[0][0]
        ImportYOY_l = ImportYOY.tail(1).values[0][0]
        ExportYOY_l = ExportYOY.tail(1).values[0][0]
        fig, ax = plt.subplots(figsize=(9,6))
        ax.yaxis.tick_right()

        #ax.yaxis.tick_right()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=20)
        ax.plot(ImportYOY.index,ImportYOY,'b-',label=' Import YoY: '+ str(ImportYOY_l),zorder=10,lw=3)
        ax.plot(ExportYOY.index,ExportYOY,'r-',label= ' Export YoY: '+str(ExportYOY_l),zorder=20,lw=3)
        ax2 = ax.twinx()
        ax2.bar(TradeBal.index,TradeBal.iloc[:,0],color='g',width=20,alpha=.5, label=' Trade Balance (m USD)'+str(TradeBal_l),zorder=0)
        ax2.yaxis.tick_left()
        
        plt.title(title1,fontsize=17)
        ax.legend(fontsize='large',loc='upper left')
        ax2.legend(fontsize='medium',loc=0)
        ax.yaxis.tick_right()

        fig.autofmt_xdate()
        fig.show()
        plt.savefig('korea/trade.png', bbox_inches='tight',dpi = 300)
    
    def deficit(self,curr='KRW',freq='Q'):
        budget = self.getbbg(type_='Fiscal',currency='KRW',fill='ffill')
        gdp = self.getbbg(type_='GDP',currency='KRW',fill='ffill')
        gdp=gdp.resample('Q').last()[:-1]
        budget=budget.resample('Q').last()[:-1]
        gdp = gdp.groupby(gdp.index.year).cumsum()
        budget = budget.join(gdp)
        gdp_tick = self.bbgticker[self.bbgticker.index==curr+'GDP'].bbgticker[0]
        budget_tick = self.bbgticker[self.bbgticker.index==curr+'Fiscal'].bbgticker[0]
        budget['budget_gdp']=budget[budget_tick]/budget[gdp_tick]*100
        return budget
    
    def consumersentiment(self,curr='KRW',count=1000):
        title1=curr+' Outstanding Credit and Consumer Income, Sentiment'
        cc = self.getbbg(type_='ConsumerConf',currency=curr,fill='ffill').resample('Q').last()[:-1]
        incomeyoy = self.getbbg(type_='IncomeYOY',currency=curr,fill='ffill').resample('Q').last()[:-1]
        hhdyoy = self.getbbg(type_='HHDGrowth',currency=curr,fill='ffill').resample('Q').last()[:-1]
        cc_l = cc.tail(1).values[0][0]
        incomeyoy_l = incomeyoy.tail(1).values[0][0]
        hhdyoy_l = hhdyoy.tail(1).values[0][0]        
        fig, ax = plt.subplots(figsize=(9,6))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=15)
        ax2 = ax.twinx()
        ax.plot(cc.index,cc,'g-',label='Consumer Confidence (LHS): '+str(cc_l))
        ax2.plot(incomeyoy.index,incomeyoy,'r-',label='Household Income YOY %: '+str(incomeyoy_l))
        ax2.plot(hhdyoy.index,hhdyoy,'b--',label='Outstanding Household credit %: '+str(hhdyoy_l))
        ax.set_xlabel('Date',fontsize='x-large');
        plt.title(title1,fontsize=17);
        ax.set_ylabel('Consumer Confidence (Green)', color='g',fontsize='large')
        ax.legend(fontsize='large',loc='upper left')
        ax2.legend(fontsize='large')
        plt.yticks(fontsize=15)
        fig.autofmt_xdate()
        fig.show()
        plt.savefig('korea/csi.png', bbox_inches='tight',dpi = 300)

        
    def sentiment(self,curr='KRW',count=1000):
        title1=curr+' Sentiment Index'
        bsi_m = self.getbbg(type_='BSI_M',currency='KRW',fill='ffill').resample('M').last()
        bsi_nm = self.getbbg(type_='BSI_NM',currency='KRW',fill='ffill').resample('M').last()
        csi = self.getbbg(type_='ConsumerConf',currency=curr,fill='ffill').resample('M').last()
        bsi_m_l = bsi_m.tail(1).values[0][0]
        bsi_nm_l = bsi_nm.tail(1).values[0][0]
        csi_l = csi.tail(1).values[0][0]        
        fig, ax = plt.subplots()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=20)
        ax.plot(bsi_m.index,bsi_m,'g-',label=curr+' Business Surve Index (Manufacturing): '+str(bsi_m_l))
        ax.plot(bsi_nm.index,bsi_nm,'b-',label=curr+' Business Surve Index (Non Manufacturing): '+ str(bsi_nm_l))
        ax.plot(csi.index,csi,'r--',label=curr+ ' Consumer Sentiment Index: '+str(csi_l))
        plt.title(title1,fontsize=21)
        ax.legend(fontsize='x-large')
        fig.autofmt_xdate()
        fig.show()

    def y_fmt(x, y):
        return '{:2.2e}'.format(x).replace('e', 'x10^')
    def netfut(self,curr='KRW',count=1000,d='20191231'):
        title1=curr+' Net foreigner KTB Futures Purchase (YTD)'
        Fut3y = self.getbbg(type_='Fut3y_Net',currency='KRW',fill='0')
        Fut10y = self.getbbg(type_='Fut10y_Net',currency='KRW',fill='0')
        Bond = self.getbbg(type_='BondFlow',currency='KRW',fill='0')
        Fut3y = Fut3y[Fut3y.index>d]
        Fut10y = Fut10y[Fut10y.index>d]
        Fut3y = Fut3y.cumsum()
        Fut10y = Fut10y.cumsum()
        fig, ax = plt.subplots(figsize=(9,6))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        ax.plot(Fut3y.index,Fut3y,'g-',label=curr+' 3y KTB Contracts')
        ax.plot(Fut10y.index,Fut10y,'b-',label=curr+' 10y KTB Contracts ')
        #xlabels = ['{:,.2f}'.format(x) + 'K' for x in g.get_xticks()/1000]
        #g.set_xticklabels(xlabels)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))
        ax.set_ylabel("Net Purchase of Contracts", size=14)
        ax.set_xlabel("Date", size=14)

        plt.title(title1,fontsize=16)
        ax.legend(fontsize='x-large')
        fig.autofmt_xdate()
        fig.show()
        plt.savefig('korea/futbuy.png', bbox_inches='tight',dpi = 300)

        return Fut3y,Fut10y

        
    def drawdown(df,window=500,chg=1):
        if chg==1:
            return df-df.rolling(window).max()
        if chg==0:
            return df/df.rolling(window).max() - 1
            
    def extrabudget(self,curr='KRW'):
        title1=curr+' Supplementary Budget'
        ebudget = self.getbbg(type_='SuppBudget',currency='KRW',fill='ffill').resample('Q').last()[:-1]
        
    def gdp_cpi(self,curr='KRW'):
        cpi = self.getbbg(type_='CPIYOY',currency=curr,fill='ffill')
        gdp = self.getbbg(type_='GDPYOY',currency=curr,fill='ffill')
        swp = self.getbbg(type_='Policy',currency=curr,fill='ffill')
        swp=swp.resample('Q').last().diff(-2)*-100
        swp=swp[:-1]
        cpi=cpi.resample('Q').last()[:-1]
        gdp=gdp.resample('Q').last()[:-1]
        cpi=cpi.rename(columns={cpi.columns[0]:'cpi'})
        gdp=gdp.rename(columns={gdp.columns[0]:'gdp'})
        swp=swp.rename(columns={swp.columns[0]:'Policy'})
        swp
        swp[np.isnan(swp)]=0
        result=pd.concat([cpi, gdp, swp], axis=1, sort=False)
        result['action']=' '
        result.loc[result['Policy']>0,'action']='Hike'
        result.loc[result['Policy']<0,'action']='Cut'
        result['Policy']=abs(result['Policy'])
        labels = cpi.index.to_period().strftime(' %y-%q').to_list()
        result['label']=labels
        x=cpi[cpi.columns[0]].values.tolist()
        y=gdp[gdp.columns[0]].values.tolist()
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]))

    def fxcarry(self,mth='3M',count=100):
        carry = self.getbbg(type_='Curr',currency='USD',fill='ffill')
        currlist = self.bbgticker.Code[self.bbgticker.Type=='Curr'].values.tolist()
        currlist.remove('USD')
        currlist.remove('MYR')
        mult=1
        if mth=='3M':
            mult=4
        fx = pd.DataFrame()
        for c in currlist:
            FX = self.bbgticker[self.bbgticker.index==c+'Curr']['bbgticker']
            FX_3M = self.bbgticker[self.bbgticker.index==c+'FX_'+mth]['bbgticker']
            FX3M = self.bbgticker[self.bbgticker.index==c+'FX'+mth]['bbgticker']
            fxspot = self.getbbg(type_='Curr',currency=c,fill='ffill')
            fx[c]=fxspot.iloc[-count:,0]

            if (len(FX3M)>0) & (len(FX_3M)>0) & (len(FX)>0):
                fxfwd = self.getbbg(type_='FX'+mth,currency=c,fill='ffill')
                fx_or = self.getbbg(type_='FX_'+mth,currency=c,fill='ffill')
                carry[c] = fxfwd.iloc[-count:,0]*mult*100/(fx_or.iloc[-count:,0]-fxfwd.iloc[-count:,0]/self.fxmult.loc[c][0])/self.fxmult.loc[c][0]
            elif (len(FX)>0) & (len(FX3M)>0):
                fxspot = self.getbbg(type_='Curr',currency=c,fill='ffill')
                fxfwd = self.getbbg(type_='FX'+mth,currency=c,fill='ffill')
                carry[c] = fxfwd.iloc[-count:,0]*mult*100/fxspot.iloc[-count:,0]/self.fxmult.loc[c][0]
        carry=carry.drop(columns=['USD CURNCY'])
        
        fx_std=fx.pct_change().rolling(50).std()*100
        #plotBarBox(carry.resample('w').last(),count=50,spread_name=mth+' FX Carry')
        #plotRange(carry.resample('w').last(),count=50,title=mth+' FX Carry')
        z = self.zscore1(fx.tail(count))
        plotS.plotRange(carry,count=count,title=mth+' FX Carry (Annualized)',filename='korea/'+mth+'carry.png')
        plotS.plotRange(z,count=count,title=' Z score FX Range',filename='korea/'+mth+'carry_z.png')
        return carry,fx_std
    
    
    def mvavg(self,df,window=200,diff=0,title=''):
        m=df.rolling(center=False,window=window).mean()    
        if diff==0:
            rv=(df-m)/m*100
        if diff==1:
            rv = (df-m)*100
        rv = rv.dropna()
        cols=rv.columns.values.tolist()
        xlabel=[c.split(' ')[0] for c in cols]
        fig, ax = plt.subplots(figsize=(9,6))
        plt.bar(rv.columns.values.tolist(),rv.tail(1).values[0].tolist())
        plt.xticks(cols,xlabel)
        plt.xticks(rotation=45,size=13);
        plt.title('Cross mkt distance from '+str(window) +  '  day moving average',fontsize=16)
        plt.savefig('korea/mvavg'+str(window)+'.png', bbox_inches='tight',dpi = 300)

    def adjustFX(self,df,currlist=['EUR','GBP','AUD','NZD']):
        cols = df.columns.to_list()
        for c in cols:
            if c[0:3] in currlist:
                df[c]=1/df[c]
        return df
        
    def bondswap(self,curr='KRW',tenor='3y',count=600, plot=1,mult=100):
        bond = self.getbbg(type_='Govt'+tenor,currency=curr,fill='ffill')
        swap = self.getbbg(type_='Swap'+tenor,currency=curr,fill='ffill')
        #twoAxisPlot(bond.tail(count).index,swap.iloc[:count,0],bond.iloc[:count,0],"Date",y1label = curr+ '  '+ tenor + '  Swap', y2label = curr + '  '+ tenor + '  Bond', title=curr + '  '+tenor+ '  Bond Swap Spread')
        if plot ==1:
            plotS.spreadPlot(bond.tail(count).index,swap.iloc[-count:,0],bond.iloc[-count:,0],xlabel='Date',ylabel='Rate  ( %)',y2label=curr+' '+ tenor + ' Bond',y1label=curr+' '+ tenor + ' Swap', title=curr+' : '+ tenor+ '  Bond vs Swap Spread'  , filename='korea/bondswap'+tenor+'.png')
        return (bond.iloc[-count:,0]-swap.iloc[-count:,0])*mult
    
    def bondSwapforAll(self,ccylist=['USD','EUR','JPY','SGD','KRW','HKD','AUD','CNY', 'TWD'],tenor='10y',count=600, plot=1,mult=100):
        bond= self.gettsforCCYlist(ccylist=ccylist,type_='Govt'+tenor)
        swap= self.gettsforCCYlist(ccylist=ccylist,type_='Swap'+tenor)    
        bss=(bond-swap)*mult
        if plot ==1 :
            plotS.plotRange(bss.tail(count),filename='korea/bssAll.png', title='Cross Mkt Bond Swap Spread: '+ tenor,count=count)   
        return (bond, swap)

    def aswforAll(self,ccylist=['KRW','CNY', 'TWD','INR','IDR','PHP'],tenor='2y',count=600,plot=1,mult=100):
        bond = self.gettsforCCYlist(ccylist=ccylist,type_='Govt'+tenor)
        basis = self.gettsforCCYlist(ccylist=ccylist,type_='Basis'+tenor)
        cross = self.gettsforCCYlist(ccylist=ccylist,type_='Cross'+tenor)
        basis.KRW = cross.KRW
        #tenorS = self.getbbg(type_='Tenor3m6m'+tenor,currency='USD',fill='ffill')
        asw= (bond - basis)*mult
        if plot ==1 :
            plotS.plotRange(asw.tail(count),filename='korea/aswAll.png', title='Cross Mkt ASW Spread as USD Fixed: '+ tenor,count=count)   
        return (bond, basis)
            
    def cddeb(self,curr='KRW',count=600):
        cd = self.getbbg(type_='Float',currency=curr,fill='ffill')
        deb = self.getbbg(type_='Deb6m',currency=curr,fill='ffill')
        #twoAxisPlot(bond.tail(count).index,swap.iloc[:count,0],bond.iloc[:count,0],"Date",y1label = curr+ '  '+ tenor + '  Swap', y2label = curr + '  '+ tenor + '  Bond', title=curr + '  '+tenor+ '  Bond Swap Spread')
        plotS.spreadPlot(cd.tail(count).index,cd.iloc[-count:,0],deb.iloc[-count:,0],xlabel='Date',ylabel='Rate  ( %)',y2label=curr+' Debenture Rate',y1label=curr+' Float Rate', title=curr+' : Debenture vs Floating Index'  , filename='korea/'+curr+'cddeb.png')

    def cdpolicy(self,curr='KRW',count=600):
        cd = self.getbbg(type_='Float',currency=curr,fill='ffill')
        policy = self.getbbg(type_='Policy',currency=curr,fill='ffill')
        #twoAxisPlot(bond.tail(count).index,swap.iloc[:count,0],bond.iloc[:count,0],"Date",y1label = curr+ '  '+ tenor + '  Swap', y2label = curr + '  '+ tenor + '  Bond', title=curr + '  '+tenor+ '  Bond Swap Spread')
        plotS.spreadPlot(policy.tail(count).index,policy.iloc[-count:,0],cd.iloc[-count:,0],xlabel='Date',ylabel='Rate  ( %)',y2label=curr+' Float Rate',y1label=curr+' Policy Rate', title=curr+' : FLoating Index vs Policy'  , filename='korea/'+curr+'cdpolicy.png')
    
    def wfii(self,curr='KRW',d='20191231'):
        bond = self.getbbg(type_='BondFlow',currency=curr,fill='0')
        eq = self.getbbg(type_='Eqflow',currency=curr,fill='0')
        bond= bond[bond.index>d]
        #b1 = bond.resample(freq='M').sum()
        
        eq = eq[eq.index>d]
        fig, ax = plt.subplots(figsize=(9,6))
        bond = bond.cumsum()
        eq = eq.cumsum()
        bond.join(eq).plot.area(stacked=False,ax=ax)
        ax.legend(['Bond Inflow','Equity Inflow'],fontsize='large')
        #ax2 = ax.twinx()
        #ax2.bar(b1.index,b1.iloc[:,0],color='b',width=20,alpha=.5, label=' Monthly Inflow',zorder=0)

        plt.title('Foreign portfolio inflow (m$): '+ curr,size=18)
        plt.xticks(rotation=45,size=14);
        plt.yticks(fontsize=14)
        plt.savefig('korea/wfii'+curr+'.png', bbox_inches='tight',dpi = 300)

    def asw(self,curr='KRW',tenor='2y',count=600):
        bond = self.getbbg(type_='Govt'+tenor,currency=curr,fill='ffill')
        cross = self.getbbg(type_='Cross'+tenor,currency=curr,fill='ffill')
        tenorS = self.getbbg(type_='Tenor3m6m'+tenor,currency='USD',fill='ffill')
        #usswap = self.getbbg(type_='Swap'+tenor,currency='USD')
        basis = self.getbbg(type_='Basis'+tenor,currency=curr,fill='ffill')
        asw=pd.DataFrame()
        asw['ASW'+tenor]= (bond.iloc[:,0] -cross.iloc[:,0])*100 +tenorS.iloc[:,0] 
        asw = asw.tail(count)
        basis = basis.tail(count)
        plotS.twoAxisPlot(asw.index,basis,asw,'Date','Cross currency Basis: '+curr+ ' '+tenor,'Asset Swap('+curr+' '+ tenor +') USD3m+X','ASW level (3m+X) & xccy basis:' + curr+' '+tenor,filename='korea/asw'+curr+'.png')
        
    def BL(self,curr='KRW',count=900,end_=1):
        
        bl_lc = self.getbbg(type_='BL_LC',currency=curr,fill='ffill').resample('M').last().tail(50)[:-end_]
        bl_sme = self.getbbg(type_='BL_SME',currency=curr,fill='ffill').resample('M').last().tail(50)[:-end_]
        bl_hh = self.getbbg(type_='BL_HH',currency=curr,fill='ffill').resample('M').last().tail(50)[:-end_]
        fig, ax = plt.subplots(figsize=(9,6))
        p1=plt.bar(bl_lc.index, bl_lc.iloc[:,0],width=20)
        p2=plt.bar(bl_sme.index, bl_sme.iloc[:,0],width=12,bottom=bl_lc.iloc[:,0])
        plt.ylabel('Bank Lending (Trn won)')
        plt.title(curr+':  Bank Lending to Large Corp & SMEs Change MoM(Trn Won)',size=17)
        plt.legend((p1[0], p2[0]), ('Large Corporate', 'SME'))
        plt.xticks(rotation=90);

        plt.savefig('korea/bl_'+curr+'.png', bbox_inches='tight',dpi = 300)
        fig, ax = plt.subplots(figsize=(9,6))
        p1=plt.bar(bl_hh.index, bl_hh.iloc[:,0],width=20)
        plt.ylabel('Bank Lending to HouseHolds')
        plt.title(curr+': Bank Lending to HouseHolds Change MoM(Trn Won)',size=17)
        plt.xticks(rotation=90);
        plt.savefig('korea/blhh_'+curr+'.png', bbox_inches='tight',dpi = 300)
        
        
    def cpi(self,curr='KRW'):
        title1=curr+': Inflation'
        cpi = self.getbbg(type_='CPIYOY',currency=curr,fill='ffill').resample('M').last()[:-1]
        cpiexp = self.getbbg(type_='InflExp',currency=curr,fill='ffill').resample('M').last()[:-1]
        cpiexfe = self.getbbg(type_='CPIexFE',currency=curr,fill='ffill').resample('M').last()[:-1]
        cpi_l = cpi.tail(1).values[0][0]
        cpiexp_l = cpiexp.tail(1).values[0][0]
        cpiexfe_l = cpiexfe.tail(1).values[0][0]
        fig, ax = plt.subplots(figsize=(9,6))
        ax.yaxis.tick_right()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.plot(cpi.index,cpi,'g-',label='CPI YOY: '+str(cpi_l))
        ax.plot(cpiexp.index,cpiexp,'b-',label='Household Inflation expectation: '+ str(cpiexp_l))
        ax.plot(cpiexfe.index,cpiexfe,'r-',label='CPI Excluding Food & Energy: '+str(cpiexfe_l))
        plt.title(title1,fontsize=17)
        ax.legend(fontsize='large')
        plt.tick_params(labelleft=True, labelright=True)
        fig.autofmt_xdate()
        fig.show()
        plt.savefig('korea/cpi'+curr+'.png', bbox_inches='tight',dpi = 300)
        
        
    def uscnplot(self,ccy1='USD',ccy2='CNY',type_='Govt2y',count=2500,filename=''):
        cc1b = self.getbbg(type_=type_,currency=ccy1,fill='ffill')
        cc2b = self.getbbg(type_=type_,currency=ccy2,fill='ffill')
        fx = self.getbbg(type_='Curr',currency=ccy2,fill='ffill')
        bspread = pd.DataFrame()
        bspread[ccy1+'_'+ccy2]=(cc2b.iloc[:,0]-cc1b.iloc[:,0])*-100
        plotS.twoAxisPlot(cc1b.tail(count).index,bspread.iloc[-count:,0],fx.iloc[-count:,0],"Date",ccy1+' & ' +ccy2 +' ' +type_+" yield spread",ccy1+ccy2,ccy1+' & ' +ccy2+'  '+type_+' yield spread and '+ccy1+ccy2,filename=filename+ccy1+ccy2+type_+'diff.png')


    def employment(self,curr='KRW',count=50,folder='korea/'):
        emprate = self.getbbg(type_='UnEmployment',currency=curr,fill='ffill').resample('M').last().tail(count)[:-1]
        emp = self.getbbg(type_='Employment',currency=curr,fill='ffill').resample('M').last().tail(count)[:-1]
        emppart = self.getbbg(type_='LaborParticipation',currency=curr,fill='ffill').resample('M').last().tail(count)[:-1]
        emprate_l = emprate.tail(1).values[0][0]
        emp_l = emp.tail(1).values[0][0]
        emppart_l = emppart.tail(1).values[0][0]

        
        fig, (ax1, ax2) = plt.subplots(2, sharex=True,figsize=(9,8))
        ax3 = ax1.twinx();

        ax1.plot(emppart.index,emppart.iloc[:,0],'g-',label='Employment Participation Rate (RHS):'+str(emppart_l),zorder=1)
        ax3.plot(emp.index,emp.iloc[:,0],'r--',label='Total Employment (1000s) (LHS): '+str(emp_l))
        #ax3.yaxis.tick_left()
        #ax1.yaxis.tick_right()
        ax2.plot(emprate.index,emprate.iloc[:,0],'grey', label = 'Unemployement Rate: '+str(emprate_l))
        fig.suptitle(curr+': Labor Market',size=17)
        ax1.set_ylabel('Labor Force Paricipation Rate', color='g',size=14)
        ax3.set_ylabel('Total Employment', color='b',size=14)
        ax2.set_ylabel('Unemployment Rate', color='grey',size=14)  
        ax1.tick_params(axis='y', which='major', pad=20)
        ax3.tick_params(axis='y', which='major', pad=20)
        ax1.legend(fontsize='medium',loc=4)
        ax3.legend(fontsize='medium',loc=0)
        ax2.legend(fontsize='medium')
        #ax1.yaxis.tick_right()
        
        ax1.set_xlabel('Date',fontsize = 15);
        ax1.xaxis.grid(True) 
        ax1.yaxis.grid(True) 
        #plt.locator_params(axis='x', nbins=14) #Show five dates
    
        #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m:%Y"))
        ax2.xaxis.grid(True) 
        plt.xticks(rotation=45,size=15)
        #fig.autofmt_xdate()
        fig.show()
        plt.savefig(folder+curr+'LaborMkt.png', bbox_inches='tight',dpi = 300)
    
    def equity(self,curr='KRW',count=1000,d='20191231'):
        title1=curr+' Equity Performance (YTD)'
        equity1 = self.getbbg(type_='Equity',currency='KRW',fill='ffill')
        equity2 = self.getbbg(type_='Equity2',currency='KRW',fill='ffill')
        equity1 = equity1[equity1.index>d]
        equity2 = equity2[equity2.index>d]
        fig, ax = plt.subplots(figsize=(9,6))
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax1 = ax.twinx();

        ax.plot(equity1.index,equity1.iloc[:,0],'g-',label=equity1.columns[0].split(" ")[0]+" :"+str(equity1.tail(1).values[0][0]))
        ax1.plot(equity2.index,equity2.iloc[:,0],'b-',label=equity2.columns[0].split(" ")[0]+" :"+str(equity2.tail(2).values[0][0]))
        #ax.set_ylabel("Equity", size=14)
        ax.set_xlabel("Date", size=14)
        #ax.yaxis.tick_right()
        #ax1.yaxis.tick_left()
        plt.title(title1,fontsize=16)
        ax.legend(fontsize='large',loc=4)
        ax1.legend(fontsize='large',loc=0)
        ax.set_ylabel(equity1.columns[0].split(" ")[0], color='g',size=14)
        ax1.set_ylabel(equity2.columns[0].split(" ")[0], color='b',size=14)
        ax.tick_params(axis='y', which='major', pad=20)
        ax1.tick_params(axis='y', which='major', pad=20)
       
        fig.autofmt_xdate()
        plt.xticks(rotation=45,size=15)

        fig.show()
        plt.savefig('korea/equity.png', bbox_inches='tight',dpi = 300)

    def gen2dpca(self,type_='',eigennum=3,title_='',w='',count=1000,filename=''):
        cols = self.dfbbg.columns;
        dates = self.dfbbg.tail(count).index;
        ctitle = "Period: "+ dates[0].strftime('%d/%m/%Y') +" to " + dates[-1].strftime('%d/%m/%Y')
        #orig=df.as_matrix();
        orig=self.dfbbg.tail(count).values;
        len_prefix=len(self.prefix)
        if type_=='FX':
            tenor = range(len(cols));
            tenor_label = [c.split()[0] for c in cols.values]   
        elif (len(type_)>2) & (type_[0:2] =='FX'):
            tenor=[c.split()[0][3:][:-1] for c in cols.values]
            tenor = [float(c) for c in tenor]
            tenor_label = [str(i)+'M' for i in tenor]                 
            
        else:
            tenor=[c.split()[0][len_prefix:] for c in cols.values]
            if 'F' in tenor:
                tenor[tenor.index('F')]='.5'
            tenor = [float(c) for c in tenor]
            tenor_label = [str(i)+'y' for i in tenor]                 
        pca=PCA(n_components=eigennum)
        pca.fit(self.dfbbg.tail(count))
        pca_ratios= pca.explained_variance_ratio_
        #cov=pca.get_covariance()
        pca_ratios_str=["{:.2f}%".format(x*100) for x in pca_ratios]
        pca_labels = ['PCA'+str(i+1) for i in range(eigennum)];
        ratio_labels=[str(pca_labels[i])+" : " + str(pca_ratios_str[i]) for i in range(len(pca_ratios_str))]
        eigenv = pca.components_;
        df_pca = pca.transform(self.dfbbg.tail(count))
        proj = pca.inverse_transform(df_pca)
        
        residuals = orig-proj;
        fig1 = plt.subplots(figsize=(8,6));
        plt.plot(tenor,eigenv.T);
        plt.xticks(tenor,tenor_label,rotation=45,fontsize=15);
        plt.legend(ratio_labels,loc=2 ,fontsize = 'small');
        plt.title(title_+" PCA Factors: "+ctitle,fontsize = 16);
        plt.show();
        plt.savefig('korea/pca'+filename+str(count)+'.png', bbox_inches='tight',dpi = 300)
        e1 = [c[0] for c in eigenv.T]
        e2 = [c[1] for c in eigenv.T]
        e3 = [c[2] for c in eigenv.T]
        ev =  pd.DataFrame()
        ev['swaption']=tenor_label
        ev['expiry'] = ev['swaption'].str[4:6]
        ev['maturity'] = ev['swaption'].str[6:]
        ev['e1']=e1
        ev['e2']=e2
        ev['e3']=e3

        fig1, ax1 = plt.subplots(figsize=(8,6))
        plt.plot(dates,df_pca)
        ax1.set_xlabel("date")
        fig1.autofmt_xdate()
        plt.legend(pca_labels,loc=2 ,fontsize = 'small')
        plt.title(title_+" PCA Factors: "+ctitle,fontsize = 16)
        plt.xticks(rotation=45,fontsize=15)

        plt.show()
        plt.savefig('korea/pcats'+filename+str(count)+'.png', bbox_inches='tight',dpi = 300)

        self.residualPlot(residuals,tenor,tenor_label,title=title_,n=count,filename=filename)
        self.pcaplot(eigenv,tenor_label)
        return (df_pca, residuals, pca,tenor, tenor_label,eigenv.T)


    def genpca1(self,df,type_='',eigennum=3,title_='',w='',count=1000,filename='',plotpca=0):
        cols = df.columns;
        dates = df.tail(count).index;
        ctitle = "Period: "+ dates[0].strftime('%d/%m/%Y') +" to " + dates[-1].strftime('%d/%m/%Y')
        #orig=df.as_matrix();
        orig=df.tail(count).values;
        len_prefix=5
        if type_=='FX':
            tenor = range(len(cols));
            tenor_label = [c.split()[0] for c in cols.values]   
        elif (len(type_)>2) & (type_[0:2] =='FX'):
            tenor=[c.split()[0][3:][:-1] for c in cols.values]
            tenor = [float(c) for c in tenor]
            tenor_label = [str(i)+'M' for i in tenor]                 
            
        else:
            tenor=[c.split()[0][len_prefix:] for c in cols.values]
            if 'F' in tenor:
                tenor[tenor.index('F')]='.5'
            tenor = [float(c) for c in tenor]
            tenor_label = [str(i)+'y' for i in tenor]                 
        pca=PCA(n_components=eigennum)
        pca.fit(df.tail(count))
        pca_ratios= pca.explained_variance_ratio_
        #cov=pca.get_covariance()
        pca_ratios_str=["{:.2f}%".format(x*100) for x in pca_ratios]
        pca_labels = ['PCA'+str(i+1) for i in range(eigennum)];
        ratio_labels=[str(pca_labels[i])+" : " + str(pca_ratios_str[i]) for i in range(len(pca_ratios_str))]
        eigenv = pca.components_;
        df_pca = pca.transform(df.tail(count))
        proj = pca.inverse_transform(df_pca)
        
        residuals = orig-proj;
        fig1 = plt.subplots(figsize=(8,6));
        plt.plot(tenor,eigenv.T);
        plt.xticks(tenor,tenor_label,rotation=45,fontsize=15);
        plt.legend(ratio_labels,loc=2 ,fontsize = 'small');
        plt.title(title_+" PCA Factors: "+ctitle,fontsize = 16);
        plt.show();
        plt.savefig('korea/pca'+filename+str(count)+'.png', bbox_inches='tight',dpi = 300)

        fig1, ax1 = plt.subplots(figsize=(8,6))
        plt.plot(dates,df_pca)
        ax1.set_xlabel("date")
        fig1.autofmt_xdate()
        plt.legend(pca_labels,loc=2 ,fontsize = 'small')
        plt.title(title_+" PCA Factors: "+ctitle,fontsize = 16)
        plt.xticks(rotation=45,fontsize=15)

        plt.show()
        plt.savefig('korea/pcats'+filename+str(count)+'.png', bbox_inches='tight',dpi = 300)

        self.residualPlot(residuals,tenor,tenor_label,title=title_,n=count,filename=filename)
        if plotpca == 1:
            self.pcaplot(eigenv,tenor_label)
        return (df_pca, residuals, pca,tenor)
    
    def compareFwdCurve(self,fwdprefix='KWFS01',fwd='01',fflag=1):
        fwd_b = bbgData()
        if fwd=='':
            fwd=0
        fwd_b.genSwaptickers(self.currency,tenor_set=self.tenor_set,fwdflag=fflag,start=fwd,method='')
        fwd_b.populate(start=self.dfbbg.index[0].strftime("%Y%m%d"),fill='ffill')
        
        scurve = self.dfbbg.tail(1)
        fwdcurve = fwd_b.dfbbg.tail(1)
        tenors = [str(c)+'y'  for c in self.bbgticker.Maturity]
        y1label = 'Spot '
        if self.start!='0':
            y1label = str(self.start) +'y fwd '
            
        plotS.spreadPlot(tenors,scurve.values[0],fwdcurve.values[0],xlabel='Tenors',ylabel='Rate  ( %)',y2label=str(fwd)+'y fwd swap cuve' ,y1label=y1label+ 'Swap curve', title=self.currency + ': Swap Curve comparison '+ y1label+' vs '+ str(fwd)+'y fwd'   , filename='korea/'+self.currency+'spotfwdYC.png',g2type='bar')
        return fwd_b
    
    def regexTrade(self,trade,spottrade=1):
        trades = re.split(' ',trade)
        if len(trades)==1:
            cur=trade[0:3].upper()
            fwd_yn=len(re.split('[f]',trade[3:]))
            fwd = re.split('[f]',trade[3:])[0]
            if fwd_yn==1:
                if spottrade==1:
                    fwd = ''
                if spottrade==0:
                    fwd ='0C'
            elif fwd_yn==2:
                if len(fwd)==1:
                    fwd = '0'+re.split('[f]',trade[3:])[0]
            tenors= list(filter(None,re.split('[s]',re.split('[f]',trade[3:])[fwd_yn-1])))
            tenors = [c+'y' for c in tenors]
            return (cur, fwd, tenors)
        else:
            return [self.regexTrade(t,spottrade=spottrade) for t in trades]
                
    def rd_prefix(self,fwd,curr):
        if self.fxmult.empty:
            self.bbgticker,self.fxmult=self.readTickers1() 

        if fwd in ['0C','0F','0I', '01']:
            spotprefix = self.fxmult[self.fxmult.index==curr]['spotprefix'][0]
            spot_fwd = ''
        else:
            spotprefix = self.fxmult[self.fxmult.index==curr]['fwdprefix'][0]
            spot_fwd = int(fwd)-1
            if spot_fwd<10:
                spotprefix = spotprefix + '0'+str(spot_fwd)
            else:
                spotprefix = spotprefix +str(spot_fwd)
        spot_f = str(spot_fwd)
        if len(spot_f)==1:
            spot_f = '0'+spot_f
        return (spotprefix,spot_f)

    def rd_trade(self,trade):
        if isinstance(trade,list):
            return [self.rd_trade(t) for t in trade]
        (curr, fwd, tenors) = trade
        if self.fxmult.empty:
            self.bbgticker,self.fxmult=self.readTickers1() 

        if fwd in ['0C','0F','0I', '01']:
            spot_fwd = ''
        else:
            spot_fwd = int(fwd)-1
        spot_f = str(spot_fwd)
        if len(spot_f)==1:
            spot_f = '0'+spot_f
        return (curr,spot_f,tenors)
        
    def getTradeTs(self,trade,start='20150101'):
        if self.fxmult.empty:
            self.bbgticker,self.fxmult=self.readTickers1() 
            
        if isinstance(trade, tuple):
            #print(trade)
            (curr, fwd, tenors) = trade
            trade = curr+fwd+''.join(tenors)
        else:    
            (curr, fwd, tenors) = self.regexTrade(trade)
        if fwd=='':
            prefix = self.fxmult[self.fxmult.index==curr]['spotprefix'][0]
            fflag = 0
        else:
            prefix = self.fxmult[self.fxmult.index==curr]['fwdprefix'][0]
            fflag = self.fxmult[self.fxmult.index==curr]['fwdflag'][0]
        #print(curr)
        if fwd =='':
            fwd=0
        self.genSwaptickers(curr, tenor_set=3, fwdflag=fflag,start = fwd,method='')
        self.populate(start=start)
        trade_data =self.getSpread(curr=curr,tenor=tenors)
        trade_data.rename(columns={ trade_data.columns[0]: trade }, inplace = True)
        return (curr, fwd, tenors, trade_data)

    def getCrossTradeTs(self,trades,mult=100,start='20150101'):
        if isinstance(trades, list):
            trade1 = trades[0]
            trade2 = trades[1]
        else:
            trades = self.regexTrade(trades)
            return self.getCrossTradeTs(trades,start=start)
        tmp1 = bbgData()
        tmp2 = bbgData()
        trade_data1 = tmp1.getTradeTs(trade1,start=start)[3]
        trade_data2 = tmp2.getTradeTs(trade2,start=start)[3]
        tradets= (trade_data2.iloc[:,0]-trade_data1.iloc[:,0])*mult
        tradets.name = trade_data1.columns[0]+trade_data2.columns[0]
        tradets = tradets.to_frame()
        return tradets
    
    def getTradeTsForAll(self,trade,ccylist=['KRW','USD','SGD','HKD','AUD', 'CNY','TWD', 'JPY'],start='20150101'):
        if isinstance(trade, tuple):
            (curr, fwd, tenors) = trade
        else:    
            (curr, fwd, tenors) = self.regexTrade(trade,spottrade=0)
        (cur_s, fwd_s, tenors_s, trade_data) = self.getTradeTs((ccylist[0],fwd,tenors),start=start)

        for cur in ccylist[1:]:
            tmp = bbgData()
            (cur_s, fwd_s, tenors_s, trade_data_s) = tmp.getTradeTs((cur,fwd,tenors),start=start)
            trade_data=trade_data.join(trade_data_s,how='outer')

        return (ccylist, fwd, tenors, trade_data)
        
    def getCrossTsForAll(self,trade,ccylist=['KRW','USD','SGD','HKD','AUD', 'CNY','TWD', 'JPY'],start='20150101'):
        if isinstance(trade, tuple):
            (curr, fwd, tenors) = trade
        else:    
            (curr, fwd, tenors) = self.regexTrade(trade,spottrade=0)
        (cur_s, fwd_s, tenors_s, trade_data) = self.getTradeTs((ccylist[0],fwd,tenors),start=start)

        for cur in ccylist[1:]:
            tmp = bbgData()
            (cur_s, fwd_s, tenors_s, trade_data_s) = tmp.getTradeTs((cur,fwd,tenors),start=start)
            trade_data=trade_data.join(trade_data_s,how='outer')

        return (ccylist, fwd, tenors, trade_data)


    def analyzeTrade(self,trade,periods=800, mset=3,ccylist=['KRW','USD','SGD','HKD','AUD', 'CNY','TWD', 'JPY'],start='20150101',folder='korea/'):
        if self.fxmult.empty:
            self.bbgticker,self.fxmult=self.readTickers1() 
        (curr, fwd, tenors) = self.regexTrade(trade,spottrade=0)
        prefix = self.fxmult[self.fxmult.index==curr]['fwdprefix'][0]
        fflag = self.fxmult[self.fxmult.index==curr]['fwdflag'][0]
        self.genSwaptickers(curr, tenor_set=mset, fwdflag=fflag,start = fwd,method='')
        self.populate(start=start)
        self.genpca(title_=curr+' '+fwd +'  Rates',count=periods,filename=curr+'PCAr')

        (spotprefix,spot_f) = self.rd_prefix(fwd,curr)
        rd_fflag = fflag
        if spot_f =='':
            rd_fflag=0
        #print(rd_fflag)
        rd_bbgData=self.compareFwdCurve(fwd=spot_f,fflag=rd_fflag)
        
        trade_data =self.getSpread(curr=curr,tenor=tenors)
        rd_trade_data =rd_bbgData.getSpread(curr=curr,tenor=tenors)
        rd_trade_name = curr+spot_f+'f'+''.join([c[:-1]+'s' for c in tenors])
        sw5= self.getSpread(curr=curr,tenor=['5y'])
        trade_data.rename(columns={ trade_data.columns[0]: trade }, inplace = True)
        rd_trade_data.rename(columns={ rd_trade_data.columns[0]: rd_trade_name }, inplace = True)   
        
        plotS.plotSwap(sw5.tail(periods),trade_data.tail(periods))
        
        plotS.spreadPlot(rd_trade_data.tail(periods).index,trade_data.tail(periods),rd_trade_data.tail(periods),xlabel='Date',ylabel='Rate  ( %)',y2label=rd_trade_name,y1label=trade, title=curr+' Trade Rolldown comparison '+ trade +' vs ' + rd_trade_name  , filename=folder+trade+'RollDown.png',g2type='bar',mult=1)
        tmp = bbgData()
        rd_tmp = bbgData()
        (ccylist, fwd, tenors, trade_data) = tmp.getTradeTsForAll(trade,ccylist=ccylist)
        #print(curr+spot_f)
        (ccylist, spot_f, tenors, rd_trade_data) = rd_tmp.getTradeTsForAll((curr,spot_f, tenors),ccylist=ccylist)
        
        plotS.plotRange(trade_data.tail(periods),count=periods,title='Cross Market: '+fwd+' fwd'+ ''.join(tenors) + ' Swap',filename=folder+trade+'crossMktTrade.png')
        
        crossmkttradelvl = trade_data.tail(1)
        crossmktrdtradelvl = rd_trade_data.tail(1)
        ccys= [c[0:3] for c in trade_data.columns.tolist()]
        plotS.spreadPlot(ccys,crossmkttradelvl.values[0],crossmktrdtradelvl.values[0],xlabel='Currencies',ylabel='Rate  ( %)',y2label=spot_f+'y fwd swap cuve' ,y1label=fwd+ 'Swap curve', title='Cross Market: '+fwd+' fwd vs '+ spot_f+' fwd '+ ''.join(tenors) + ' Swap Roll Down'   , filename=folder+trade+'crossMktTradeRD.png',g2type='bar',mult=1)
        
        return (trade_data, rd_trade_data)
    


    def analyzeCrossTrade(self,trade,periods=800, mset=3,ccylist=['KRW','USD','SGD','HKD','AUD', 'CNY','TWD', 'JPY'], offset=300,start='20150101',folder='korea/'):
        trades =self.regexTrade(trade,spottrade=0)
        rd_trades = self.rd_trade(trades)
        ts = self.getCrossTradeTs(trades,start=start)
        rd_ts = self.getCrossTradeTs(rd_trades,start=start)
        plotS.spreadPlot(rd_ts.tail(periods).index,ts.tail(periods),rd_ts.tail(periods),xlabel='Date',ylabel='Rate  ( %)',y2label=rd_ts.columns[0],y1label=ts.columns[0], title=' Trade Rolldown comparison '+ ts.columns[0] +' vs ' + rd_ts.columns[0]  , filename='korea/'+ts.columns[0]+'RollDown.png',g2type='bar',mult=1)

        tmp1 = bbgData()    
        (ccylist1, fwd1, tenors1, trade_data1) = tmp1.getTradeTsForAll(trades[0],ccylist=ccylist,start=start)
        tmp1.genpca1(trade_data1,title_='Cross Market: ',count=periods,type_='FX',filename='sw5')

        tmp1 = bbgData()    
        (rd_ccylist1, rd_fwd1, rd_tenors1, rd_trade_data1) = tmp1.getTradeTsForAll(rd_trades[0],ccylist=ccylist,start=start)

        if (trades[0][1] != trades[1][1]) or (trades[0][2] != trades[1][2]):
            tmp2 = bbgData()    
            (ccylist2, fwd2, tenors2, trade_data2) = tmp2.getTradeTsForAll(trades[1],ccylist=ccylist,start=start)
            tmp2.genpca1(trade_data2,title_='Cross Market: ',count=periods,type_='FX',filename='sw5')
            tmp2 = bbgData()    
            (rd_ccylist2, rd_fwd2, rd_tenors2, rd_trade_data2) = tmp2.getTradeTsForAll(rd_trades[1],ccylist=ccylist,start=start)

        curr  = trades[0][0]    
        suffix1 = trades[0][1]+''.join(trades[0][2])
        suffix2 = trades[1][1]+''.join(trades[1][2])
        rd_suffix1 = rd_trades[0][1]+''.join(rd_trades[0][2])
        rd_suffix2 = rd_trades[1][1]+''.join(rd_trades[1][2])

        if trades[0][0]==trades[1][0]:
            for t in trade_data1.columns.to_list():
                if curr != t[0:3]:
                    ts[t[0:3]+suffix1+'_'+suffix2] = (trade_data2.loc[:,t[0:3]+suffix2] - trade_data1.loc[:,t[0:3]+suffix1])*100
                    rd_ts[t[0:3]+rd_suffix1+'_'+rd_suffix2] = (rd_trade_data2.loc[:,t[0:3]+rd_suffix2] - rd_trade_data1.loc[:,t[0:3]+rd_suffix1])*100
            plotS.plotRange(ts.tail(periods),count=periods,title='Cross Market: '+suffix1+' vs'+ suffix2 + ' Swap',filename=folder+'crossMktTradeTS.png')
            ts_tail = ts.tail(1)
            rd_ts_tail = rd_ts.tail(1)
            ts_tail.columns = ts_tail.columns.str.slice(0,3)
            rd_ts_tail.columns = rd_ts_tail.columns.str.slice(0,3)
            plotS.spreadPlot(ts_tail.columns.to_list(),ts_tail.values[0],rd_ts_tail.values[0],xlabel='Currencies',ylabel='Spread  (bp)',y2label=rd_suffix1 +' vs '+ rd_suffix2 +' spread' ,y1label=suffix1+ ' vs ' +suffix2 +' Spread', title='Cross Market: '+suffix1+'  vs '+ suffix2+' Rolldown ', filename=folder+'crossMktTradeRD.png',g2type='bar',mult=1)

        else:
            for t in trade_data1.columns.to_list():
                if (curr != t[0:3]) and ( trade[1][0]!=t[0:3]):
                    ts[curr+suffix1+t[0:3]+suffix2] = (trade_data1.loc[:,t[0:3]+suffix2] - trade_data1.loc[:,curr+suffix1])*100
                    rd_ts[curr+rd_suffix1+t[0:3]+rd_suffix2] = (rd_trade_data1.loc[:,t[0:3]+rd_suffix2] - rd_trade_data1.loc[:,curr+rd_suffix1])*100
            ts.columns=[''.join(re.split('[^A-Z]',s)) for s in ts.columns.to_list()]                    
            rd_ts.columns=[''.join(re.split('[^A-Z]',s)) for s in rd_ts.columns.to_list()]
            plotS.plotRange(ts.tail(periods),count=periods,title='Cross Market: '+suffix1+' vs'+ suffix2 + ' Swap',filename=folder+'crossMktTradeTS.png',labelcount=6)
            ts_tail = ts.tail(1)
            rd_ts_tail = rd_ts.tail(1)
            plotS.spreadPlot(ts_tail.columns.to_list(),ts_tail.values[0],rd_ts_tail.values[0],xlabel='Currencies',ylabel='Spread  (bp)',y2label=rd_suffix1 +' vs '+ rd_suffix2 +' spread' ,y1label=suffix1+ ' vs ' +suffix2 +' Spread', title='Cross Market: '+suffix1+'  vs '+ suffix2+' Rolldown ', filename=folder+'crossMktTradeRD.png',g2type='bar',mult=1)
            
        return (ts, rd_ts)
    
    def analyze(self,trade,periods=800, mset=3,ccylist=['KRW','USD','SGD','HKD','AUD', 'CNY','TWD', 'JPY','NZD'],start='20150101',folder='korea/'):
        trades = re.split(' ',trade)
        if len(trades)==1:
            (ts,rd_ts)=self.analyzeTrade(trade,periods=periods, mset=mset,ccylist=ccylist,start=start,folder=folder)
        if len(trades)==2:
            (ts,rd_ts)=self.analyzeCrossTrade(trade,periods=periods, mset=mset,ccylist=ccylist,start=start,folder=folder)
        return (ts, rd_ts)
    
    def betaPlot(self,y1,x1,plotY=1,n=125,ylabel='',xlabel='',title='',betathreshold=-.6,file='korea/ustktb10beta.png'):
        y = y1.copy()
        x = x1.copy()
        
        x['const'] = 1
        xname = x.columns[0]
        model = RollingOLS(endog =y , exog=x[['const',xname]],window=n)
        rres = model.fit()
        y['beta']=rres.params.iloc[:,1]
        
        y1=y.iloc[:,0]
        y2=y.iloc[:,1]
        y1label = ylabel
        if ylabel =='':
            y1label=y.columns[0]
        y2label='Beta'
        if title=='':
            title= 'Beta and  '+ y1label
        xlabel='Date'
        fig, ax = plt.subplots(figsize=(9,6));
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax2 = ax.twinx();
        d = x.index
        ax.plot(d,y1,'g-');
        ax2.plot(d,y2,'b-');
        ax.set_xlabel(xlabel);
        plt.title(title,fontsize=18);
        ax.set_ylabel(y1label, color='g',fontsize=13)
        ax2.set_ylabel(y2label, color='b',fontsize=13)  
        plt.yticks(fontsize=14)
        yy = y[y.beta<betathreshold].index.to_list()
        for i in yy:
            plt.axvline(x=i,color='y',alpha=.25)
        plt.show()
        plt.savefig(file, bbox_inches='tight',dpi = 300)
   
        return (y,x);

    def beta(self,df,x,n=52,periods=500,freq='W',diff=1,diffperiod=1,mult=100, title=''):
        if freq != '':
            df = df.resample(freq).last()
        if diff!=0:
            if diff ==1:
                df = df.diff(diffperiod)*mult
            if diff == 2:
                df = df.pct_change(diffperiod)*mult
        cols = df.columns.to_list()
        cols.remove(x)
        df_x = df[[x]]
        df_x['const'] = 1
        df_beta = pd.DataFrame(index=df[x].index)
        #beta.index = df_x.index
        #c = pd.DataFrame()
        for t in cols:
            #print(t)
            model = RollingOLS(endog =df[t] , exog=df_x[['const',x]],window=n)
            rres = model.fit()
            df_beta[t] = rres.params.iloc[:,1]
        plotS.plotRange(df_beta.tail(periods),filename='korea/sw5_beta.png', title=title+' beta (agst '+x+') Z Score',count=periods)
        return df_beta


    def crossBeta(self,df1,df2,n=52,periods=500,freq='W',diff1=1,diff2=1,diffperiod=1,mult=100):
        if freq != '':
            df1 = df1.resample(freq).last()
            df2 = df2.resample(freq).last()
        if diff1!=0:
            if diff1 ==1:
                df1 = df1.diff(diffperiod)*mult
            if diff1 == 2:
                df1 = df1.pct_change(diffperiod)*mult
        if diff2!=0:
            if diff2 ==1:
                df2 = df2.diff(diffperiod)*mult
            if diff2 == 2:
                df2 = df2.pct_change(diffperiod)*mult
        ccylist = list(set(df1.columns) & set(df2.columns))
        df_beta = pd.DataFrame(index=df1.index)
        
        for c in ccylist:
            df_x = df2[[c]]
            df_y = df1[[c]]
            df_x['const'] = 1
            model = RollingOLS(endog =df_y , exog=df_x[['const',c]],window=n)
            rres = model.fit()
            df_beta[c] = rres.params.iloc[:,1]
        #plotS.plotRange(df_beta.tail(periods),filename='korea/sw5_beta.png', title='5y Swap beta Z Score',count=periods)
        return df_beta
        
        
    def regResidual(self,y1,x1,plotY=1,n=52,ylabel='',xlabel='',filename='ustktb10beta.png',y1hline=[],y2hline=[]):
        y = y1.copy()
        x = x1.copy()
        
        x['const'] = 1
        xname = x.columns[0]
        model = RollingOLS(endog =y , exog=x[['const',xname]],window=n)
        rres = model.fit()
        res=(y.iloc[:,0]-(x['const']*rres.params['const']+x[xname]*rres.params[xname]))*100        
        y1label = ylabel
        if ylabel =='':
            y1label=y.columns[0]
        y2label='Residual of regression'
        title= 'Residual and  '+ y1label
        plotS.twoAxisPlot(y.index,y1=y.iloc[:,0],y2=res,y1label=y1label,y2label=y2label,title=title,xlabel='Date',y1hline=y1hline,y2hline=y2hline,filename=filename)   
        return res;
    
    def freshTS(self,tickers='',currency='KRW',type_='CPIYOY',flds=['PX_LAST'],start='20150101',end='',freq='DAILY',maxDataPoints=5000,fill=0,verbose=0):
        if (tickers == '')  or (len(tickers)==0):
            if self.bbgticker.empty:
                if len(self.secs)==0 :
                    self.bbgticker,self.fxmult=self.readTickers1()            

            if currency=='':
                self.secs=self.bbgticker[self.bbgticker.Type==type_].bbgticker.values.tolist()
            elif type_=='':
                self.secs=self.bbgticker[self.bbgticker.Code==currency].bbgticker.values.tolist()
            else:    
                self.secs=self.bbgticker[self.bbgticker.index==currency+type_].bbgticker.values.tolist()
        else:
            self.secs=[t.upper() for t in tickers]
            #print(self.secs)
            #print("came here 3")
        
        self.populate(flds=flds,start=start,end=end,freq=freq,maxDataPoints=maxDataPoints,fill=fill,verbose=verbose)
        return self.dfbbg
    
    def he_quad(self,curr='KRW',diff=1):
        tmp = bbgData()
        cpi_ts = tmp.freshTS(currency=curr,type_='CPIYOY',fill=0)
        gdp_ts = tmp.freshTS(currency=curr,type_='GDPYOY',fill=0)
        gdp_label = gdp_ts.iloc[-1,0]
        cpi_label = cpi_ts.iloc[-1,0]
        
        if diff == 1:
            cpi_ts = cpi_ts.diff(1)
            gdp_ts = gdp_ts.diff(1)
        gdp_ts = gdp_ts[gdp_ts.index<=datetime.today()]
        cpi_ts = cpi_ts[cpi_ts.index<=datetime.today()]
        
        fig, ax = plt.subplots(figsize=(8,8))         # Sample figsize in inches4
        plt.plot(cpi_ts.tail(10),gdp_ts.tail(10),'<-b',markersize=15, linewidth=2, label ='CPI: '+'%.4f'%cpi_label+ "; "+' GDP: '+'%.4f'%gdp_label)
        plt.plot(cpi_ts.tail(1),gdp_ts.tail(1),'<-r',markersize=15, linewidth=2)
        ax.tick_params(axis = 'both',  labelsize = 24)
        plt.axvline(x=0,linewidth=5)
        plt.axhline(y=0,linewidth=5)
        ax.set_xlabel("CPI Change", size=25)
        ax.set_ylabel("GDP Change", size=25)
        plt.legend(fontsize='large')
        plt.xticks(rotation=45,fontsize=15)

        plt.title(curr+': GDP vs CPI Change ',fontsize=25)

        return
    
    def renameColtoCurr(self,df,suffix=''):
        df.columns=self.bbgticker[self.bbgticker.bbgticker.isin(df.columns.to_list())].Code.to_list()
        return df
    
    def getTicker(self,trade):
        if type(trade)==list:
            return [self.getTicker(t) for t in trade]
        if type(trade)!=tuple:
            trade = self.regexTrade(trade.strip())
            return self.getTicker(trade)
        curr = trade[0]
        start = trade[1]
        tickers = []
        #print(trade)
        if self.fxmult.empty:
            self.bbgticker,self.fxmult=self.readTickers1() 
        if start == '':
            prefix =self.fxmult[self.fxmult.index==curr]['spotprefix'][0]
            suffix =self.fxmult[self.fxmult.index==curr]['spotsuffix'][0]
        else:
            prefix =self.fxmult[self.fxmult.index==curr]['fwdprefix'][0]
            suffix =self.fxmult[self.fxmult.index==curr]['fwdsuffix'][0]
        fwdflag = self.fxmult[self.fxmult.index==curr]['fwdflag'][0]
        for t in trade[2]:
            tenor=t[:-1]
            if start == '':
                tickers.append(prefix+tenor+suffix)            
            else:   
                if fwdflag == 0:
                    tickers.append(prefix+start+tenor+suffix)            
                if fwdflag == 1:
                    if len(tenor)==1:
                        tenor = '0'+tenor
                    tickers.append(prefix+start+tenor+suffix)
        return tickers

    def flatten(self,list_of_lists):
        if len(list_of_lists) == 0:
            return list_of_lists
        if isinstance(list_of_lists[0], list):
            return self.flatten(list_of_lists[0]) + self.flatten(list_of_lists[1:])
        return list_of_lists[:1] + self.flatten(list_of_lists[1:])
    
    def getTickerTs(self,tickers):
        #print(tickers)
        if type(tickers)!=list:
            return self.dfbbg.loc[:,tickers]
        if len(tickers)==1:            
            return self.getTickerTs(tickers[0])
        if len(tickers)==2:
            return self.getTickerTs(tickers[1])-self.getTickerTs(tickers[0])
        if len(tickers)==3:
                    return 2*self.getTickerTs(tickers[1])-self.getTickerTs(tickers[0]) - self.getTickerTs(tickers[2])
        
        