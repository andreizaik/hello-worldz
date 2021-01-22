#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:30:05 2020

@author: andreiscudder
"""
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


class ConstructFromDict(object):
    """ CounstructFromDict
        a simpe mix-in example to provide
        generic constructor from *dictionary, **key_word pairs
    """
    def __init__(self, *dictargs, **kwargs):
        """ Initialize the object from an arbitrary dict
        """
        for dictionary in dictargs:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
            
    @classmethod
    def Sample(cls, *dicts, **kwargs):
        """ Generate an intialized sample object
        """
        return cls(cls.sample, *dicts, **kwargs)
    
    
class Bond(ConstructFromDict):
    """
    A sample bond class
    """
    usecont = True
    sample = dict(
        notional = 1, 
        price = 100, 
        coupon = .05, 
        coupon_freq = 2, 
        maturity = 10, 
        rate = 0.01,
        maturitydate = datetime.date(2030,1,1),
        face = 100,
        spread = 0
        )
    
    def __init__(self, *dictargs, **kwargs):
        
        # get cashflows
        super().__init__(*dictargs, **kwargs)
        
        self.ncf = self.maturity * self.coupon_freq
        self.tau =  self.maturity * (np.arange(self.ncf) + 1)/self.ncf
        self.cashflows = np.ones(self.ncf) * self.notional * self.face * self.coupon/self.coupon_freq
        self.cashflows[-1] += self.face
    
        self.calc()
    
    def discount(self, tau = 0, rate = .02, usecont = False):
        """

        Parameters
        ----------
        tau : TYPE, optional
            DESCRIPTION. The default is 0.
        rate : TYPE, optional
            DESCRIPTION. The default is .02.
        usecont : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        res : TYPE
            DESCRIPTION.

        """
        if usecont:
            res = np.exp(-tau*rate)
        else:
            res = (1 + rate)**-tau
        
        return res
    
    @staticmethod
    def getRc(rate, m):
        """
        Convert m-annual rate to continous rate
        """
        return m * np.log(1 + rate/m)
    
    def calc(self, price = None, rate = None):
        
        # discount at benchmark rate or at full curve
        df = self.discount(tau = self.tau, rate = rate or self.rate)
        self.dcf = self.cashflows * df
        self.getYTM()
        
        self.ytmdf = self.discount(self.tau, self.ytm)
        self.ytmdcf = self.ytmdf  * self.cashflows
        
        self.cmb = pd.DataFrame(dict(tau = self.tau, 
                                     cf = self.cashflows, 
                                     df = df, 
                                     dcf = self.dcf,
                                     ytmdf = self.ytmdf, 
                                     ytmdcf = self.ytmdcf))
        
  
    
    def getPrice(self, x = 0):
        
        df = self.discount(tau = np.arange(self.ncf) + 1, rate = x / self.coupon_freq)
        dcf = self.cashflows * df
        return np.sum(dcf)
        
    def getYTM(self, price = None):
        
        price = price or self.price
        res = minimize_scalar(lambda x:abs(price - self.getPrice(x)))
        self.ytm = res.x
        return res
     
    def getDuration(self):
        return sum(self.ytmdcf * self.tau) / sum(self.ytmdcf)
    
    def plotPrice(self):
        pass
        
        

def getPlot(dfprices):
    
    fig = px.line(dfprices, x = dfprices.date, 
                  y = dfprices.px, color = dfprices.sym, 
                  facet_row = dfprices.Type, height=1400)
    
    fig.update_yaxes(matches=None)
    fig.update_layout(
        title="Yahoo Data",
        font=dict(
            family="Arial",
            size=10,
            color="#7f7f7f"
            )
    )
    
    return fig

def getSubPlots(dfprices):
    
    n = len(dfprices.Type.unique())
    fig = make_subplots(rows=n, 
                        subplot_titles=dfprices.Type.unique(),
                        horizontal_spacing = 0.05,
                        vertical_spacing = 0.01)
    
    # fig.add_trace(
    #     go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
    #     row=1, col=1
    # )
    
    # fig.add_trace(
    #     go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
    #     row=1, col=2
    # )
    
    for (sublpot, data) in enumerate(dfprices.groupby('Type')):
        (type_name, type_data) = data
        
        for sym, sym_data in type_data.groupby('sym'):
            tmp = go.Scatter(x = sym_data.date, y = sym_data.px, name=sym)
            #tmp = px.line(j, x = j.date, y = j.px, color = j.sym, height=300)
            
            fig.add_trace(tmp, row = sublpot + 1, col = 1)
    
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=11)
    
    fig.update_layout(height=1800, 
                      width=1000, 
                      showlegend=False,
                      title_text="Products",
                      font=dict(
                            family="Arial",
                            size=8,
                            color="#7f7f7f"
                        ))
    return fig