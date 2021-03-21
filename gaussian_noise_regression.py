from PhiRelevance.PhiUtils1 import phiControl,phi

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class GaussianNoiseRegression:
    """
    Class GaussianNoiseRegression takes arguments as follows:
        data - Pandas data frame with target value as last column, rest columns should be feature/attributes
        method - "auto"(default, also called "extremes"),"range"
        extrType - "high", "both"(default), "low"
        thr_rel - user defined relevance threadhold between 0 to 1, all the target values with relevance above
                  the threshold are candicates to be oversampled
        controlPts - list of control points formatted as [y1, phi(y1), phi'(y1), y2, phi(y2), phi'(y2)], where
                     y1: target value; phi(y1): relevane value of y1; phi'(y1): derivative of phi(y1), etc.
        c_perc - under and over sampling strategy, Gaussian noise in this implementation should be applied in each bump with oversampling(interesting) sets, 
                 possible types are defined below,
                 "balance" - will try to distribute the examples evenly across the existing bumps 
                 "extreme" - invert existing frequency of interesting/uninteresting set
                 <percentage> - A list of percentage values with the following formats,
                                for any percentage value < 1, there should be either 1 percentage value applies to all bumps of undersampling set,
                                or multiple percentage values mapping to each bump of undersampling set;
                                for any percentage value > 1, there should be either 1 percentage value applies to all bumps of oversampling set
                                or multiple percentage values mapping to each bump of oversampling set;
                
    """
    def __init__(self, data, method='auto', extrType='both', thr_rel=1.0, controlPts=[], c_perc="balance"):
        
        self.data = data;
        
        self.method = 'extremes' if method in ['extremes', 'auto'] else 'range'
        
        if self.method == 'extremes':
            if extrType in ['high','low','both']:
                self.extrType = extrType
            else:
                self.extrType = 'both'
        else:
            self.extrType =''

        self.thr_rel = thr_rel
        
        if method == 'extremes':
            self.controlPts = []
        else:
            self.controlPts = controlPts
        
        self.c_perc_undersampling = []
        self.c_perc_oversampling = []
        if str == type(c_perc):
            self.c_perc = c_perc if c_perc in ["balance", "extreme"] else c_perc
        elif list == type(c_perc):
            self.c_perc = 'percentage list'
            self.processCPerc(c_perc)
        
        self.coef = 1.5

    def processCPerc(self, c_perc):
        for x in c_perc:
            if x < 1:
                self.c_perc_undersampling.append(x)
            elif x > 1:
                self.c_perc_oversampling.append(x)
            else:
                print('c_perc valie in list should not be 1!')
        print('c_perc_undersampling:')
        print(self.c_perc_undersampling)    
        print('c_perc_oversampling')
        print(self.c_perc_oversampling)

    def getMethod(self):
        return self.method

    def getData(self):
        return self.data

    def getExtrType(self):
        return self.extrType

    def getThrRel(self):
        return self.thr_rel

    def getControlPtr(self):
        return self.controlPts

    def getCPerc(self):
        if self.c_perc in ['balance', 'extreme']:
            return self.c_perc
        else:
            return self.c_perc_undersampling, self.c_perc_oversampling

    def resample(self):

        yPhi, ydPhi, yddPhi = self.calc_rel_values()

        data1 = self.preprocess_data(yPhi)
        #interesting set
        self.interesting_set = self.get_interesting_set(data1)
        #uninteresting set
        self.uninteresting_set = self.get_uninteresting_set(data1)
        #calculate bumps
        self.bumps_undersampling, self.bumps_oversampling = self.calc_bumps(data1)

        if self.c_perc == 'percentage list':
            resampled = self.process_percentage()

        #clean up resampled set and return
        self.postprocess_data(resampled)
        return resampled

    def postprocess_data(self, resampled):
        resampled.drop('yPhi',axis=1,inplace=True )
        resampled.sort_index(inplace=True)
        return resampled

    def preprocess_data(self, yPhi):
        #append column 'yPhi'
        data1 = self.data
        data1['yPhi'] = yPhi
        data1 = self.data.sort_values(by=['Tgt'])
        return data1
        
    def get_uninteresting_set(self, data):
        uninteresting_set = data[data.yPhi < self.thr_rel]
        return uninteresting_set

    def get_interesting_set(self, data):
        interesting_set = data[data.yPhi >= self.thr_rel]
        return interesting_set

    def calc_rel_values(self):
        #retrieve target(last column) from DataFrame
        y = self.data.iloc[:,-1]

        #generate control ptrs 
        if self.method == 'extremes':
            controlPts, npts = phiControl(y, extrType=self.extrType)
        else:
            controlPts, npts = phiControl(y, 'range', extrType="", controlPts=self.controlPts)

        #calculate relevance value
        yPhi, ydPhi, yddPhi = phi(y, controlPts, npts, self.method)
        return yPhi, ydPhi, yddPhi

    def calc_bumps(self, df):

        thr_rel = self.thr_rel
        less_than_thr_rel = True if df.loc[0,'yPhi'] < thr_rel else False
        bumps_oversampling = []
        bumps_undersampling = []
        bumps_oversampling_df = pd.DataFrame(columns = df.columns)       
        bumps_undersampling_df = pd.DataFrame(columns = df.columns)

        for idx, row in df.iterrows():
            if less_than_thr_rel and (row['yPhi'] < thr_rel):
                bumps_undersampling_df = bumps_undersampling_df.append(row)
            elif less_than_thr_rel and row['yPhi'] >= thr_rel:
                bumps_undersampling.append(bumps_undersampling_df)
                bumps_undersampling_df = pd.DataFrame(columns = df.columns)
                bumps_oversampling_df = bumps_oversampling_df.append(row)
                less_than_thr_rel = False
            elif (not less_than_thr_rel) and (row['yPhi'] >= thr_rel):
                bumps_oversampling_df = bumps_oversampling_df.append(row)
            elif (not less_than_thr_rel) and (row['yPhi'] < thr_rel):
                bumps_oversampling.append(bumps_oversampling_df)
                bumps_oversampling_df = pd.DataFrame(columns = df.columns)
                bumps_undersampling_df = bumps_undersampling_df.append(row)
                less_than_thr_rel = True

        if less_than_thr_rel and (df.iloc[-1,:]['yPhi'] < thr_rel):
            bumps_undersampling.append(bumps_undersampling_df)
        elif not less_than_thr_rel and (df.iloc[-1,:]['yPhi'] >= thr_rel):
            bumps_oversampling.append(bumps_oversampling_df)

        return bumps_undersampling, bumps_oversampling        

    def process_percentage(self):
        pass