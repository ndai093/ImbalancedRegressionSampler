from PhiRelevance.PhiUtils1 import phiControl,phi

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class RandomUnderSamplerRegression:
    """
    Class RandomUnderSamplerRegression takes arguments as follows:
        data - Pandas data frame with target value as last column
        method - "auto"("extremes") as default,"range"
        extrType - "high", "both" as default, "low"
        thr_rel - user defined relevance threadhold between 0 to 1, all the target values with relevance below
                  the threshold are candicates to be undersampled
        controlPts - list of control points formatted as [y1, phi(y1), phi'(y1), y2, phi(y2), phi'(y2)], where
                     y1: target value; phi(y1): relevane value of y1; phi'(y1): derivative of phi(y1), etc.
        c_perc - undersampling percentage should be applied in each bump with uninteresting values, 
                 string type as defined below,
                 "balance" - will try to distribute the examples evently across the existing bumps 
                 "extreme" - invert existing frequency of interesting/uninteresting set
                 "<percentage>" - percentage undersampling set should follow.

    """
    def __init__(self, data, method='auto', extrType='both', thr_rel=1.0, controlPts=[], c_perc="balance"):
        
        self.data = data;
        
        self.method = 'extremes' if method in ['extremes', 'auto'] else 'range'
        
        if method == 'extremes':
            self.extrType = extrType if extrType in ['high', 'both', 'low'] else 'both'
        else:
            self.extrType =''

        self.thr_rel = thr_rel
        
        if method == 'extremes':
            self.controlPts = []
        else:
            self.controlPts = controlPts
        
        self.c_perc = c_perc if c_perc in ["balance", "extreme"] else c_perc
        
        self.coef = 1.5

    def resample(self):

        #retrieve target(last column) from DataFrame
        y = self.data.iloc[:,-1]

        #generate control ptrs 
        if self.method == 'extremes':
            controlPts, npts = phiControl(y, extrType=self.extrType)
        else:
            controlPts, npts = phiControl(y, 'range', extrType="", controlPts=self.controlPts, coef=-1.0)

        #calculate relevance value
        yPhi, ydPhi, yddPhi = phi(y, controlPts, npts, self.method)

        #append column 'yPhi'
        self.data['yPhi'] = yPhi

        data1 = self.data.sort_values(by=['Tgt'])
        data1.insert(0, 'index', range(0, len(data1)))
        data1.set_index('index', inplace=True)
        #interesting set
        interesting_set = data1[data1.yPhi >= self.thr_rel]
        total = len(data1)
        #uninteresting set
        bumps = __calc_bumps(self, data1)
        if self.c_perc == 'balance':
            resampled = __process_balance(self, bumps, interesting_set)
        elif self.c_perc == 'extreme':
            resampled = __process_extreme(self, bumps, total, interesting_set)
        else:
            resampled = __process_percentage(self, bumps)

        #clean up resampled set and return
        resampled.drop(columns=['index','yPhi'])
        resampled.sort_values(by=resampled.columns.to_list()[0])
        return resampled

    def __process_balance(self, bumps, interesting_set):
        resample_size = interesting_set.count // len(bumps)
        resampled_sets = []
        for s in bumps:
            resampled_sets.append(s.sample(n=resample_size))
        resampled_sets.append(interesting_set)
        result = pd.concat(resampled_sets)
        return result

    def __process_extreme(self, bumps, total, interesting_set):
        ratio = float(total - interesting_set.count) / interesting_set.count
        resample_size = (interesting_set.count * (1/ratio))/len(bumps)
        resample_size_int = int(resample_size)
        for s in bumps:
            resampled_sets.append(s.sample(n=resample_size_int))
        resampled_sets.append(interesting_set)
        result = pd.concat(resampled_sets)
        return result

    def __process_percentage(self, bumps):
        percentage = float(self.c_perc)
        resampled_sets = []
        for s in bumps:
            resampled_sets.append(s.sample(frac=percentage))
        result = pd.concat(resampled_sets)
        return result

    def __calc_bumps(self, df):

        thr_rel = self.thr_rel
        less_than_thr_rel = True if df.loc[0,'yPhi'] < thr_rel else False
        begin = 0
        bumps = []

        for idx, row in df.iterrows():
            if less_than_thr_rel and row['yPhi'] >= thr_rel:
                bumps.append(df.iloc[begin:idx,:])
                less_than_thr_rel = False
            elif not less_than_thr_rel and row['yPhi'] < thr_rel:
                begin = idx
                less_than_thr_rel = True

        if less_than_thr_rel and begin != len(df):
            bumps.append(df.iloc[begin:len(df),:])

        return bumps
