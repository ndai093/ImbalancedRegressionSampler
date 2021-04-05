from PhiRelevance.PhiUtils1 import phiControl,phi

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class RandomUnderSamplerRegression:
    """
    Class RandomUnderSamplerRegression takes arguments as follows:
        data - Pandas data frame with target value as last column; if read from .csv, recommend to use 'index_col=0'
        method - "auto"("extremes") as default,"range"
        extrType - "high", "both" as default, "low"
        thr_rel - user defined relevance threadhold between 0 to 1, all the target values with relevance below
                  the threshold are candicates to be undersampled
        controlPts - list of control points formatted as [y1, phi(y1), phi'(y1), y2, phi(y2), phi'(y2)], where
                     y1: target value; phi(y1): relevane value of y1; phi'(y1): derivative of phi(y1), etc.
        c_perc - undersampling percentage should be applied in each bump with uninteresting values, 
                 possible types are defined below,
                 "balance" - will try to distribute the examples evenly across the existing bumps 
                 "extreme" - invert existing frequency of interesting/uninteresting set
                 <percentage> - A list of percentage values with either one value apply to all bumps of undersampling set
                                or multiple percentage values mapping to each bump of undersampling set

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
        
        if str == type(c_perc):
            self.c_perc = c_perc if c_perc in ["balance", "extreme"] else c_perc
        elif list == type(c_perc):
            self.c_perc = c_perc
        
        self.coef = 1.5
    
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
        return self.c_perc

    def resample(self):

        yPhi, ydPhi, yddPhi = self.calc_rel_values()

        data1 = self.preprocess_data(yPhi)
        #interesting set
        interesting_set = self.get_interesting_set(data1)
        #uninteresting set
        bumps_undersampling, bumps_interesting = self.calc_bumps(data1)

        if self.c_perc == 'balance':
            resampled = self.process_balance(bumps_undersampling, interesting_set)
        elif self.c_perc == 'extreme':
            resampled = self.process_extreme(bumps_undersampling, bumps_interesting, interesting_set)
        elif isinstance(self.c_perc, list):
            resampled = self.process_percentage(bumps_undersampling, interesting_set)

        #clean up resampled set and return
        self.postprocess_data(resampled)
        return resampled

    def postprocess_data(self, resampled):
        self.data.drop('yPhi',axis=1,inplace=True )
        resampled.drop('yPhi',axis=1,inplace=True )
        resampled.sort_index(inplace=True)
        return resampled

    def preprocess_data(self, yPhi):
        #append column 'yPhi'
        data1 = self.data
        data1['yPhi'] = yPhi
        data1 = self.data.sort_values(by=['Tgt'])
        return data1
        
    def get_interesting_set(self, data):
        interesting_set = data[data.yPhi >= self.thr_rel]
        return interesting_set
    
    def get_undersampling_set(self, data):
        undersampleing_set = data[data.yPhi < self.thr_rel]
        return undersampleing_set     
        
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

    def process_balance(self, bumps_undersampling, interesting_set):
        resample_size = round(len(interesting_set) / len(bumps_undersampling))
        #print('process_balance(): resample_size per bump='+str(resample_size))
        resampled_sets = []
        for s in bumps_undersampling:
            resampled_sets.append(s.sample(n=resample_size))
        #includes interesting set
        resampled_sets.append(interesting_set)
        result = pd.concat(resampled_sets)
        return result

    def process_extreme(self, bumps_undersampling, bumps_interesting, interesting_set):
        
        #print('process_extreme(): size of bumps_undersampling='+str(len(bumps_undersampling)))
        #print('process_extreme(): size of bumps_interesting='+str(len(bumps_interesting)))
        #print('process_extreme(): size of interesting_set='+str(len(interesting_set)))
        resampled_sets = []
        #calculate average cnt
        len_interesting_set = len(interesting_set)
        len_total = len(self.data)
        #print('process_extreme(): size of total_set='+str(len_total))
        average_cnt_interesting_set = len_interesting_set/len(bumps_interesting)
        #print('process_extreme(): average_cnt_interesting_set='+str(average_cnt_interesting_set))
        resample_size = (average_cnt_interesting_set**2.0)/(len_total-len_interesting_set)
        #print('process_extreme(): resample_size='+str(resample_size))
        resample_size_per_bump = round(resample_size / len(bumps_undersampling))
        #print('process_extreme(): resample_size_per_bump='+str(resample_size_per_bump))

        for s in bumps_undersampling:
            resampled_sets.append(s.sample(n = resample_size_per_bump))
        #includes interesting set       
        resampled_sets.append(interesting_set)
        result = pd.concat(resampled_sets)
        return result

    def process_percentage(self, bumps_undersampling, interesting_set):
        #make sure all percentage values are float values and <= 1.0
        for c in self.c_perc:
            if (not isinstance(c, float)) or (c>1.0):
                print('c_perc must be list of float number <= 1.0')
                return[]
        #make sure c_perc values matches bumps
        resampled_sets = []
        if (len(bumps_undersampling) != len(self.c_perc)) and (len(self.c_perc) != 1):
            print('c_perc value list must have either one value or values equal to number of bumps')
            return []
        elif len(self.c_perc) == 1: 
            undersample_perc = self.c_perc[0]
            #print('len(self.c_perc) == 1')
            #print('process_percentage(): undersample_perc='+str(undersample_perc))
            for s in bumps_undersampling:
                #print('process_percentage(): bump size='+str(len(s)))
                resample_size = round(len(s)*undersample_perc)
                #print('process_percentage(): resample_size='+str(resample_size))
                resampled_sets.append(s.sample(n = resample_size))
            #adding interesting set
            resampled_sets.append(interesting_set)
            result = pd.concat(resampled_sets)
        else:
            for i in range(len(bumps_undersampling)):
                #print('len(self.c_perc) > 1 loop i='+str(i))
                undersample_perc = self.c_perc[i]
                #print('process_percentage(): undersample_perc='+str(undersample_perc))
                resample_size = round(len(bumps_undersampling[i])*undersample_perc)
                #print('process_percentage(): resample_size='+str(resample_size))
                resampled_sets.append(bumps_undersampling[i].sample(n = resample_size))
            #adding interesting set
            resampled_sets.append(interesting_set)
            result = pd.concat(resampled_sets)
        return result

    def calc_bumps(self, df):

        thr_rel = self.thr_rel
        less_than_thr_rel = True if df.loc[0,'yPhi'] < thr_rel else False
        bumps_undersampling = []
        bumps_interesting = []
        bumps_undersampling_df = pd.DataFrame(columns = df.columns)       
        bumps_interesting_df = pd.DataFrame(columns = df.columns)

        for idx, row in df.iterrows():
            if less_than_thr_rel and (row['yPhi'] < thr_rel):
                bumps_undersampling_df = bumps_undersampling_df.append(row)
            elif less_than_thr_rel and row['yPhi'] >= thr_rel:
                bumps_undersampling.append(bumps_undersampling_df)
                bumps_undersampling_df = pd.DataFrame(columns = df.columns)
                bumps_interesting_df = bumps_interesting_df.append(row)
                less_than_thr_rel = False
            elif (not less_than_thr_rel) and (row['yPhi'] >= thr_rel):
                bumps_interesting_df = bumps_interesting_df.append(row)
            elif (not less_than_thr_rel) and (row['yPhi'] < thr_rel):
                bumps_interesting.append(bumps_interesting_df)
                bumps_interesting_df = pd.DataFrame(columns = df.columns)
                bumps_undersampling_df = bumps_undersampling_df.append(row)
                less_than_thr_rel = True

        if less_than_thr_rel and (df.iloc[-1,:]['yPhi'] < thr_rel):
            bumps_undersampling.append(bumps_undersampling_df)
        elif not less_than_thr_rel and (df.iloc[-1,:]['yPhi'] >= thr_rel):
            bumps_interesting.append(bumps_interesting_df)

        return bumps_undersampling, bumps_interesting
