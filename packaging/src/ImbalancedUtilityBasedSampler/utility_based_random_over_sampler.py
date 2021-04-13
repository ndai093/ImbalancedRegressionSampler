from PhiRelevance.PhiUtils import phiControl,phi

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class UtilityBasedRandomOverSampler:
    """
    Class UtilityBasedRandomOverSampler takes arguments as follows:
        data - Pandas data frame with target value as last column
        method - "auto"(default, also called "extremes"),"range"
        extrType - "high", "both"(default), "low"
        thr_rel - user defined relevance threadhold between 0 to 1, all the target values with relevance above
                  the threshold are candicates to be oversampled
        controlPts - list of control points formatted as [y1, phi(y1), phi'(y1), y2, phi(y2), phi'(y2)], where
                     y1: target value; phi(y1): relevane value of y1; phi'(y1): derivative of phi(y1), etc.
        c_perc - oversampling strategy should be applied in each bump with interesting sets, 
                 possible types are defined below,
                 "balance" - will try to distribute the examples evenly across the existing bumps 
                 "extreme" - invert existing frequency of interesting/uninteresting set
                 <percentage> - A list of percentage values with either one value apply to all bumps of oversampling set
                                or multiple percentage values mapping to each bump of oversampling set
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

        yPhi, ydPhi, yddPhi = self.__calc_rel_values()

        data1 = self.__preprocess_data(yPhi)
        #interesting set
        uninteresting_set = self.get_uninteresting_set(data1)
        #uninteresting set
        bumps_uninteresting, bumps_oversampling = self.__calc_bumps(data1)

        if self.c_perc == 'balance':
            resampled = self.__process_balance(bumps_oversampling, uninteresting_set)
        elif self.c_perc == 'extreme':
            resampled = self.__process_extreme(bumps_oversampling, bumps_uninteresting, uninteresting_set)
        elif isinstance(self.c_perc, list):
            resampled = self.__process_percentage(bumps_oversampling, uninteresting_set)

        #clean up resampled set and return
        self.__postprocess_data(resampled)
        return resampled

    def __postprocess_data(self, resampled):
        resampled.drop('yPhi',axis=1,inplace=True )
        resampled.sort_index(inplace=True)
        return resampled

    def __preprocess_data(self, yPhi):
        #append column 'yPhi'
        data1 = self.data
        data1['yPhi'] = yPhi
        data1 = self.data.sort_values(by=['Tgt'])
        return data1
        
    def get_uninteresting_set(self, data):
        uninteresting_set = data[data.yPhi < self.thr_rel]
        return uninteresting_set

    def get_oversampling_set(self, data):
        oversampling_set = data[data.yPhi >= self.thr_rel]
        return oversampling_set

    def __calc_rel_values(self):
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

    def __calc_bumps(self, df):

        thr_rel = self.thr_rel
        less_than_thr_rel = True if df.loc[0,'yPhi'] < thr_rel else False
        bumps_oversampling = []
        bumps_uninteresting = []
        bumps_oversampling_df = pd.DataFrame(columns = df.columns)       
        bumps_uninteresting_df = pd.DataFrame(columns = df.columns)

        for idx, row in df.iterrows():
            if less_than_thr_rel and (row['yPhi'] < thr_rel):
                bumps_uninteresting_df = bumps_uninteresting_df.append(row)
            elif less_than_thr_rel and row['yPhi'] >= thr_rel:
                bumps_uninteresting.append(bumps_uninteresting_df)
                bumps_uninteresting_df = pd.DataFrame(columns = df.columns)
                bumps_oversampling_df = bumps_oversampling_df.append(row)
                less_than_thr_rel = False
            elif (not less_than_thr_rel) and (row['yPhi'] >= thr_rel):
                bumps_oversampling_df = bumps_oversampling_df.append(row)
            elif (not less_than_thr_rel) and (row['yPhi'] < thr_rel):
                bumps_oversampling.append(bumps_oversampling_df)
                bumps_oversampling_df = pd.DataFrame(columns = df.columns)
                bumps_uninteresting_df = bumps_uninteresting_df.append(row)
                less_than_thr_rel = True

        if less_than_thr_rel and (df.iloc[-1,:]['yPhi'] < thr_rel):
            bumps_uninteresting.append(bumps_uninteresting_df)
        elif not less_than_thr_rel and (df.iloc[-1,:]['yPhi'] >= thr_rel):
            bumps_oversampling.append(bumps_oversampling_df)

        return bumps_uninteresting, bumps_oversampling        

    def __process_balance(self, bumps_oversampling, uninteresting_set):
        resample_size = round(len(uninteresting_set) / len(bumps_oversampling))
        #print('process_balance(): resample_size per bump='+str(resample_size))
        resampled_sets = []
        for s in bumps_oversampling:
            resampled_sets.append(s.sample(n=resample_size, replace=True))
        #includes uninteresting set
        resampled_sets.append(uninteresting_set)
        result = pd.concat(resampled_sets)
        return result        

    def __process_extreme(self, bumps_oversampling, bumps_uninteresting, uninteresting_set):
        
        #print('process_extreme(): size of bumps_oversampling='+str(len(bumps_oversampling)))
        #print('process_extreme(): size of bumps_uninteresting='+str(len(bumps_uninteresting)))
        #print('process_extreme(): size of uninteresting_set='+str(len(uninteresting_set)))
        resampled_sets = []
        #calculate average cnt
        len_uninteresting_set = len(uninteresting_set)
        len_total = len(self.data)
        #print('process_extreme(): size of total_set='+str(len_total))
        average_cnt_uninteresting_set = len_uninteresting_set/len(bumps_uninteresting)
        #print('process_extreme(): average_cnt_uninteresting_set='+str(average_cnt_uninteresting_set))
        resample_size = (average_cnt_uninteresting_set**2.0)/(len_total-len_uninteresting_set)
        #print('process_extreme(): resample_size='+str(resample_size))
        resample_size_per_bump = round(resample_size / len(bumps_oversampling))
        #print('process_extreme(): resample_size_per_bump='+str(resample_size_per_bump))

        for s in bumps_oversampling:
            resampled_sets.append(s.sample(n = resample_size_per_bump, replace=True))
        #includes interesting set       
        resampled_sets.append(uninteresting_set)
        result = pd.concat(resampled_sets)
        return result        

    def __process_percentage(self, bumps_oversampling, uninteresting_set):
        #make sure all percentage values are float values and > 1.0
        for c in self.c_perc:
            if (not isinstance(c, float)) or (c<1.0):
                print('c_perc must be list of float number >= 1.0')
                return[]
        #make sure c_perc values matches bumps
        resampled_sets = []
        if (len(bumps_oversampling) != len(self.c_perc)) and (len(self.c_perc) != 1):
            print('c_perc value list must have either one value or values equal to number of bumps')
            return []
        elif len(self.c_perc) == 1: 
            oversampling_ratio = self.c_perc[0]
            #print('len(self.c_perc) == 1')
            #print('process_percentage(): oversampling_ratio='+str(oversampling_ratio))
            for s in bumps_oversampling:
                #print('process_percentage(): bump size='+str(len(s)))
                resample_size = round(len(s)*oversampling_ratio)
                #print('process_percentage(): resample_size='+str(resample_size))
                resampled_sets.append(s.sample(n = resample_size, replace=True))
            #adding uninteresting set
            resampled_sets.append(uninteresting_set)
            result = pd.concat(resampled_sets)
        else:
            for i in range(len(bumps_oversampling)):
                #print('len(self.c_perc) > 1 loop i='+str(i))
                oversampling_ratio = self.c_perc[i]
                #print('process_percentage(): oversampling_ratio='+str(oversampling_ratio))
                resample_size = round(len(bumps_oversampling[i])*oversampling_ratio)
                #print('process_percentage(): resample_size='+str(resample_size))
                resampled_sets.append(bumps_oversampling[i].sample(n = resample_size, replace=True))
            #adding uninteresting set
            resampled_sets.append(uninteresting_set)
            result = pd.concat(resampled_sets)
        return result
