from PhiRelevance.PhiUtils1 import phiControl,phi

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

class SmoteRRegression:
    """
    Class SmoteRRegression takes arguments as follows:
        data - Pandas data frame with target value as last column, rest columns should be features/attributes
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
        k - The number of nearest neighbors, default value is 5      
    """
    def __init__(self, data, method='auto', extrType='both', thr_rel=1.0, controlPts=[], c_perc="balance", k=5):
        
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
        
        self.k = k
        self.coef = 1.5

    def processCPerc(self, c_perc):
        for x in c_perc:
            if x < 1.0:
                self.c_perc_undersampling.append(float(x))
            elif x > 1.0:
                self.c_perc_oversampling.append(float(x))
            else:
                print('c_perc value in list should not be 1!')
        print(f'c_perc_undersampling: {self.c_perc_undersampling}')
        print(f'c_perc_oversampling: {self.c_perc_oversampling}')

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
    
    def getK(self):
        return self.k

    def set_obj_interesting_set(self, data):
        self.interesting_set = self.get_interesting_set(data)

    def get_obj_interesting_set(self):
        return self.interesting_set

    def set_obj_uninteresting_set(self, data):
        self.uninteresting_set = self.get_uninteresting_set(data)

    def get_obj_uninteresting_set(self):
        return self.uninteresting_set

    def set_obj_bumps(self, data):
        self.bumps_undersampling, self.bumps_oversampling = self.calc_bumps(data)

    def get_obj_bumps(self):
        return self.bumps_undersampling, self.bumps_oversampling

    def resample(self):

        yPhi, ydPhi, yddPhi = self.calc_rel_values()

        data1 = self.preprocess_data(yPhi)
        #interesting set
        self.set_obj_interesting_set(data1)
        #uninteresting set
        self.set_obj_uninteresting_set(data1)
        #calculate bumps
        self.set_obj_bumps(data1)

        if self.c_perc == 'percentage list':
            resampled = self.process_percentage()
        elif self.c_perc == 'balance':
            resampled = self.process_balance()
        elif self.c_perc == 'extreme':
            resampled = self.process_extreme()

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

    def process_percentage_new_base_samples(self):

        #process undersampling
        len_c_perc_undersampling = len(self.c_perc_undersampling)
        print(f'len_c_perc_undersampling={len_c_perc_undersampling}')
        len_bumps_undersampling = len(self.bumps_undersampling)
        print(f'len_bumps_undersampling={len_bumps_undersampling}')
        resampled_sets = []

        if len_c_perc_undersampling == 0:
            print('no undersampling, append uninteresting set directly')
            resampled_sets.append(self.uninteresting_set)
        elif len_c_perc_undersampling == 1:
            undersample_perc = self.c_perc_undersampling[0]
            print('len(self.c_perc) == 1')
            print(f'process_percentage(): undersample_perc={undersample_perc}')
            #iterate undersampling bumps to apply undersampling percentage
            for s in self.bumps_undersampling:
                print(f'process_percentage(): bump size={len(s)}')
                resample_size = round(len(s)*undersample_perc)
                print(f'process_percentage(): resample_size={resample_size}')
                resampled_sets.append(s.sample(n = resample_size))
        elif len_c_perc_undersampling == len_bumps_undersampling:
            for i in range(len(self.bumps_undersampling)):
                print(f'len(self.c_perc) > 1 loop i={i}')
                undersample_perc = self.c_perc_undersampling[i]
                print(f'process_percentage(): undersample_perc={undersample_perc}')
                resample_size = round(len(self.bumps_undersampling[i])*undersample_perc)
                print(f'process_percentage(): resample_size={resample_size}')
                resampled_sets.append(self.bumps_undersampling[i].sample(n = resample_size))
        else:
            print(f'length of c_perc for undersampling {len_c_perc_undersampling} != length of bumps undersampling {len_bumps_undersampling}')
        #uninteresting bumps are now stored in list resampled_sets
        #also adding original interesting set
        resampled_sets.append(self.interesting_set)

        #Oversampling with SmoteR
        len_c_perc_oversampling = len(self.c_perc_oversampling)
        print(f'len_c_perc_oversampling={len_c_perc_oversampling}')
        len_bumps_oversampling = len(self.bumps_oversampling)
        print(f'len_bumps_oversampling={len_bumps_oversampling}')
        resampled_oversampling_set = []
        if len(self.c_perc_oversampling) == 1:
            #oversampling - new samples set
            c_perc_frac, c_perc_int = 0.0, 0.0
            for s in self.bumps_oversampling:
                # size of the new samples
                print(f'c_perc_oversampling[0]={self.c_perc_oversampling[0]}')
                if self.c_perc_oversampling[0]>1.0 and self.c_perc_oversampling[0]<2.0:
                    size_new_samples_set = round(len(s)*(self.c_perc_oversampling[0]-1))
                    print(f'size_new_samples_set={size_new_samples_set}')
                    resampled_oversampling_set.append(s.sample(n = size_new_samples_set))
                elif self.c_perc_oversampling[0]>2.0:
                    c_perc_frac, c_perc_int = math.modf(self.c_perc_oversampling[0])
                    print(f'c_perc_int, c_perc_frac =={c_perc_int, c_perc_frac}')
                    if c_perc_frac > 0.0:
                        size_frac_new_samples_set = round(len(s)*c_perc_frac)
                        resampled_oversampling_set.append(s.sample(n=size_frac_new_samples_set))
                    ss = s.loc[s.index.repeat(int(c_perc_int)-1)]
                    resampled_oversampling_set.append(ss)
        
        elif len_c_perc_oversampling == len_bumps_oversampling:
            for i in range(len(self.bumps_oversampling)):
                print(f'len(self.c_perc) > 1 loop i={i}')
                c_perc_bump = self.c_perc_oversampling[i]
                print(f'process_percentage(): undersample_perc={c_perc_bump}')

                if c_perc_bump>1.0 and c_perc_bump<2.0:
                    size_new_samples_set = round(len(s)*(c_perc_bump-1))
                    print(f'size_new_samples_set={size_new_samples_set}')
                    resampled_oversampling_set.append(s.sample(n = size_new_samples_set))
                elif c_perc_bump>2.0:
                    c_perc_frac, c_perc_int = math.modf(self.c_perc_oversampling[0])
                    print(f'c_perc_int, c_perc_frac =={c_perc_int, c_perc_frac}')
                    if c_perc_frac>0.0:
                        size_frac_new_samples_set = round(len(self.bumps_oversampling[i])*c_perc_frac)
                        resampled_oversampling_set.append(self.bumps_oversampling[i].sample(n=size_frac_new_samples_set))
                    ss = self.bumps_oversampling[i].loc[self.bumps_oversampling[i].index.repeat(int(c_perc_int)-1)]
                    resampled_oversampling_set.append(ss)        

        else:
            print(f'length of c_perc for oversampling {len_c_perc_oversampling} != length of bumps oversampling {len_bumps_oversampling}')
        
        #Combining all undersampling sets and interesting set
        undersampling_and_interesting = pd.concat(resampled_sets)
        #Combining all new samples
        new_samples_set = pd.concat(resampled_oversampling_set)
        
        return undersampling_and_interesting, new_samples_set

    def smoter_calc(self, new_samples_set, interesting_set):
        pass