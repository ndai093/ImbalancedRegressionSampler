from PhiRelevance.PhiUtils1 import phiControl,phi

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from random import seed
from random import randint
from random import random

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
        
        seed(1)
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
        undersampling_and_interesting, new_samples_set = self.preprocess_percentage()
        reduced_cols = new_samples_set.columns.values.tolist()[:-1]
        dups_sample_counts = new_samples_set.pivot_table(index=reduced_cols, aggfunc='size')
        interesting_set_list = self.interesting_set.iloc[:,:-1].values.tolist()

        #new samples from smote
        new_samples_smote = []
        for index, value in dups_sample_counts.items():
            base_sample = list(index)
            #print(f'base_sample={base_sample}')
            kNN_result = self.kNN_calc(self.k, base_sample, interesting_set_list)
            #Generating new samples
            for x in range(value):
                idx = randint(0, 4)
                #print(f'x={x},idx={idx}')
                nb = kNN_result[idx]
                #Generate attribute values
                new_sample = []
                for y in range(len(base_sample)-1):
                    diff = abs(base_sample[y]-nb[y])
                    new_sample.append(base_sample[y]+random()*diff)
                #Calc target value
                a = np.array(new_sample)
                b = np.array(base_sample[:-1])
                d1 = np.linalg.norm(a-b)
                c = np.array(nb[:-1])
                d2 = np.linalg.norm(a-c)
                new_target = (d2*base_sample[-1]+d1*nb[-1])/(d1+d2)
                new_sample.append(new_target)
                #print(f'new_sample={new_sample}')
                new_samples_smote.append(new_sample)
        print(f'len={len(new_samples_smote)}')
        #print(f'{new_samples_smote}')
        #Generate final result
        undersampling_and_interesting.drop('yPhi',axis=1,inplace=True )
        df_new_samples_smote = pd.DataFrame(new_samples_smote)
        df_new_samples_smote.columns = reduced_cols
        frames = [undersampling_and_interesting, df_new_samples_smote]
        result = pd.concat(frames)
        return result

    def preprocess_percentage(self):

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

    def kNN_calc(self, k, sample_as_list, interesting_set_list):
        a = np.array(sample_as_list[:-1])
        for sample_interesting in interesting_set_list:
            b = np.array(sample_interesting[:-1])
            dist = np.linalg.norm(a-b)
            sample_interesting.append(dist)
        kNN_result = sorted(interesting_set_list, key=lambda x:x[-1])[1:(k+1)]
        for j in interesting_set_list:
            del j[-1]
        return kNN_result

    def process_balance(self):
        new_samples_set = self.preprocess_balance()
        reduced_cols = new_samples_set.columns.values.tolist()[:-1]
        dups_sample_counts = new_samples_set.pivot_table(index=reduced_cols, aggfunc='size')
        interesting_set_list = self.interesting_set.iloc[:,:-1].values.tolist()

        #new samples from smote
        new_samples_smote = []
        for index, value in dups_sample_counts.items():
            base_sample = list(index)
            #print(f'base_sample={base_sample}')
            kNN_result = self.kNN_calc(self.k, base_sample, interesting_set_list)
            #Generating new samples
            for x in range(value):
                idx = randint(0, 4)
                #print(f'x={x},idx={idx}')
                nb = kNN_result[idx]
                #Generate attribute values
                new_sample = []
                for y in range(len(base_sample)-1):
                    diff = abs(base_sample[y]-nb[y])
                    new_sample.append(base_sample[y]+random()*diff)
                #Calc target value
                a = np.array(new_sample)
                b = np.array(base_sample[:-1])
                d1 = np.linalg.norm(a-b)
                c = np.array(nb[:-1])
                d2 = np.linalg.norm(a-c)
                new_target = (d2*base_sample[-1]+d1*nb[-1])/(d1+d2)
                new_sample.append(new_target)
                #print(f'new_sample={new_sample}')
                new_samples_smote.append(new_sample)
        print(f'len={len(new_samples_smote)}')
        #print(f'{new_samples_smote}')
        
        #Generate final result
        data = self.getData()
        data.drop('yPhi',axis=1,inplace=True )
        df_new_samples_smote = pd.DataFrame(new_samples_smote)
        df_new_samples_smote.columns = reduced_cols
        frames = [data, df_new_samples_smote]
        result = pd.concat(frames)
        return result        

    def preprocess_balance(self):
        new_samples_per_bump = round(len(self.uninteresting_set) / len(self.bumps_oversampling))
        print(f'process_balance(): resample_size per bump={new_samples_per_bump}')
        resampled_oversampling_set = []
        for s in self.bumps_oversampling:
            ratio = new_samples_per_bump / len(s)
            print(f'ratio={ratio}')
            if ratio>1.0 and ratio<2.0:
                size_new_samples_set = round(len(s)*(ratio-1))
                print(f'size_new_samples_set={size_new_samples_set}')
                resampled_oversampling_set.append(s.sample(n = size_new_samples_set))
            elif ratio>2.0:
                c_perc_frac, c_perc_int = math.modf(ratio)
                print(f'c_perc_int, c_perc_frac =={c_perc_int, c_perc_frac}')
                if c_perc_frac > 0.0:
                    size_frac_new_samples_set = round(len(s)*c_perc_frac)
                    resampled_oversampling_set.append(s.sample(n=size_frac_new_samples_set))
                ss = s.loc[s.index.repeat(int(c_perc_int)-1)]
                resampled_oversampling_set.append(ss)        
        #combining new samples
        new_samples_set = pd.concat(resampled_oversampling_set)
        return new_samples_set        

    def process_extreme(self):
        new_samples_set = self.preprocess_extreme()
        reduced_cols = new_samples_set.columns.values.tolist()[:-1]
        dups_sample_counts = new_samples_set.pivot_table(index=reduced_cols, aggfunc='size')
        interesting_set_list = self.interesting_set.iloc[:,:-1].values.tolist()

        #new samples from smote
        new_samples_smote = []
        for index, value in dups_sample_counts.items():
            base_sample = list(index)
            #print(f'base_sample={base_sample}')
            kNN_result = self.kNN_calc(self.k, base_sample, interesting_set_list)
            #Generating new samples
            for x in range(value):
                idx = randint(0, 4)
                #print(f'x={x},idx={idx}')
                nb = kNN_result[idx]
                #Generate attribute values
                new_sample = []
                for y in range(len(base_sample)-1):
                    diff = abs(base_sample[y]-nb[y])
                    new_sample.append(base_sample[y]+random()*diff)
                #Calc target value
                a = np.array(new_sample)
                b = np.array(base_sample[:-1])
                d1 = np.linalg.norm(a-b)
                c = np.array(nb[:-1])
                d2 = np.linalg.norm(a-c)
                new_target = (d2*base_sample[-1]+d1*nb[-1])/(d1+d2)
                new_sample.append(new_target)
                #print(f'new_sample={new_sample}')
                new_samples_smote.append(new_sample)
        print(f'len={len(new_samples_smote)}')
        #print(f'{new_samples_smote}')
        
        #Generate final result
        data = self.getData()
        data.drop('yPhi',axis=1,inplace=True )
        df_new_samples_smote = pd.DataFrame(new_samples_smote)
        df_new_samples_smote.columns = reduced_cols
        frames = [data, df_new_samples_smote]
        result = pd.concat(frames)
        return result 

    def preprocess_extreme(self):
        print(f'process_extreme(): size of bumps_oversampling={len(self.bumps_oversampling)}')
        print(f'process_extreme(): size of bumps_undersampling={len(self.bumps_undersampling)}')
        print(f'process_extreme(): size of uninteresting_set={len(self.uninteresting_set)}')

        #calculate average cnt
        len_uninteresting_set = len(self.uninteresting_set)
        print(f'process_extreme(): len_uninteresting_set={len_uninteresting_set}')
        len_total = len(self.data)
        print(f'process_extreme(): size of total_set={len_total}')
        average_cnt_uninteresting_set = len_uninteresting_set/len(self.bumps_undersampling)
        print(f'process_extreme(): average_cnt_uninteresting_set={average_cnt_uninteresting_set}')
        resample_size = (average_cnt_uninteresting_set**2.0)/(len_total-len_uninteresting_set)
        print(f'process_extreme(): resample_size={resample_size}')
        resample_size_per_bump = round(resample_size / len(self.bumps_oversampling))
        print(f'process_extreme(): resample_size_per_bump={resample_size_per_bump}')

        resampled_oversampling_set = []
        for s in self.bumps_oversampling:
            ratio = resample_size_per_bump / len(s)
            if ratio>1.0 and ratio<2.0:
                size_new_samples_set = round(len(s)*(ratio-1))
                print(f'process_extreme(): size_new_samples_set={size_new_samples_set}')
                resampled_oversampling_set.append(s.sample(n = size_new_samples_set))
            elif ratio>2.0:
                c_perc_frac, c_perc_int = math.modf(ratio)
                print(f'process_extreme(): c_perc_int, c_perc_frac =={c_perc_int, c_perc_frac}')
                if c_perc_frac > 0.0:
                    size_frac_new_samples_set = round(len(s)*c_perc_frac)
                    print(f'process_extreme(): size_frac_new_samples_set={size_frac_new_samples_set}')
                    resampled_oversampling_set.append(s.sample(n=size_frac_new_samples_set))
                ss = s.loc[s.index.repeat(int(c_perc_int)-1)]
                resampled_oversampling_set.append(ss)        

        #combining new samples
        new_samples_set = pd.concat(resampled_oversampling_set)
        return new_samples_set        