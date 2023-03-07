# General
import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime, timedelta, date
import json
import ast
import glob
import os
import pickle
import re
from itertools import groupby
import time

'''
This Class will return [Cleaned LabResults]

        # Step 1: Clean 1
        df2 = self.clean(df)
        
        # Step 2: Clean 2
        df_concat2 = self.clean2(df2)
        
        # Step 3: Replace Values
        df_concat2 = self.lab_replace_values(df_concat2)
        
        # Step 4: change_labItemUnit_labItemNormalValueRef
        df_concat2 = self.change_labItemUnit_labItemNormalValueRef(df_concat2)
        
You can (select your required features) after this clean
'''

class ExtractLabResult:
    def __init__(self):
        pass
        
    def f(self, x):
        try:
            return ast.literal_eval(str(x))   
        except:
            return []

    def extract_labResults(self, lab_list_dict):
        result = {} 
        for i, dictionary in enumerate(lab_list_dict):
            lab_head_data = dictionary.get('labHeadData', {})
            if lab_head_data.get('confirmReport') != 'Y':
                continue
            result[i] = {
                'departmentName': lab_head_data.get('departmentName'),
                'spcltyName': lab_head_data.get('spcltyName'),
                'labReportData': {}
            }
            lab_list = dictionary.get('labReportData', [])
            for d in lab_list:
                if d.get('confirm') != 'Y':
                    continue
                result[i]['labReportData'][d['labItemsCode']] = {
                    'labItemsNameRef': d['labItemsNameRef'],
                    'labOrderResult': d['labOrderResult'],
                    'labItemsNormalValueRef': d['labItemsNormalValueRef'],
                    'labItemsUnit': d['labItemsUnit']
                }
        return result

    def clean_labResults(self, lab_result_2):
        '''
        To clean these 3 -> labOrderResult, labItemsNormalValueRef, labItemsUnit

        [labOrderResult]
        Mostly labOrderResult is filled with number or string pattern but some of it has like "0 - 0 Unit" pattern
        which it may be the labItemsNormalValueRef and labItemsUnit(Obviously not supposed to be in labOrderResult).
        So, in "0 - 0 Unit" case, we can move it to both of them if it is empthy or "-".

        Moreover, In labOrderResult, we'll only keep the number value (e.g.63.19 (Stage 2) -> use 63.19).

        [labItemsNormalValueRef]
        "0 - 0" , "0 - 0 Unit" and string should be a pattern of labItemsNormalValueRef 
        but sometimes, labItemsNormalValueRef is exactly like labItemsUnit.
        So, if we can't extract it by using the regex labItemsNormalValueRef pattern and they're exacly alike.
        then we'll remove the labItemsNormalValueRef and leave the labItemsUnit there.
        '''

        number_unit_regex = r'^(-?\d+(?:[\.\,]\d+)?\s*-\s*-?\d+(?:[\.\,]\d+)?|[>=<]*\s*-?\d+(?:[\.\,]\d+)?)(.*)'

        for labHeadDataKey, labHeadData in lab_result_2.items():
            for labItemsCode, lab_data in labHeadData['labReportData'].items():
                # Extract number and unit from labOrderResult
                lab_order_result = lab_data['labOrderResult'].strip()
                lab_order_result_match = re.match(number_unit_regex, lab_order_result)
                if lab_order_result_match:
                    try:
                        lab_data['labOrderResult'] = lab_order_result_match.group(1).strip()
                        if "stage" not in lab_order_result_match.group(2).lower():
                            lab_data['labItemsUnit'] = lab_order_result_match.group(2).strip()
                    except IndexError:
                        pass

                # Extract number and unit from labItemsNormalValueRef
                lab_items_normal_value_ref = lab_data['labItemsNormalValueRef'].strip()
                lab_items_normal_value_ref_match = re.match(number_unit_regex, lab_items_normal_value_ref)
                if lab_items_normal_value_ref_match:
                    try:
                        lab_data['labItemsNormalValueRef'] = lab_items_normal_value_ref_match.group(1)
                        if (lab_data['labItemsUnit'] == '' or lab_data['labItemsUnit'] == '-') and "stage" not in lab_items_normal_value_ref_match.group(2).lower():
                            lab_data['labItemsUnit'] = lab_items_normal_value_ref_match.group(2)
                    except IndexError:
                        pass
                elif lab_data['labItemsNormalValueRef'].strip().lower() == lab_data['labItemsUnit'].strip().lower():
                    lab_data['labItemsNormalValueRef'] = ''

        return lab_result_2
        
    def extract_labResults_data(self, data):
        df = data.copy()
        df['lab_result_2'] = df['labResults'].apply(json.loads).apply(self.extract_labResults)
        return df

    def clean_lab_result_data(self, data):
        df = data.copy()
        df['lab_result_3'] = df['lab_result_2'].apply(self.clean_labResults)
        return df

    def extract_labResults3(self, df):
        lab_result = df.get("labResults")
        if lab_result and lab_result.get(df["labHeadData"]):
            return lab_result[df["labHeadData"]]
        return None

    def extract_labReportData(self, df):
        if pd.isna(df['labReportData']) != True:
            for i in range(len(list(df['labReportData'].items()))):
                if list(df['labReportData'].items())[i][0] == df['labItemCode']:
                    return list(df['labReportData'].items())[i][1]
        else:
            return df['labReportData']

    def clean(self, df):
        df2 = self.extract_labResults_data(df)
        df2 = self.clean_lab_result_data(df2)
        df2.drop(['labResults','lab_result_2'], axis=1, inplace=True)
        df2.rename(columns = {'lab_result_3' : 'labResults'},inplace=True)
        return df2

    def clean2(self, df2):
        ex = df2.explode('labResults')
        ex.rename(columns = {'labResults' : 'labHeadData'},inplace=True)
        df_merge = ex.merge(df2, how = 'left' , on=['rowID','cid','visitDateTime'])
        df_merge['labHeadData2'] = df_merge.apply(self.extract_labResults3, axis=1)
        df_merge['labHeadData2'].fillna(value = np.NaN, inplace=True)
        df_merge['labHeadData2'] = df_merge['labHeadData2'].apply(lambda x: {} if pd.isna(x) else x)
        df_concat = pd.concat([df_merge[['rowID', 'cid', 'visitDateTime','labHeadData']], pd.json_normalize(df_merge['labHeadData2'], max_level = 0)], axis=1)
        df_concat_ex = df_concat.explode('labReportData').reset_index(drop=True)
        df_concat_ex.rename(columns = {'labReportData':'labItemCode'}, inplace=True)
        df_merge2 = df_concat_ex.merge(df_concat, how = 'left' , on=['rowID','cid','visitDateTime','labHeadData','departmentName','spcltyName'])
        df_merge2['labReportData2'] = df_merge2.apply(self.extract_labReportData, axis=1)
        df_merge2.drop(['labReportData'], axis=1,inplace=True)
        df_merge2.rename(columns={'labReportData2':'labReportData'}, inplace=True)
        df_concat2 = pd.concat([df_merge2[['rowID','cid','visitDateTime','labHeadData','departmentName','spcltyName','labItemCode']], pd.json_normalize(df_merge2['labReportData'])], axis=1)
        df_concat2 = df_concat2.reindex(columns = ['rowID','cid','visitDateTime','labHeadData','departmentName',
           'spcltyName','labItemCode','labItemsNameRef','labOrderResult','labItemsNormalValueRef','labItemsUnit'])
        
        return df_concat2
    
    def lab_replace_values(self, df):
        replace_values = json.loads(open("replace_values.json", "r", encoding="utf8").read())
        df.replace({"labItemsNameRef": replace_values}, inplace=True)
        return df
    
    def change_labItemUnit_labItemNormalValueRef(self, df):

        for i in df['labItemsNameRef'].value_counts()[0:50].index:
            df['labItemsUnit'].loc[df['labItemsNameRef'] == i] = df[
        'labItemsUnit'].loc[df['labItemsNameRef'] == i].value_counts().keys()[0]
    
        # change labItemsNormalValueRef to the top value_count for the Top 50 lab
        for i in df['labItemsNameRef'].value_counts()[0:50].index:
            df['labItemsNormalValueRef'].loc[df['labItemsNameRef'] == i] = df[
                'labItemsNormalValueRef'].loc[df['labItemsNameRef'] == i].value_counts().keys()[0]
            
        return df 
   
    
    def export_to_csv(self, df):
        df.to_csv('cleaned_labResult.csv', index= False)


    def execution(self, data: pd.DataFrame):
        df = data.copy()
        
        # Step 1: Clean 1
        df2 = self.clean(df)
        
        # Step 2: Clean 2
        df_concat2 = self.clean2(df2)
        
        # Step 3: Replace Values
        df_concat2 = self.lab_replace_values(df_concat2)
        
        # Step 4: change_labItemUnit_labItemNormalValueRef
        # df_concat2 = self.change_labItemUnit_labItemNormalValueRef(df_concat2)
        
        # Export file
        # self.export_to_csv(df_concat2)
        
        print(df_concat2[["labItemsNameRef","labItemsNormalValueRef"]].value_counts()[:50])
        return df_concat2