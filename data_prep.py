from zipfile import ZipFile
import os
import pandas as pd
import csv
from logger import Logger

class DataPrep:
    def __init__(self):
        self.df = pd.DataFrame()

    def unzip_data(self):
        with ZipFile("ARem.zip", "r") as a:
            a.extractall(os.path.join(os.getcwd(), 'data'))

    def data_extraction(self):
        try:
            if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
                self.unzip_data()
            for dir, subdirs, files in os.walk(os.path.join(os.getcwd(), 'data')):
                if len(subdirs) == 0:
                    label = dir.split('\\')[-1]
                    print(label)
                    for i in files:
                        with open(os.path.join(os.getcwd(), 'data', label, i)) as csv_file:
                            reader = csv.reader(csv_file, delimiter=',')
                            next(reader)
                            next(reader)
                            next(reader)
                            next(reader)
                            cols = next(reader)
                            b = []
                            df1 = pd.DataFrame()
                            for row in reader:
                                if len(row) <= 1:
                                    row = row[0].split(' ')[0:7]
                                b.append(row[0:7])
                            df1 = pd.DataFrame(b, columns=cols)
                            df1['label'] = label
                        self.df = pd.concat([self.df, df1], axis=0)
            self.df.to_csv('sample.csv', index=None)
            Logger('test.log').logger('ERROR', f'successfully extracted data')
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Error extracting data \n {str(e)}')

a = DataPrep()
a.data_extraction()
