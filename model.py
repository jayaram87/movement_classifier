import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, f1_score
import pickle
import os
from logger import Logger


class Model:
    def __init__(self, file):
        self.df = pd.read_csv(file).iloc[:, 1:]

    def independent_features(self):
        try:
            return self.df.loc[:, [col for col in self.df.columns if col != 'label']]
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Unable to create independent features from the dataframe \n {str(e)}')

    def dependent_feature(self):
        try:
            return self.df.loc[:, 'label']
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Unable to create dependent feature from the dataframe \n {str(e)}')

    def missing_imputation(self):
        try:
            a = self.df.describe().loc['count'] < 42239
            if len(a[a.values == True].index) > 0:
                for col in a[a.values == True].index:
                    if col == 'label':
                        continue
                    else:
                        self.df.col.fillna(self.df.col.mode()[0], inplace=True)
            else:
                Logger('test.log').logger('INFO', f'No missing values')
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Error imputing missing values \n {str(e)}')

    def categorical_label_encoding(self, df):
        try:
            le = LabelEncoder()
            le.fit(df)
            filename = f'label_encoder.sav'
            pickle.dump(le, open(filename, 'wb'))
            Logger('test.log').logger('INFO', f'labels are {le.classes_}')
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Couldnt label encode categorical variable \n {str(e)}')

    def std_scaler(self, df):
        try:
            scaler = StandardScaler()
            scaler.fit(df.values)
            filename = 'std_scaler.sav'
            pickle.dump(scaler, open(filename, 'wb'))
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Numerical columns couldnt be scaled \n {str(e)}')

    def feature_selection_vif(self, x):
        try:
            cols = pd.DataFrame([[x.columns[i], variance_inflation_factor(x.values,i)] for i in range(x.shape[1])], columns=["FEATURE", "VIF_SCORE"])
            cols = cols[cols.VIF_SCORE < 10]['FEATURE'].tolist()
            return cols
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Error selecting features using vif \n {str(e)}')

    def data_transformation(self):
        try:
            self.missing_imputation()
            x = self.independent_features()
            y = self.dependent_feature()
            if not os.path.isfile(os.path.join(os.getcwd(), 'std_scaler.sav')):
                self.std_scaler(x)
            se = pickle.load(open("std_scaler.sav", 'rb'))
            x = pd.DataFrame(se.transform(x), columns=x.columns)
            features = self.feature_selection_vif(x)
            x = x[features].values
            if not os.path.isfile(os.path.join(os.getcwd(), f'label_encoder.sav')):
                self.categorical_label_encoding(y)
            le = pickle.load(open("label_encoder.sav", 'rb'))
            y = le.transform(y)
            return features, x, y
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Couldnt transform the dataset \n {str(e)}')

    def train_test_split(self):
        try:
            features, x, y = self.data_transformation()
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            return features, x_train, x_test, y_train, y_test
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Couldnt split the dataset into train test sets \n {str(e)}')

    def eval_model(self, model, x, y):
        try:
            y_pred = model.predict(x)
            f1 = f1_score(y, y_pred, average='weighted')
            return f1
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Error couldnt evaluate the model \n {str(e)}')

    def logr_model(self, solver):
        try:
            features, x_train, x_test, y_train, y_test = self.train_test_split()
            lr = LogisticRegression(solver=solver)
            lr.fit(x_train, y_train)
            return f'lg_{solver}', lr, self.eval_model(lr, x_test, y_test)
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Error creating a LR model \n {str(e)}')

    def best_model(self):
        try:
            models = {}
            best_model = None
            for i in [self.logr_model(solver='newton-cg'), self.logr_model(solver='sag'), self.logr_model(solver='saga'), self.logr_model(solver='lbfgs')]:
                models[i[0]] = [i[1], i[2]]
            print(models)
            best_model = sorted(models.values(), key=lambda val: val[1], reverse=True)[0][0]
            filename = 'model.sav'
            pickle.dump(best_model, open(filename, 'wb'))
            return best_model
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Couldnt save the best model \n {str(e)}')