import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

from pygam import LogisticGAM, s, f
from scipy.special import expit, logit

# import pdb

class explainer:
    def __init__(self, X_train, y_train, X_test, y_test, p_thresholds=[0.1, 0.5, 0.9], seed=7):
        self.seed = seed
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_train_new = pd.DataFrame()
        self.X_test_new = pd.DataFrame()

        self.selected_features = []
        self.breakpoints_list = []
        self.seed = 7
        self.gam = 0
        self.clf = 0
        self.calibrated_clf = 0
        self.p_thresholds = p_thresholds
        self.scoring_thresholds = []

        # self.unit_score = 1
        self.beta_values = []
        self.intercept_value = 0
        self.scores = []

        self.scoring_table_columns = ['Score', 'Probability']
        self.scoring_table = pd.DataFrame(columns = self.scoring_table_columns)

    def fit(self, top_n, method = 'GAM'):
        print('| Step 1  ==> Selecting top n features')
        scaler = MinMaxScaler()
        select = SelectKBest(chi2, k=top_n)
        select.fit(scaler.fit_transform(self.X_train), self.y_train)
        self.selected_features = select.get_feature_names_out(self.X_train.columns)
        if method == 'GAM':
            print('| Step 2 ==> Transforming features based on GAM')
            self.find_features_categories_gam()
        elif method == 'quantile':
            print('| Step 2 ==> Transforming features based on quantiles')
            self.find_features_categories_quantiles()
        else:
            print('No method selected!')
            return 'Incomplete method'

        print('| Step 3 ==> Fitting logistic regression model on transformed categories')
        self.fit_logreg()

        print('| Step 4 ==> Calibrating logistic regression model')

        skf = StratifiedKFold(n_splits=5, random_state=self.seed, shuffle=True)

        self.plot_calibration_original()
        self.calibrate(cv=skf)
        self.plot_calibration_calibrated()

        print('| Step 5 ==> Fit clinical score calculator')
        self.fit_calculator()

    def find_features_categories_gam(self):
        index = 0
        for feature in self.selected_features:
            if len(self.X_train[feature].value_counts()) < 3:
                g = f
            else:
                g = s

            if index == 0:
                equation = g(index)
            else:
                equation += g(index)

            index += 1

        # Get GAM response for individual features
        self.gam = LogisticGAM(equation).gridsearch(
            self.X_train[self.selected_features].values, self.y_train)

        self.breakpoints_list = []
        self.X_train_new = pd.DataFrame()
        self.X_test_new = pd.DataFrame()

        for i in range(len(self.selected_features)):
            range_arr = self.find_feature_ranges_gam(i)
            feature = self.selected_features[i]
            self.breakpoints_list.append(range_arr)

            if len(self.X_train[feature].value_counts()) < 3:
                self.X_train_new[feature] = self.X_train[feature]
                self.X_test_new[feature] = self.X_test[feature]
            else:
                df = self.find_feature_categories(feature, range_arr, self.X_train)
                self.X_train_new[df.columns] = df
                df = self.find_feature_categories(feature, range_arr, self.X_test)
                self.X_test_new[df.columns] = df

    def find_feature_ranges_gam(self, i):
        XX = self.gam.generate_X_grid(term=i)

        xs = XX[:, i]
        ys = self.gam.partial_dependence(term=i, X=XX)

        # Find ranges with different risks with respect to baseline
        index = 0
        ys_sign = np.sign(ys)

        range_arr = []
        merge_flag = 0

        for xi in xs:
            if index == 0 or index == (len(ys) - 1):
                range_arr.append(xi)
                start = xi
            else:
                if ys_sign[index] != ys_sign[index-1]:
                    x1, y1 = xs[index-1], ys[index-1]
                    x2, y2 = xs[index], ys[index]

                    gradient = (y2-y1)/(x2-x1)
                    intercept = y1 - gradient * x1
                    x_intercept = - intercept / gradient

                    end = x_intercept
                    range_index = np.where((xs >= start) &
                                           (xs <= end))

                    if len(range_index[0]) != 0:
                        range_arr.append(end)
                        start = end
            index += 1

        return range_arr

    def find_features_categories_quantiles(self, quantiles=[0.2, 0.8]):
        self.breakpoints_list = []
        self.X_train_new = pd.DataFrame()
        self.X_test_new = pd.DataFrame()

        df_quantiles = self.X_train[self.selected_features].quantile(quantiles)

        for i in range(len(self.selected_features)):
            feature = self.selected_features[i]
            if len(df_quantiles[feature].value_counts()) == len(quantiles):
                range_arr = df_quantiles[feature].values
                range_arr = np.append(self.X_train[feature].min(), range_arr)
                range_arr = np.append(range_arr, self.X_train[feature].max())
            else:
                max_value = self.X_train[feature].max()
                min_value = self.X_train[feature].min()
                range_arr = [ min_value+(i+1)/len(quantiles)*(max_value-min_value) for i in range(len(quantiles))]
                range_arr = np.append(min_value, range_arr)
                range_arr = np.append(range_arr, max_value)


            if len(self.X_train[feature].value_counts()) < 3:
                self.breakpoints_list.append(np.array([0, 0.5, 1]))
                self.X_train_new[feature] = self.X_train[feature]
                self.X_test_new[feature] = self.X_test[feature]
            else:
                self.breakpoints_list.append(range_arr)
                df = self.find_feature_categories(feature, range_arr, self.X_train)
                self.X_train_new[df.columns] = df
                df = self.find_feature_categories(feature, range_arr, self.X_test)
                self.X_test_new[df.columns] = df

    def find_feature_categories(self, feature, range_arr, X_df):
        title_arr = []
        values_arr = []

        for j in range(len(range_arr)):
            if j != 0:
                if j == 1:
                    title = feature + ' <= %.5g' % range_arr[j]
                    title_arr.append(title)
                    values_arr.append(( X_df[feature] <= range_arr[j] ).values)
                elif j == len(range_arr)-1:
                    title = feature + ' > %.5g' % range_arr[j-1]
                    title_arr.append(title)
                    values_arr.append(( X_df[feature] > range_arr[j-1]  ).values)
                else:
                    title = '{0: .5g} <  {1:s} <= {2: .5g}'.format(
                        range_arr[j-1], feature, range_arr[j])
                    title_arr.append(title)
                    values_arr.append(
                        (X_df[feature] > range_arr[j-1]).values \
                        & (X_df[feature] <= range_arr[j]).values )


        values_arr = np.transpose(np.array(values_arr))
        df = pd.DataFrame(values_arr, columns=title_arr)
        if df.shape[1] == 2:
            df = df.iloc[: , :-1]
        return df.astype(int).set_index(X_df.index)

    def fit_logreg(self):
        self.clf = LogisticRegression(random_state=self.seed, solver='liblinear')
        self.clf.fit(self.X_train_new, self.y_train)

    def plot_calibration_original(self, n_bins=10):
        y_pred = self.clf.predict_proba(self.X_test_new)[:, 1]
        self.plot_calibration_curve(y_pred, n_bins)

    def calibrate(self, cv=5):
        self.calibrated_clf = CalibratedClassifierCV(self.clf, cv=cv, method='isotonic')
        self.calibrated_clf.fit(self.X_train_new, self.y_train)

    def plot_calibration_calibrated(self, n_bins=10):
        y_pred = self.calibrated_clf.predict_proba(self.X_test_new)[:, 1]
        self.plot_calibration_curve(y_pred, n_bins)

    def plot_calibration_curve(self, y_pred, n_bins=10):
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(self.y_test, y_pred, n_bins=n_bins)

        plt.plot(mean_predicted_value, fraction_of_positives, "s-")
        plt.show()

        plt.hist(y_pred, range=(0, 1), bins=n_bins, histtype="step", lw=2)
        plt.show()

    def calculate_beta_values(self):
        beta_values_list = []
        intercept_values_list = []
        for calibrated_clf in self.calibrated_clf.calibrated_classifiers_:
            beta_values = calibrated_clf.base_estimator.coef_[0]
            beta_values_list.append(beta_values)
            intercept_values_list.append(calibrated_clf.base_estimator.intercept_[0])

        self.beta_values = np.array(beta_values_list).sum(axis=0) / len(beta_values_list)
        self.intercept_value = np.array(intercept_values_list).mean()

    def fit_calculator(self, threshold=0.3, verbose=False):
        print('Fit clincal score calculator')
        self.calculate_beta_values()
        beta_values_no_zeros = self.beta_values[np.abs(self.beta_values) >= threshold]
        self.unit_beta_value = np.min(np.abs(beta_values_no_zeros))
        self.scores = np.rint(np.true_divide(self.beta_values, self.unit_beta_value))
        min_score = np.min(self.scores)
        self.scores = self.scores - min_score

        self.scoring_table = pd.DataFrame(columns=self.scoring_table_columns)
        self.scoring_thresholds = []

        self.base_log_odds = self.intercept_value + min_score * len(self.selected_features) * self.unit_beta_value

        max_value = sum(self.scores)

        for i in range(int(max_value) + 1):
            print('Score: ' + str(i))
            log_odds = self.base_log_odds + self.unit_beta_value * i

            new_row = pd.DataFrame([[
                i, expit(log_odds)
            ]], columns=self.scoring_table_columns)
            self.scoring_table = self.scoring_table.append(new_row)

            print('Probability: ' + str(expit(log_odds)))
            print('')

        for p_threshold in self.p_thresholds:
            print("Threshold:" + str(p_threshold))
            score_threshold = self.scoring_table[
                self.scoring_table['Probability'] <= p_threshold
            ]['Score'].max()
            self.scoring_thresholds.append(score_threshold)
            print(score_threshold)
            print("")

    def get_clf_performance(self):
        roc_auc = roc_auc_score(self.y_test,
            self.clf.predict_proba(self.X_test_new)[:, 1])
        print("ROC AUC: " + str(roc_auc))

        average_precision = average_precision_score(self.y_test,
            self.clf.predict_proba(self.X_test_new)[:, 1])
        print("Average Precision: " + str(average_precision))

        accuracy = accuracy_score(self.y_test,
            self.clf.predict(self.X_test_new))
        print("Accuracy: " + str(accuracy))

    def get_calibrated_clf_performance(self):
        roc_auc = roc_auc_score(self.y_test,
            self.calibrated_clf.predict_proba(self.X_test_new)[:, 1])
        print("ROC AUC: " + str(roc_auc))

        average_precision = average_precision_score(self.y_test,
            self.calibrated_clf.predict_proba(self.X_test_new)[:, 1])
        print("Average Precision: " + str(average_precision))

        accuracy = accuracy_score(self.y_test,
            self.calibrated_clf.predict(self.X_test_new))
        print("Accuracy: " + str(accuracy))

    def predict_calculator(self, df_original, verbose=False, threshold_choice=1):
        df_converted = pd.DataFrame()
        i = 0
        for feature in self.selected_features:
            if len(df_original[feature].value_counts()) < 3:
                df_converted[feature] = df_original[feature]
            else:
                range_arr = self.breakpoints_list[i]
                df = self.find_feature_categories(feature, range_arr, df_original)
                df_converted[df.columns] = df
            i += 1

        scores = np.dot(df_converted.values, self.scores)
        probs = expit(scores * self.unit_beta_value + self.base_log_odds)
        predictions = (scores >= self.scoring_thresholds[threshold_choice]).astype(int)

        return scores, probs, predictions

