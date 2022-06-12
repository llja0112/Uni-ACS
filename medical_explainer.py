import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from csaps import csaps
from pygam import LogisticGAM, s, f

from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

class explainer:
    def __init__(self, clf, X_train, y_train, X_test, y_test, p_thresholds=[0.1, 0.5, 0.9], pipeline_clf='Log_Reg', seed = 7):
        self.version = '0.1'
        self.seed = seed
        self.clf = clf
        self.calibrated_clf = np.nan
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.fraction_of_positives = 0
        self.mean_predicted_value = 0
        self.shap_values = 0

        self.variables = []
        self.breakpoints_list = []
        self.shap_array_list = []
        self.shap_sd_array_list = []
        self.shap_n_array_list = []
        # self.p_array_list = []
        # self.or_array_list = []
        self.max_shap_score = 0

        self.xs_array = []
        self.ys_array = []

        self.unit_shap_value = 0
        self.score_array_list = []

        self.p_thresholds = p_thresholds
        self.scoring_thresholds = []
        self.scoring_table_columns = ['Score', 'Probability']
        self.scoring_table = pd.DataFrame(columns = self.scoring_table_columns)

        self.pipeline_clf = pipeline_clf


    def __version__(self):
        return self.version

    def nan_helper(self, y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    def plot_calibration_original(self, n_bins=10):
        y_pred = self.clf.predict_proba(self.X_test)[:, 1]
        self.plot_calibration_curve(y_pred, n_bins)

    def plot_calibration_calibrated(self, n_bins=10):
        y_pred = self.calibrated_clf.predict_proba(self.X_test)[:, 1]
        self.plot_calibration_curve(y_pred, n_bins)

    def plot_calibration_curve(self, y_pred, n_bins=10):
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(self.y_test, y_pred, n_bins=n_bins)

        plt.plot(mean_predicted_value, fraction_of_positives, "s-")
        plt.ylabel('Fraction of positives')
        plt.xlabel('Mean predicted value')
        plt.show()

        plt.hist(y_pred, range=(0, 1), bins=n_bins, histtype="step", lw=2)
        plt.ylabel('Count')
        plt.xlabel('Mean predicted value')
        plt.show()

    def calibrate(self, cv=5):
        self.calibrated_clf = CalibratedClassifierCV(self.clf, cv=cv, method='isotonic')
        self.calibrated_clf.fit(self.X_train, self.y_train)

    def calculate_kernel_shap(self):
        self.calibrated_clf.fit(self.X_train.values, self.y_train.values)
        shap_values_list = []
        shap_expected_values_list = []
        i = 0
        for calibrated_classifier in self.calibrated_clf.calibrated_classifiers_:
            print('Kernel Explainer Iteration ' + str(i))
            if self.pipeline_clf == 'Neural_Network':
                estimator = calibrated_classifier.base_estimator.named_steps['Neural_Network']
            else:
                estimator = calibrated_classifier.base_estimator

            explainer = shap.KernelExplainer(
                calibrated_classifier.base_estimator.predict, self.X_train[:500].values)
            shap_values = explainer.shap_values(self.X_train.values, nsamples=500)
            # shap_values = explainer.shap_values(self.X_train[50:1000].values, nsamples=500)

            # shap_values_post = shap_values[1] + explainer.expected_value[1]
            shap_values_post = shap_values + explainer.expected_value
            shap_values_post = np.where(shap_values_post >= 0.0001, shap_values_post, 0.0001)
            shap_values_post = np.where(shap_values_post <= 0.9999, shap_values_post, 0.9999)

            # shap_values_list.append(shap_values_post)
            shap_values_list.append(logit(shap_values_post))
            # shap_expected_values_list.append(explainer.expected_value[1])
            shap_expected_values_list.append(logit(explainer.expected_value))
            i += 1

        self.shap_values = np.array(shap_values_list).sum(axis=0) / len(shap_values_list)
        self.expected_value = np.mean(shap_expected_values_list)

    def calculate_tree_shap(self):
        shap_values_list = []
        shap_expected_values_list = []
        for calibrated_classifier in self.calibrated_clf.calibrated_classifiers_:
            explainer = shap.TreeExplainer(
                calibrated_classifier.base_estimator,
                feature_perturbation = "tree_path_dependent")
            shap_values = explainer.shap_values(self.X_train)
            expected_value = explainer.expected_value
            if len(shap_values) == 2:
                shap_values = logit(shap_values[1]+expected_value[1]) - logit(expected_value[1])
                expected_value = logit(expected_value[1])
            shap_values_list.append(shap_values)
            shap_expected_values_list.append(expected_value)

        self.shap_values = np.array(shap_values_list).sum(axis=0) / len(shap_values_list)
        self.expected_value = np.mean(shap_expected_values_list)

    def calculate_linear_shap(self):
        shap_values_list = []
        shap_expected_values_list = []
        for calibrated_classifier in self.calibrated_clf.calibrated_classifiers_:
            if self.pipeline_clf == 'Log_Reg':
                estimator = calibrated_classifier.base_estimator.named_steps['Log_Reg']
                X_train = calibrated_classifier.base_estimator.named_steps['Scaler'].transform(self.X_train)
            else:
                estimator = calibrated_classifier.base_estimator
                X_train = self.X_train

            explainer = shap.LinearExplainer(
                estimator,
                X_train)
                # feature_perturbation = "interventional")
            shap_values = explainer.shap_values(X_train)
            shap_values_list.append(shap_values)
            shap_expected_values_list.append(explainer.expected_value)

        self.shap_values = np.array(shap_values_list).sum(axis=0) / len(shap_values_list)
        self.expected_value = np.mean(shap_expected_values_list)

    def get_clf_performance(self):
        roc_auc = roc_auc_score(self.y_test,
            self.clf.predict_proba(self.X_test)[:, 1])
        print("ROC AUC: " + str(roc_auc))

        average_precision = average_precision_score(self.y_test,
            self.clf.predict_proba(self.X_test)[:, 1])
        print("Average Precision: " + str(average_precision))

        accuracy = accuracy_score(self.y_test,
            self.clf.predict(self.X_test))
        print("Accuracy: " + str(accuracy))

    def get_calibrated_clf_performance(self):
        roc_auc = roc_auc_score(self.y_test,
            self.calibrated_clf.predict_proba(self.X_test)[:, 1])
        print("ROC AUC: " + str(roc_auc))

        average_precision = average_precision_score(self.y_test,
            self.calibrated_clf.predict_proba(self.X_test)[:, 1])
        print("Average Precision: " + str(average_precision))

        accuracy = accuracy_score(self.y_test,
            self.calibrated_clf.predict(self.X_test))
        print("Accuracy: " + str(accuracy))

    def find_breakpoints(self, column_label, moving_average_size=50,
                         spline_sample_size=100, plot_graphs=False):
        column_index = self.X_train.columns.get_indexer([column_label])[0]
        column_shap_values = self.shap_values[:, column_index]
        column_values = self.X_train[column_label].values

        if plot_graphs:
            plt.scatter(self.X_train[column_label], self.shap_values[:, column_index])
            plt.show()

        # sort according to column values
        sorted_index = np.argsort(column_values)
        column_shap_values_sorted = column_shap_values[sorted_index]
        column_values_sorted = column_values[sorted_index]

        # Start of moving average algorithm
        xs = np.linspace(
            column_values_sorted[0],
            column_values_sorted[-1],
            moving_average_size)
        x = []
        y = []

        index = 0
        for xi in xs:
            if index == 0:
                start = xi
                x.append(xi)
                y.append(column_shap_values_sorted[0])
            else:
                if index == moving_average_size - 1:
                    range_index = np.where(
                        (column_values_sorted >= start) &
                        (column_values_sorted <= xi))
                else:
                    range_index = np.where(
                        (column_values_sorted >= start) &
                        (column_values_sorted < xi))
                x_mean = (start + xi)/2

                if len(range_index[0]) > 0:
                    y_mean = np.mean(column_shap_values_sorted[range_index])
                else:
                    y_mean = np.nan

                x.append(x_mean)
                y.append(y_mean)
                start = xi

            if index == moving_average_size - 1:
                x.append(xi)
                y.append(column_shap_values_sorted[-1])
            index += 1

        y = np.array(y)

        # Interpolate nan values
        nans, mask = self.nan_helper(y)
        y[nans]= np.interp(mask(nans), mask(~nans), y[~nans])

        # Find the best fitting spline
        xs = np.linspace(x[0], x[-1], spline_sample_size)
        smoothing_result = csaps(x, y, xs)
        ys = smoothing_result.values

        self.xs_array.append(xs)
        self.ys_array.append(ys)

        if plot_graphs:
            plt.plot(x, y, 'o', xs, ys, '-')
            plt.show()

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
                    range_index = np.where((column_values_sorted >= start) &
                                           (column_values_sorted <= end))

                    if len(range_index[0]) != 0:
                        range_arr.append(end)
                        start = end
            index += 1


        # Calculate risks and odds ratio within each range
        # column_actual_shap_values_sorted = np.add(
        #     column_shap_values_sorted, self.expected_value)

        shap_array = []
        shap_sd_array = []
        shap_n_array = []
        # p_array = []
        # or_array = []

        for index in range(len(range_arr)):
            if index != 0:
                range_index =  np.where((column_values_sorted >= range_arr[index-1]) &
                                        (column_values_sorted <= range_arr[index]))

                # mean_shap_value = np.mean(column_actual_shap_values_sorted[range_index])
                shap_array.append(np.mean(column_shap_values_sorted[range_index]))
                shap_sd_array.append(np.std(column_shap_values_sorted[range_index]))
                shap_n_array.append(len(column_shap_values_sorted[range_index]))
                # p = expit(mean_shap_value)
                # p_array.append(p)
                # or_array.append(p/(1-p))

        return range_arr, shap_array, shap_sd_array, shap_n_array#, p_array, or_array

    def find_breakpoints_novel(self, verbose=False):
        self.xs_array = []
        self.ys_array = []
        for variable in self.variables:
            # breakpoints, shap_array, p_array, or_array = self.find_breakpoints(
            breakpoints, shap_array, shap_sd_array, shap_n_array = self.find_breakpoints(
                variable, plot_graphs=False)
            self.breakpoints_list.append(breakpoints)
            self.shap_array_list.append(shap_array)
            self.shap_sd_array_list.append(shap_sd_array)
            self.shap_n_array_list.append(shap_n_array)
            # self.p_array_list.append(p_array)
            # self.or_array_list.append(or_array)

            self.max_shap_score += np.max(shap_array)

            if verbose:
                print(variable)
                print(breakpoints)
                print(shap_array)
                print(shap_sd_array)
                print(shap_n_array)
                # print(p_array)
                # print(or_array)
                print("")

    def find_breakpoints_quantile(self, quantiles=[0.2, 0.5, 0.8], verbose=False):
        self.xs_array = []
        self.ys_array = []
        df_quantiles = self.X_train[self.variables].quantile(quantiles)
        for i in range(len(self.variables)):
            variable = self.variables[i]
            if len(df_quantiles[variable].value_counts()) == len(quantiles):
                breakpoints = df_quantiles[variable].values
                breakpoints = np.append(self.X_train[variable].min(), breakpoints)
                breakpoints = np.append(breakpoints, self.X_train[variable].max())
            else:
                max_value = self.X_train[variable].max()
                min_value = self.X_train[variable].min()
                breakpoints = [ min_value+(i+1)/len(quantiles)*(max_value-min_value) for i in range(len(quantiles))]
                breakpoints = np.append(min_value, breakpoints)
                breakpoints = np.append(breakpoints, max_value)

            column_index = self.X_train.columns.get_indexer([variable])[0]
            column_shap_values = self.shap_values[:, column_index]
            column_values = self.X_train[variable].values

            # sort according to column values
            sorted_index = np.argsort(column_values)
            column_shap_values_sorted = column_shap_values[sorted_index]
            column_values_sorted = column_values[sorted_index]


             # Calculate risks and odds ratio within each range
            # column_actual_shap_values_sorted = np.add(
            #     column_shap_values_sorted, self.expected_value)

            shap_array = []
            shap_sd_array = []
            shap_n_array = []
            # p_array = []
            # or_array = []

            for index in range(len(breakpoints)):
                if index != 0:
                    range_index =  np.where((column_values_sorted >= breakpoints[index-1]) &
                                            (column_values_sorted <= breakpoints[index]))

                    # mean_shap_value = np.mean(column_actual_shap_values_sorted[range_index])
                    shap_array.append(np.mean(column_shap_values_sorted[range_index]))
                    shap_sd_array.append(np.mean(column_shap_values_sorted[range_index]))
                    shap_n_array.append(len(column_shap_values_sorted[range_index]))
                    # p = expit(mean_shap_value)
                    # p_array.append(p)
                    # or_array.append(p/(1-p))

            self.breakpoints_list.append(breakpoints)
            self.shap_array_list.append(shap_array)
            self.shap_sd_array_list.append(shap_sd_array)
            self.shap_n_array_list.append(shap_n_array)
            # self.p_array_list.append(p_array)
            # self.or_array_list.append(or_array)
            self.max_shap_score += np.max(shap_array)

            if verbose:
                print(variable)
                print(breakpoints)
                print(shap_array)
                print(shap_sd_array)
                print(shap_n_array)
                # print(p_array)
                # print(or_array)
                print("")

    def fit(self, top_n=10, verbose=False, method='novel', shap_method='linear', n_splits=5, calculator_threshold=0.05):
        # Feature selection with top SHAP values

        self.breakpoints_list = []
        self.shap_array_list = []
        self.shap_sd_array_list = []
        self.shap_n_array = []
        # self.p_array_list = []
        # self.or_array_list = []
        self.shap_max_score = 0

        skf = StratifiedKFold(n_splits=n_splits, random_state=self.seed, shuffle=True)

        print('| Step 1  ==> Calibrating model')
        self.plot_calibration_original()
        self.calibrate(cv=skf)
        self.plot_calibration_calibrated()
        self.get_clf_performance()
        self.get_calibrated_clf_performance()
        print('')

        print('| Step 2 ==> Calculate SHAP values')
        if shap_method == 'linear':
            self.calculate_linear_shap()
        elif shap_method == 'tree':
            self.calculate_tree_shap()
        elif shap_method == 'kernel':
            # print('Neural Network Placeholder')
            self.calculate_kernel_shap()
        else:
            print('SHAP explainer not executed')
            return ''
        print('')

        self.variables = self.X_train.columns[
            np.argsort(np.abs(self.shap_values).mean(0))
        ][::-1][:top_n].values

        print('| Step 3 ==> Fit clinical score calculator')
        if method == 'novel':
            print('Novel fitting')
            self.find_breakpoints_novel(verbose)
        elif method == 'quantile':
            print('Quantile fitting')
            self.find_breakpoints_quantile(verbose=verbose)
        else:
            print('Nothing fitted!')
            return ''

        self.fit_calculator(calculator_threshold)
        self.generate_calculator_scores()

    def find_breakpoint_level(self, value, index, verbose=False):
        level = 0
        for breakpoint in self.breakpoints_list[index][1:-1]:
            if value <= breakpoint:
                if verbose:
                    verbose_string = "Value <= " + str(breakpoint)
                    if level > 0:
                        verbose_string = str(self.breakpoints_list[index][level-1]) + \
                            ' < ' + verbose_string
                    print(verbose_string)
                return level
            level += 1

        if verbose:
            print('Value > ' + str(value))
        return level

    def predict_row(self, df_row, verbose=False, threshold_choice=-1):
        index = 0
        cum_shap_value = 0

        for variable in self.variables:
            level = self.find_breakpoint_level(df_row[variable], index)
            shap_value = self.shap_array_list[index][level]
            cum_shap_value += shap_value

            if verbose:
                print(variable)
                print(level)
                print(self.shap_array_list[index][level])

            index += 1

        prob = expit(cum_shap_value)

        if threshold_choice == -1:
            threshold_p = expit(self.max_shap_score) / 2
        else:
            threshold_p = self.score_thresholds[threshold_choice]

        if prob >= threshold_p:
            prediction = 1
        else:
            prediction = 0
        return prob, prediction

    def predict(self, df, threshold_choice=-1):
        predictions = []
        for index, row in df.iterrows():
            prob, prediction = self.predict_row(row, threshold_choice)
            predictions.append(prediction)

        return np.array(predictions)

    def predict_proba(self, df):
        probs = []
        for index, row in df.iterrows():
            prob, prediction = self.predict_row(row)
            probs.append(prob)

        return np.array(probs)

    def predict_row_calculator(self, df_row, verbose=False, reference_zero = True):
        index = 0
        cum_score = 0
        cum_logodds = 0

        for variable in self.variables:
            if verbose:
                print("Variable:"+ variable)
                print("Variable value:" + str(df_row[variable]))

            level = self.find_breakpoint_level(df_row[variable], index, verbose=verbose)
            score = self.score_array_list[index][level]
            logodds = score * self.unit_shap_value

            if reference_zero == True:
                score += abs(min(self.score_array_list[index]))

            if verbose:
                print("Score: " + str(score))
                print("")

            index += 1
            cum_score += score
            cum_logodds += logodds

        prob = expit(cum_logodds + self.expected_value)

        return cum_score, prob

    def predict_calculator(self, df, threshold_choice=1, verbose=False):
        score_array = []
        prob_array = []
        for index, row in df.iterrows():
            score, prob = self.predict_row_calculator(row, verbose=False)
            score_array.append(score)
            prob_array.append(prob)
        score_array = np.array(score_array)
        prob_array = np.array(prob_array)

        return (
            score_array,
            prob_array,
            (score_array >= self.score_thresholds[threshold_choice]).astype(int)
        )

    def fit_calculator(self, calculator_threshold=0.05):
        index = 0
        for shap_array in self.shap_array_list:
            shap_array = np.array(shap_array)
            if index == 0 or np.min(shap_array) < self.unit_shap_value:
                shap_values_no_zeros = shap_array[np.abs(shap_array) >= calculator_threshold]
                self.unit_shap_value = np.min(np.absolute(shap_values_no_zeros))
            index += 1

        self.score_array_list = []
        for shap_array in self.shap_array_list:
            self.score_array_list.append(np.rint(np.true_divide(
                shap_array, self.unit_shap_value)))

    def generate_calculator_scores(self, reference_zero=True):
        neg_score_list = []
        pos_score_list = []

        for score_array in self.score_array_list:
            neg_score_list.append(np.min(score_array))
            pos_score_list.append(np.max(score_array))

        min_score = np.sum(neg_score_list)
        max_score = np.sum(pos_score_list)

        self.scoring_table = pd.DataFrame(columns = self.scoring_table_columns)
        self.score_thresholds = []

        i = min_score
        while i <= max_score:
            score = i
            if reference_zero == True:
                score += abs(min_score)

            prob = expit(i * self.unit_shap_value + self.expected_value)
            new_row = pd.DataFrame([[score, prob]],columns=self.scoring_table_columns)
            self.scoring_table = pd.concat([self.scoring_table, new_row])

            i += 1

        self.scoring_table = self.scoring_table.reset_index(drop=True)

        print("")
        print("")

        for p_threshold in self.p_thresholds:
            print("Probability threshold: " + str(p_threshold))
            score_threshold = self.scoring_table[self.scoring_table['Probability'] <= p_threshold]['Score'].max()
            self.score_thresholds.append(score_threshold)
            print("Score threshold: " + str(score_threshold))
            print("")

    # Prototype method
    def print_calculator(self):
        i = 0
        import math
        for variable in self.variables:
            shap_array = self.shap_array_list[i]
            breakpoints = self.breakpoints_list[i]
            shap_sd_array = self.shap_sd_array_list[i]
            shap_n_array = self.shap_n_array_list[i]
            
            min_shap_value = np.min(shap_array)
            min_shap_value_index = np.argmin(shap_array)
            min_shap_value_sd = shap_sd_array[min_shap_value_index]
            min_shap_value_n = shap_n_array[min_shap_value_index]
            
            j = 0
            for shap_value in shap_array:
                if j == 0:
                    upper_threshold = round(breakpoints[j+1], 2)
                    print(variable + "<=" + str(upper_threshold))
                elif j+1 < len(shap_array):
                    lower_threshold = round(breakpoints[j], 2)
                    upper_threshold = round(breakpoints[j+1], 2)
                    print(str(lower_threshold) + "<" 
                        + variable + "<=" 
                        + str(upper_threshold))
                else:
                    lower_threshold = round(breakpoints[j], 2)
                    print(str(lower_threshold) + "<" + variable)
                
                diff_shap_value = shap_value - min_shap_value
                odds_ratio = math.exp(diff_shap_value)
                odds_ratio = round(odds_ratio, 3)        
                print("Odds Ratio: " + str(odds_ratio))
        
                if j != min_shap_value_index:
                    shap_value_sd = shap_sd_array[j]
                    shap_value_n = shap_n_array[j]
                    sd = math.sqrt(((shap_value_n-1)*shap_value_sd + (min_shap_value_n-1)*min_shap_value_sd)/(shap_value_n+min_shap_value_n-2))
                    se = sd * math.sqrt(1/shap_value_n + 1/min_shap_value_n)
                    odds_ratio_upper = round(math.exp(diff_shap_value + se*1.96),3)
                    odds_ratio_lower = round(math.exp(diff_shap_value - se*1.96),3)
                    print("Confidence interval: " + str(odds_ratio_lower) + ", " + str(odds_ratio_upper))
                
                j+=1

            print()
            i+=1

    def plot_calculator_features(self, titles=None):
        shap_values_df = pd.DataFrame(self.shap_values, columns=self.X_train.columns)

        plt.figure(figsize=(16, 6))

        if not titles:
            titles = self.variables
        # titles = ["LOS (Days)", "RDW", "ICU LOS (Days)", "Age (years old)", "No. of Inotropes",
        #         "LDH (U/L)", "Haptoglobin (mg/dL)", "Phosphate (mmol/L)", "No. of ICU stays", "Albumin (g/dL)"]
        j = 0
        for variable in self.variables:
            # print(variable)
            x = self.X_train[variable].values
            y = shap_values_df[variable].values

            y = y[np.argsort(x)]
            x = x[np.argsort(x)]

            score_x = []
            score_y = []
            
            i = 0
            for score in self.shap_array_list[j]:
                score_x.append(self.breakpoints_list[j][i])
                score_y.append(score)
                score_x.append(self.breakpoints_list[j][i+1])
                score_y.append(score)
                i+=1
                
            xs = self.xs_array[j]
            ys = self.ys_array[j]

            #plt.subplots() 
            plt.subplot(2, 5, j+1)
            plt.scatter(x, y, s=2, c='lightblue')
            plt.plot(xs, ys, c='navy')
            plt.plot(score_x, score_y, c='gold')
            plt.axhline(0, linestyle='--', color='gray')
            plt.title(titles[j])
            j+=1

        plt.subplots_adjust(hspace=0.3)
        plt.show()
