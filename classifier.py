import itertools

import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

from causal_discovery.relationship_finder import RelationshipsFinder
from datasets.dataset import Dataset
from fairness_metrics.metrics import Metrics


def classify(x: pd.DataFrame, y: pd.DataFrame, model_type='SVM', cv=False) :
    """
    Classification Function
    :param x: Feature set
    :param y: Labels
    :param model_type: Model type- SVM or Regressiion
    :param cv: cross validation or not
    :return: None
    """

    clf = svm.SVC() if model_type == 'SVM' else LogisticRegression()
    if cv :
        scores = cross_val_score(clf, x, y, cv=5)
        print("Model accuracies: ", scores)
        print("Average: ", round(sum(scores) / len(scores), 3))

    else :
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)
        model = clf.fit(x_train, y_train)
        print("Accuracy of "+model_type+" without CV: ", model.score(x_test, y_test))
        pred = clf.predict(x_test)
        return (x_test, y_test, pred)

    return


def get_dataframe(x, y, requested) :
    print("-----------------------------------------------------------------------------------------------------------")
    if requested == 'Af_vs_all' :
        print("Case: African American vs Rest\n")
        x.loc[x.race != 'African-American', 'race'] = 'Other'

    elif requested == "Af_vs_Caucasian" :
        print("Case: African American vs Caucasian\n")
        df = pd.concat([x, y], axis=1)
        df = df.loc[df['race'].isin(['African-American', 'Caucasian'])]
        y = pd.DataFrame(df['is_recid'])
        x = df.loc[:, df.columns != 'is_recid']
    else :
        print("All categories of race included\n")

    print(x['race'].value_counts())
    print("-----------------------------------------------------------------------------------------------------------")

    return x, y


def generate_independencies(x, y) :
    finder = RelationshipsFinder(pd.concat([x, y], axis=1))

    tolerances = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075,
                  0.08, 0.085, 0.09, 0.095, 0.1]
    for tolerance in tolerances :
        print("Tolerance: ", tolerance)
        finder.find_relationships_difference(tolerance)
        print("\n")


def get_conditional_distribution(df, vars) :
    """
    Function to get conditional distributions on a set of variables
    :param df: dataframe
    :param vars: variables to condition on
    :return: possible combinations of variable values, stratified dataframes
    """
    train_dfs = []
    if not vars :
        return [], df

    if len(vars) > 1 :
        var_values = []
        for v in vars :
            values = list(df[v].unique())
            var_values.append(values)

        combos = list(itertools.product(*var_values))
        print(combos)

        for combo in combos :
            var_combo_zip = list(zip(vars, combo))
            q = " and ".join([(str(var)+'=='+str(val)) for (var, val) in var_combo_zip])
            cdf = df.query(q)
            train_dfs.append(cdf)
    else :
        var_values = list(df[vars[0]].unique())
        for val in var_values :
            q = str(vars[0])+'=='+str(val)
            cdf = df.query(q)
            train_dfs.append(cdf)
        combos = var_values

    return combos, train_dfs

def main() :
    dataset = Dataset('config_compas.json')
    x, y = dataset.get_data(readable=True)

    # r = "Af_vs_all"
    r = "Af_vs_Caucasian"
    # r = "all"

    x, y = get_dataframe(x, y, requested=r)

    x_encoded, y_encoded = dataset.encode_data(x_df=x, y_df=y)

    # experimental setup
    # sex _|_ race | priors count

    # modify this to our experiment cases
    conditional_variables = ['race']
    sensitive_attributes = ['sex']
    combos, train_dfs = get_conditional_distribution(pd.concat((x_encoded, y_encoded), axis=1), conditional_variables)
    m = Metrics()

    if not combos :
        y_train = train_dfs['is_recid'].copy()
        x_train = train_dfs.drop('is_recid', axis=1).copy()

        x_test, y_test, pred = classify(x_train, y_train)
        print(x_test, y_test)
        df = pd.concat((x_test, y_test), axis=1).dropna().reset_index()
        y_hat = pd.DataFrame(pred).rename({0 : 'y_hat'}, axis=1)

        dp = m.demographic_parity(x=df, y_hat=y_hat, sensitive_attributes=sensitive_attributes)
        print(dp)
        pp = m.predictive_parity(x=x_test.reset_index(drop=True), y=y_test.reset_index(drop=True), y_hat=y_hat,
                                 sensitive_attributes=sensitive_attributes)
        eo = m.equalized_odds(x=x_test.reset_index(drop=True), y=y_test.reset_index(drop=True), y_hat=y_hat,
                              sensitive_attributes=sensitive_attributes)
        print(pp)
        print(eo)

    for combo, strat_df in list(zip(combos, train_dfs)) :
        combo, strat = list(zip(combos, train_dfs))[-1]
        print("Conditional Attributes : "+str(conditional_variables))
        # print("Sensitive attributes : "+str(sensitive_attributes))
        print(" Combination: "+str(vars)+" "+str(combo))
        y_train = strat_df['is_recid'].copy()
        x_train = strat_df.drop('is_recid', axis=1).copy()

        x_test, y_test, pred = classify(x_train, y_train)

        df = pd.concat((x_test, y_test), axis=1).dropna().reset_index()
        y_hat = pd.DataFrame(pred).rename({0 : 'y_hat'}, axis=1)

        y_test.name = 'y'
        dp = m.demographic_parity(x=df, y_hat=y_hat, sensitive_attributes=sensitive_attributes)
        pp = m.predictive_parity(x=x_test.reset_index(drop=True), y=y_test.reset_index(drop=True), y_hat=y_hat,
                                 sensitive_attributes=sensitive_attributes)
        eo = m.equalized_odds(x=x_test.reset_index(drop=True), y=y_test.reset_index(drop=True), y_hat=y_hat,
                              sensitive_attributes=sensitive_attributes)

        print(" Demographic Pariy: ")
        print(dp)
        # print(" Preditive Parity: ")
        # print(pp)
        # print(" Equilized Odds: ")
        #print(eo)

if __name__ == '__main__' :
    main()
