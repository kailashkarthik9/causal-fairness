import pandas as pd
from scipy.stats import bernoulli
from sklearn import svm
from sklearn.metrics import accuracy_score

from causal_discovery.relationship_finder import RelationshipsFinder
from fairness_metrics.metrics import Metrics


class SCM:
    @staticmethod
    def generate_sample():
        # All exogenous variables are from a bernoulli with p=0.6
        # A_1 = U_{A_1}
        # A_2 = U_{A_2}
        # V_5 = U_{V_5}
        # V_6 = U_{V_6}
        # V_2 = (A_1 or A_2) and U_{V_2}
        # V_1 = (A_1 and V_5 and V_2) or U_{V_1}
        # V_3 = V_1 and V_2 and U_{V_3}
        # V_4 = V_2 or V_3
        # Y = (A_1 and V_3) or (A_2 and V_4) or (A_1 and A_2 and U_Y)
        u_a_1, u_a_2, u_v_5, u_v_6, u_v_2, u_v_1, u_v_3, u_y = bernoulli(0.5).rvs(8)
        a_1 = u_a_1
        a_2 = u_a_2
        v_5 = u_v_5
        v_2 = (a_1 or a_2) and u_v_2
        # v_2 = 1 if (a_1 or a_2) and u_v_2 else -1
        v_1 = (a_1 and v_5 and v_2) or u_v_1
        # v_1 = 1 if (a_1 and v_5 and v_2) or u_v_1 else -1
        v_6 = u_v_6 ^ v_2
        # v_6 = 3 * u_v_6 - v_2
        v_3 = (v_1 ^ v_2) or u_v_3
        # v_3 = 2 * v_1 - 3 * v_2 + 5 * u_v_3
        v_4 = (v_2 and v_3) ^ v_6
        # v_4 = 3 * v_2 + 6 * v_3 - v_6
        y = (a_1 and v_3) ^ (a_2 and v_4) ^ (a_1 ^ a_2 ^ u_y)
        # y = 1 if 10*a_1 - a_2 + 5.6 * v_3 - 1.4 * v_4 + 5 > 0 else 0
        return {
            "a_1": a_1,
            'a_2': a_2,
            'v_1': v_1,
            'v_2': v_2,
            'v_3': v_3,
            'v_4': v_4,
            'v_5': v_5,
            'v_6': v_6,
            'y': y
        }

    @staticmethod
    def generate_dataset():
        samples = list()
        for idx in range(10000):
            sample = SCM.generate_sample()
            samples.append(sample)
        data = pd.DataFrame(data=samples, columns=['a_1', 'a_2', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'y'])
        return data


class SigmaFair:
    def __init__(self, data):
        self.classifier = svm.SVC()
        self.data = pd.read_csv(data)
        self.train_data = self.data.sample(frac=0.8, random_state=200)
        self.test_data = self.data.drop(self.train_data.index)

    def get_fair_data(self):
        train_data_ = self.train_data  # .drop(labels=['a_1', 'a_2', 'v_5'], axis=1)
        test_data_ = self.test_data  # .drop(labels=['a_1', 'a_2', 'v_5'], axis=1)
        for v_1 in [0, 1]:
            for v_2 in [0, 1]:
                yield train_data_[(train_data_.v_1 == v_1) & (train_data_.v_2 == v_2)], test_data_[
                    (test_data_.v_1 == v_1) & (test_data_.v_2 == v_2)]

    def train_fair_classifiers(self):
        for train, test in self.get_fair_data():
            train_x = train[['v_3', 'v_4', 'v_6']]
            train_y = train[['y']]
            test_x = test[['v_3', 'v_4', 'v_6']]
            test_y = test[['y']]
            self.classifier.fit(train_x, train_y)
            y_hat = self.classifier.predict(test_x)
            accuracy = accuracy_score(test_y, y_hat)
            y_hat = pd.DataFrame(y_hat).rename({0: 'y_hat'}, axis=1)
            dp = Metrics.demographic_parity(test, y_hat, ['a_1'])
            print('ok')

    def train_classifier(self):
        train_x = self.train_data[['a_1', 'a_2', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6']]
        train_y = self.train_data[['y']]
        test_x = self.test_data[['a_1', 'a_2', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6']]
        test_y = self.test_data[['y']]
        self.classifier.fit(train_x, train_y)
        y_hat = self.classifier.predict(test_x)
        accuracy = accuracy_score(test_y, y_hat)
        y_hat = pd.DataFrame(y_hat).rename({0: 'y_hat'}, axis=1)
        dp = Metrics.demographic_parity(test_x, y_hat, ['a_1'])
        print('ok')


if __name__ == '__main__':
    dataset = SCM.generate_dataset()
    dataset.to_csv('sample_dataset.csv', index=False)
    # plt.hist(dataset['y'])
    # plt.show()
    # plt.hist(dataset[dataset.a_1==0]['y'])
    # plt.show()
    # plt.hist(dataset[dataset.a_1 == 1]['y'])
    # plt.show()
    fair = SigmaFair('sample_dataset.csv')
    finder = RelationshipsFinder(fair.data)
    fair.train_classifier()
    fair.train_fair_classifiers()
    print('ok')
