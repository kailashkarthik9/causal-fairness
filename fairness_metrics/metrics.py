import pandas as pd

from causal_discovery.relationship_finder import RelationshipsFinder


class Metrics:
    @staticmethod
    def demographic_parity(x: pd.DataFrame, y_hat: pd.DataFrame, sensitive_attributes: list):
        data = pd.concat([x, y_hat], axis=1)
        finder = RelationshipsFinder(data)
        return finder.get_conditional_distribution(['y_hat'], sensitive_attributes)

    @staticmethod
    def predictive_parity(x, y, y_hat, sensitive_attributes):
        data = pd.concat([x, y, y_hat], axis=1)
        finder = RelationshipsFinder(data)
        return finder.get_conditional_distribution(['y'], sensitive_attributes + ['y_hat'])

    @staticmethod
    def equalized_odds(x, y, y_hat, sensitive_attributes):
        data = pd.concat([x, y, y_hat], axis=1)
        finder = RelationshipsFinder(data)
        return finder.get_conditional_distribution(['y_hat'], sensitive_attributes + ['y'])
