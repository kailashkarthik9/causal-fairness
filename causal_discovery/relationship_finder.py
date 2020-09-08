import itertools
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


class RelationshipsFinder:
    """
    Class to find Conditional Independence from Probability Distributions using three strategies
    1. Evaluate conditional distributions and check for equality
    2. Evaluate the difference between conditional and marginal distribution
    3. Fischer's method
    *** Copied from Causal Homework - need to make it generic to work on both the datasets
    """

    def __init__(self, data_frame):
        if isinstance(data_frame, pd.DataFrame):
            self.data_file = data_frame
        else:
            self.data_file = pd.read_csv(data_frame)
        self.variables = self.data_file.columns
        self.probabilities = self.compute_probabilities()

    def compute_probabilities(self):
        """
        Method to compute probabilities from raw data
        :return: DataFrame of probabilites
        """

        """
        Has only probabilities of occurance in the data file
        """
        probabilities = self.data_file.groupby(list(self.data_file.columns)).size().reset_index(name='count')
        total_count = len(self.data_file)
        probabilities['probability'] = probabilities['count'].apply(lambda x: round(x / total_count, 5))

        """
        Generate cartesian product and combine with above probabilities- generate all possible combinations and 
        initialize probability to 0
        """
        # probabilities = probabilities.drop('count', axis=1)
        uniques = [self.data_file[i].unique().tolist() for i in self.data_file.columns]
        cartesian = pd.DataFrame(product(*uniques), columns=self.data_file.columns)
        probabilities = cartesian.merge(probabilities, left_on=list(cartesian.columns),
                                        right_on=list(cartesian.columns), how='left')
        probabilities['probability'].fillna(0, inplace=True)

        return probabilities

    def get_marginal_distribution(self, variables):
        """
        Method to marginalize probability distribution
        :param variables: Variables to be present in the marginal
        :return: Marginal distribution
        """
        if not variables:
            return self.probabilities
        marginalized = self.probabilities.groupby(variables, as_index=False).agg({'probability': ['sum']})
        marginalized.columns = variables + ['probability']
        return marginalized

    def get_marginal_counts(self, variables):
        """
        Method to marginalize count distribution
        :param variables: Variables to be present in the marginal
        :return: Marginal counts
        """
        if not variables:
            return self.probabilities
        marginalized = self.probabilities.groupby(variables, as_index=False).agg({'count': ['sum']})
        marginalized.columns = variables + ['count']
        return marginalized

    def get_conditional_distribution(self, variables, conditioned_on):
        """
        Method to get conditional distribution
        """
        joint_probability = self.get_marginal_distribution(variables + conditioned_on)
        if not conditioned_on:
            return joint_probability
        normalization_distribution = self.get_marginal_distribution(conditioned_on)
        joint_probability['conditional_probability'] = joint_probability.apply(
            lambda x: self.get_normalized_probability(x, normalization_distribution, conditioned_on), axis=1)
        joint_probability = joint_probability.drop(['probability'], axis=1)
        joint_probability.columns = list(joint_probability.columns)[:-1] + ['probability']
        return joint_probability

    @staticmethod
    def get_normalized_probability(term_to_normalize, norm_dist, norm_vars):
        """
        Method to get normalized probability
        :param term_to_normalize: Term to noramlize
        :param norm_dist: Normalized distribution
        :param norm_vars: Normalized variables
        """
        for variable in norm_vars:
            norm_dist = norm_dist[norm_dist[variable] == term_to_normalize[variable]]
        return round(term_to_normalize['probability'] / norm_dist['probability'].values[0], 5)

    @staticmethod
    def multiply_distributions(dist_1, dist_2):
        """
        Method to multiple distributions
        """
        dist_1_columns = list(dist_1.columns)[:-1]
        dist_2_columns = list(dist_2.columns)[:-1]
        common_cols = [col for col in dist_1_columns if col in dist_2_columns]
        join_cols_dist_1 = [col for col in dist_1_columns if col not in dist_2_columns]
        join_cols_dist_2 = [col for col in dist_2_columns if col not in dist_1_columns]
        new_dist = pd.DataFrame(columns=common_cols + join_cols_dist_1 + join_cols_dist_2 + ['probability'])
        idx = 0
        for _, dist_1_row in dist_1.iterrows():
            matching_dist_2 = dist_2
            for col in common_cols:
                matching_dist_2 = matching_dist_2[matching_dist_2[col] == dist_1_row[col]]
            for _, dist_2_row in matching_dist_2.iterrows():
                new_dist_row = [dist_1_row[col] for col in common_cols]
                new_dist_row.extend([dist_1_row[col] for col in join_cols_dist_1])
                new_dist_row.extend([dist_2_row[col] for col in join_cols_dist_2])
                new_dist_row.append(round(dist_1_row['probability'] * dist_2_row['probability'], 5))
                new_dist.loc[idx] = new_dist_row
                idx += 1
        for col in common_cols:
            new_dist[col] = new_dist[col].astype(np.int64)
        for col in join_cols_dist_1:
            new_dist[col] = new_dist[col].astype(np.int64)
        for col in join_cols_dist_2:
            new_dist[col] = new_dist[col].astype(np.int64)
        return new_dist

    @staticmethod
    def find_subsets_of_size(list_, size):
        """
        Method to generate all subsets of the list of a given size
        """
        return list(map(list, itertools.combinations(list_, size)))

    def find_relationships(self):
        """
        Method to find all the independence relationships in the distribution
        :return: None - Prints all the conditional independence
        """
        for conditioning_list_size in range(len(self.variables) - 1):
            conditioning_list_of_size = self.find_subsets_of_size(self.variables, conditioning_list_size)
            for conditioning_list in conditioning_list_of_size:
                remaining_vars = [var for var in self.variables if var not in conditioning_list]
                for var_pair in self.find_subsets_of_size(remaining_vars, 2):
                    joint_dist = self.get_conditional_distribution(var_pair, conditioning_list)
                    var_1_dist = self.get_conditional_distribution(list(var_pair[0]), conditioning_list)
                    var_2_dist = self.get_conditional_distribution(list(var_pair[1]), conditioning_list)
                    product_dist = self.multiply_distributions(var_1_dist, var_2_dist)
                    if self.check_distribution_equality(joint_dist, product_dist):
                        print('Independent ' + str(var_pair) + ' | ' + str(conditioning_list))

    @staticmethod
    def check_distribution_equality(dist_1: pd.DataFrame, dist_2):
        """
        Method to check if two distributions are equal
        """
        columns = list(dist_1.columns)
        dist_2 = dist_2[columns]
        dist_1_probs = dist_1.sort_values(by=columns)['probability'].values
        dist_2_probs = dist_2.sort_values(by=columns)['probability'].values
        return np.allclose(dist_1_probs, dist_2_probs, atol=0.025)

    def find_relationships_fischer(self):
        """
        Method to find all the independence relationships in the distribution using Fischer's method
        :return: None - Prints all the conditional independence
        """
        for conditioning_list_size in range(len(self.variables) - 1):
            conditioning_list_of_size = self.find_subsets_of_size(self.variables, conditioning_list_size)
            for conditioning_list in conditioning_list_of_size:
                remaining_vars = [var for var in self.variables if var not in conditioning_list]
                for var_pair in self.find_subsets_of_size(remaining_vars, 2):
                    joint_counts = self.get_marginal_counts(var_pair + conditioning_list)
                    if conditioning_list:
                        joint_counts.sort_values(by=conditioning_list + var_pair, inplace=True)
                    var_pair_combinations = 1
                    for var in var_pair:
                        var_pair_combinations *= len(set(joint_counts[var]))
                    if self.check_independence_fischer(joint_counts, var_pair_combinations):
                        print('Independent ' + str(var_pair) + ' | ' + str(conditioning_list))

    @staticmethod
    def check_independence_fischer(distribution, var_pair_combinations):
        condition_categories = distribution.shape[0] // var_pair_combinations
        counts = list(distribution['count'].values)
        for idx in range(condition_categories):
            category_probabilities = counts[idx * var_pair_combinations: (idx + 1) * var_pair_combinations]
            _, p_value = fisher_exact([category_probabilities[0:2], category_probabilities[2:4]])
            if p_value > 0.05:
                return False
        return True

    def find_relationships_difference(self, tolerance) :
        """
        Method to find all the independence relationships in the distribution using the probability differences
        :return: None - Prints all the conditional independence
        """
        for conditioning_list_size in range(len(self.variables)-1) :
            conditioning_list_of_size = self.find_subsets_of_size(self.variables, conditioning_list_size)
            for conditioning_list in conditioning_list_of_size :
                remaining_vars = [var for var in self.variables if var not in conditioning_list]
                for var_pair in self.find_subsets_of_size(remaining_vars, 2) :
                    dist_without_var = self.get_conditional_distribution(var_pair[:1], conditioning_list)
                    dist_with_var = self.get_conditional_distribution(var_pair[:1], conditioning_list + [var_pair[1]])
                    if self.check_distribution_equality_ignoring_var(dist_without_var, dist_with_var, tolerance) :
                        print(str(var_pair[0])+' _||_ '+str(var_pair[1])+' | '+str(conditioning_list))

    @staticmethod
    def check_distribution_equality_ignoring_var(dist_without_var: pd.DataFrame, dist_with_var: pd.DataFrame,
                                                 tolerance) :
        """
        Method to check if distributions are equal ignoring a variable of interest
        """
        columns = list(dist_without_var.columns)
        columns.remove('probability')
        for _, row in dist_without_var.iterrows() :
            filtered_dist = dist_with_var
            for col in columns :
                filtered_dist = filtered_dist[filtered_dist[col] == row[col]]
            for _, filtered_row in filtered_dist.iterrows() :
                if abs(row['probability']-filtered_row['probability']) > tolerance :
                    return False
        return True


if __name__ == '__main__':
    finder = RelationshipsFinder('~/stuff/causalInference/causalFairness-Spring20/unit_tests/truncated_hw1.csv')
    finder.find_relationships_fischer()
