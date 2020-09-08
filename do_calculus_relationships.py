import pandas as pd

from causal_discovery.relationship_finder import RelationshipsFinder
from classifier import get_dataframe
from datasets.dataset import Dataset


class DoCalculusRelationships:
    def __init__(self):
        dataset = Dataset('config_compas.json')
        x, y = dataset.get_data(readable=True)
        # r = "Af_vs_all"
        r = "Af_vs_Caucasian"
        # r = "all"
        x, y = get_dataframe(x, y, requested=r)
        self.finder = RelationshipsFinder(pd.concat([x, y], axis=1))

    def get_do_distributions(self):
        def adjust_race(x):
            return x['probability'] * marginal_race[marginal_race['race'] == x['race']]['probability'].item()

        def adjust_sex(x):
            return x['probability'] * marginal_sex[marginal_sex['sex'] == x['sex']]['probability'].item()

        do_conditional = self.finder.get_conditional_distribution(['is_recid'], ['race', 'sex'])
        marginal_race = self.finder.get_marginal_distribution(['race'])
        marginal_sex = self.finder.get_marginal_distribution(['sex'])
        do_sex = do_conditional.copy()
        do_sex['probability'] = do_sex.apply(adjust_race, axis=1)
        do_sex = do_sex.groupby(['is_recid', 'sex']).sum().reset_index()
        do_race = do_conditional.copy()
        do_race['probability'] = do_race.apply(adjust_sex, axis=1)
        do_race = do_race.groupby(['is_recid', 'race']).sum().reset_index()
        return do_conditional, do_race, do_sex


if __name__ == '__main__':
    docalc = DoCalculusRelationships()
    print(docalc.get_do_distributions())
