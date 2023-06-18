import json
import pandas as pd

from tqdm import tqdm


def load_all():
    """
    Reads split of NYT-FB distant supervision data and creates metadata dictionary.

    :return: None, writes metadata dict.
    """
    df_1 = pd.read_csv('training_data.csv')
    df_2 = pd.read_csv('testing_data.csv')
    df_3 = pd.read_csv('validation_data.csv')
    all_df = pd.concat([df_1, df_2, df_3])
    all_df = all_df.fillna("Missing").replace('nan', 'Missing')
    obs = list(set(all_df['object_string'].tolist()))
    for x in obs:
        try:
            if len(x) < 2:
                print(x)
        except TypeError:
            print(x)
    sent_rel_mapper = all_df[['relation', 'rel_idx', 'sent_id', 'sentence',
                              'subject_string', 'object_string']]
    sent_rel_mapper.sort_values(by=['sent_id'], inplace=True, ignore_index=True)
    sent_rel_mapper.drop_duplicates(inplace=True, ignore_index=True)
    _metadict = {}
    for _, row in tqdm(sent_rel_mapper.iterrows()):
        try:
            if row['relation'] == 'NA':
                rel = 'NoRelation'
            else:
                rel = row['relation']
            _metadict[row['sent_id']]['relations'].append(rel)
            _metadict[row['sent_id']]['rel_idx'].append(str(row['rel_idx']))
            _metadict[row['sent_id']]['subjects'].append(row['subject_string'])
            _metadict[row['sent_id']]['objects'].append(row['object_string'])
        except KeyError:
            if row['relation'] == 'NA':
                rel = 'NoRelation'
            else:
                rel = row['relation']
            _metadict[row['sent_id']] = {'sentence': row['sentence'],
                                         'relations': [rel],
                                         'rel_ids': [str(row['rel_idx'])],
                                         'subjects': [row['subject_string']],
                                         'objects': [row['object_string']]}

    print('Found metadata for {} elements'.format(len(list(_metadict.keys()))))
    with open('nytfb_metadata.json', 'w') as f:
        json.dump(_metadict, f)


if __name__ == "__main__":
    load_all()
