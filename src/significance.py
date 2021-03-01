import pandas as pd
from scipy import stats
stl_df = pd.concat([pd.read_csv('model_version/seed_1/predictions.csv', header = None),pd.read_csv('model_version/seed_2/predictions.csv', header = None),pd.read_csv('model_version/seed_3/predictions.csv', header = None),pd.read_csv('model_version/seed_4/predictions.csv', header = None),pd.read_csv('model_version/seed_5/predictions.csv', header = None),pd.read_csv('model_version/seed_6/predictions.csv', header = None), pd.read_csv('model_version/seed_7/predictions.csv', header = None), pd.read_csv('model_version/seed_8/predictions.csv', header = None), pd.read_csv('model_version/seed_9/predictions.csv', header = None), pd.read_csv('model_version/seed_10/predictions.csv', header = None)])
mtl_df = pd.concat([pd.read_csv('model_version/seed_1/predictions.csv', header = None),pd.read_csv('model_version/seed_2/predictions.csv', header = None),pd.read_csv('model_version/seed_3/predictions.csv', header = None),pd.read_csv('model_version/seed_4/predictions.csv', header = None),pd.read_csv('model_version/seed_5/predictions.csv', header = None),pd.read_csv('model_version/seed_6/predictions.csv', header = None), pd.read_csv('model_version/seed_7/predictions.csv', header = None), pd.read_csv('model_version/seed_8/predictions.csv', header = None), pd.read_csv('model_version/seed_9/predictions.csv', header = None), pd.read_csv('model_version/seed_10/predictions.csv', header = None)])
mtl_emo_df = pd.concat([pd.read_csv('model_version/seed_1/predictions.csv', header = None),pd.read_csv('model_version/seed_2/predictions.csv', header = None),pd.read_csv('model_version/seed_3/predictions.csv', header = None),pd.read_csv('model_version/seed_4/predictions.csv', header = None),pd.read_csv('model_version/seed_5/predictions.csv', header = None),pd.read_csv('model_version/seed_6/predictions.csv', header = None), pd.read_csv('model_version/seed_7/predictions.csv', header = None), pd.read_csv('model_version/seed_8/predictions.csv', header = None), pd.read_csv('model_version/seed_9/predictions.csv', header = None),pd.read_csv('model_version/seed_10/predictions.csv', header = None)])
print('stl_df mtl_df', stats.pearsonr(stl_df[0], mtl_df[0])[0])
print('stl_df mtl_emo_df', stats.pearsonr(stl_df[0], mtl_emo_df[0])[0])
print('mtl_df mtl_emo_df', stats.pearsonr(mtl_df[0], mtl_emo_df[0])[0])

print('stl_df', stats.pearsonr(stl_df[0], stl_df[1])[0])
print('mtl_df', stats.pearsonr(mtl_df[0], mtl_df[1])[0])
print('mtl_emo_df', stats.pearsonr(mtl_emo_df[0], mtl_emo_df[1])[0])

sample_stl = stl_df.sample(2261)
sample_mtl = mtl_df.iloc[sample_stl.index]
sample_mtl_emo = mtl_emo_df.iloc[sample_stl.index]
print('sample_stl', stats.pearsonr(sample_stl[0], sample_stl[1])[0])
print('sample_mtl_emo', stats.pearsonr(sample_mtl_emo[0], sample_mtl_emo[1])[0])
print('sample_mtl', stats.pearsonr(sample_mtl[0], sample_mtl[1])[0])

print('sample_stl sample_mtl_emo', stats.pearsonr(sample_stl[0], sample_mtl_emo[0])[0])
print('sample_mtl sample_mtl_emo', stats.pearsonr(sample_mtl[0], sample_mtl_emo[0])[0])
print('sample_stl sample_mtl', stats.pearsonr(sample_stl[0], sample_mtl[0])[0])
