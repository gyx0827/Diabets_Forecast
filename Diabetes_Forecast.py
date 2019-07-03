import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error


train = pd.read_csv('train_add.csv', encoding='utf-8')
test = pd.read_csv('test.csv', encoding='utf-8')
ol = pd.read_csv('answer.csv', encoding='gb2312')

traincopy = train.copy()
testcopy = test .copy()
train['血糖'] = np.log1p(train['血糖'])

# print(train['血糖'])

def make_feat(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])

    data['性别'] = data['性别'].map({'男':1,'女':0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017.10.09')).dt.days

    data.fillna(data.mean(axis = 0),inplace = True)

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat,test_feat


train_feat,test_feat = make_feat(train,test)

train_feat_copy,test_feat_copy = make_feat(traincopy,testcopy)
print(test_feat_copy.shape)

train_feat.drop(['id'],axis = 1,inplace = True)
test_feat.drop(['id'],axis = 1,inplace = True)

predictors = [f for f in test_feat.columns if f not in ['血糖']]


def eval_error(pred,df):
    labels = df.get_labels.values.copy()
    score = mean_squared_error(labels,pred) * 0.5
    return ('0.5mse', score, False)

print('开始训练...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.8,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

def lgb_train(train_feat,test_feat):
    print('开始CV 5折训练...')
    t0 = time.time()
    train_preds = np.zeros(train_feat.shape[0])
    test_preds = np.zeros((test_feat.shape[0],5))
    kf = KFold(len(train_feat),n_folds=5,shuffle=True,random_state=500)
    for i ,(train_index,test_index) in enumerate(kf):
        print('第{}次训练'.format(i))
        train_feat1 = train_feat.iloc[train_index]
        train_feat2 = train_feat.iloc[test_index]


        model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=8,
                                      learning_rate=0.05, n_estimators=400,
                                      max_bin=30, bagging_fraction=0.9,
                                      bagging_freq=10, feature_fraction=0.5,
                                      feature_fraction_seed=10, bagging_seed=10,
                                      min_data_in_leaf=80, nthread=8,
                                      min_sum_hessian_in_leaf=0.2)

        gbm = model_lgb.fit(train_feat1[predictors],train_feat1['血糖'],categorical_feature=['性别'])
        train_preds[test_index] += gbm.predict(train_feat2[predictors])
        test_preds[:,i] = gbm.predict(test_feat[predictors])
    return train_preds,test_preds

train_preds,test_preds = lgb_train(train_feat,test_feat)

train_preds = np.expm1(train_preds)
test_preds = np.expm1(test_preds)

for i in range(len(train_preds)):
    if train_preds[i] >= 6.4:
        train_preds[i] = np.expm1((train_preds[i] - 6.4) * 0.8) * 0.45 + train_preds[i]

test_preds1 = test_preds.mean(axis = 1)

for i in range(len(test_preds1)):
    if test_preds1[i] >= 6.4:
        # test_preds1[i] = ((test_preds1[i]-6.4)*1.4)**1.26 + 6.4
        test_preds1[i] = np.expm1((test_preds1[i] - 6.4) * 0.8)*0.45 + test_preds1[i]

print('线下得分：  {}'.format(mean_squared_error(train_feat_copy['血糖'],train_preds)*0.5))

print('线上的分：  {}'.format(mean_squared_error(ol['血糖'],test_preds1) * 0.5))
# print('线上得分：    {}'.format(mean_squared_error(ol['血糖'], test_preds1) * 0.5))

submission = pd.DataFrame({'pred':test_preds1})

if mean_squared_error(train_feat_copy['血糖'], train_preds) * 0.5 < 1.2:
    submission.to_csv(r'sub{}_{}.csv'.format(round(mean_squared_error(train_feat_copy['血糖'],train_preds)*0.5,3),
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,index=False, float_format='%.4f')
















