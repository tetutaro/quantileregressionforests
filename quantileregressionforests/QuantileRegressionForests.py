#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class QuantileRegressionForests(RandomForestRegressor):
    '''
    Quantile Regression Forests の実装
    http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf

    QuantileRegressionForests は、ほとんど RandomForestRegressor だが、以下の違いがある
    1. predict() は、RandomForestRegressor だと各木の予測値の平均だが
    QuantileRegressionForestsは、各木の予測値を四分位数を用いた外れ値の除外を行った上での平均
    2. predict_avevar() という関数を新設し、外れ値の除外を行った上での平均と分散の両方を返すようにしている

    四分位数を用いた外れ値の除外
    Q_{1/4} : 下側四分位数
    Q_{3/4} : 上側四分位数
    IQR = Q_{3/4} - Q_{1/4} : 四分位範囲（interquartile range）
    [Q_{1/4} - (iqr_coef * IQR), Q_{3/4} + (iqr_coef * IQR)] 範囲外のデータを除外する
    （論文では [Q_{0.025}, Q_{0.975}] を用いている）
    一般的には iqr_coef は 1.5 だが
    （cf. https://ja.wikipedia.org/wiki/%E7%AE%B1%E3%81%B2%E3%81%92%E5%9B%B3）
    ここではデフォルトで iqr_coef = 1.2 とする
    '''
    def __init__(
        self,
        n_estimators=10,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
        warm_start=False,
        iqr_coef=1.2,
        exclude_side='both'
    ):
        '''
        param float iqr_coef: 外れ値判定を行うための四分位範囲に掛ける係数の値（デフォルト：1.2）
        param exclude_side: 'upper_only': 上側のみ、'lower_only': 下側のみ, 'both': 両方
        '''
        # iqr_coef , exclude_side 以外は RandomForestRegressor に投げる
        super(QuantileRegressionForests, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
        )
        # iqr_coef と exclude_area だけ自分で持つ
        self.iqr_coef = iqr_coef
        if exclude_side in ['upper_only', 'lower_only']:
            self.exclude_side = exclude_side
        else:
            self.exclude_side = 'both'

    def _predicted_values(self, X):
        '''
        RandomForestRegressorの各木での予測値に対して、
        外れ値の除外を行う内部関数

        :param array-like X: 特徴量
        :returns: 外れ値の除外を行った各木での予測値
        :rtype: matrix-like
        '''
        # 各木の予測結果を保持する配列
        predicted_list = list()
        # RandomForestRegressorのestimators_はsklearn.tree.DecisionTreeRegressorの配列
        for decision_tree_regressor in self.estimators_:
            # sklearn.tree.DecisionTreeRegressor.predict()を用いてデータに対する各木の予測値を得る
            predicted_list.append(
                decision_tree_regressor.predict(X)
            )
        predicted_list = np.array(predicted_list).T
        # 外れ値の除去
        # q1 : 下側四分位数、 q3 : 上側四分位数、 iqr : 四分位範囲に係数を掛けたもの　の、配列
        q1 = np.percentile(predicted_list, 25, axis=1)
        q3 = np.percentile(predicted_list, 75, axis=1)
        iqr = self.iqr_coef * (q3 - q1)
        # under_limits : 正常値の下限、 upper_limits : 正常値の上限　の、配列
        under_limits = q1 - iqr
        upper_limits = q3 + iqr
        # numpy の集計関数は、各行の要素数が同じでないと動かないので、削除するのではなく np.where を使って NaN に置き換える
        if self.exclude_side == 'upper_only':
            ret = np.array([
                np.where(values <= upper, values, np.nan)
                for values, upper
                in zip(predicted_list, upper_limits)
            ])
        elif self.exclude_side == 'lower_only':
            ret = np.array([
                np.where(under <= values, values, np.nan)
                for values, under
                in zip(predicted_list, under_limits)
            ])
        else:
            ret = np.array([
                np.where((under <= values) & (values <= upper), values, np.nan)
                for values, under, upper
                in zip(predicted_list, under_limits, upper_limits)
            ])
        return ret

    def predict(self, X):
        '''
        外れ値の除外を行った上での、各木の予測値の平均を返す関数

        :param matrix-like X: 特徴量
        :returns: 外れ値の除外を行った上での、各木予測値の平均（＝Quantile Regression Forests の予測値）
        :rtype: array-like
        '''
        predicted_values = self._predicted_values(X)
        # _predicted_values()がnanを含む値を返すため、nanを無視して計算するnumpy.nanmean()を用いる
        return np.nanmean(predicted_values, axis=1)

    def predict_avevar(self, X):
        '''
        各木の予測値の平均および分散を返す関数

        :param matrix-like X: 特徴量
        :returns: 外れ値の除外を行った上での、各木予測値の平均と分散
        :rtype: array-like, array-like
        '''
        predicted_values = self._predicted_values(X)
        # _predicted_values()がnanを含む値を返すため、nanを無視して計算するnumpy.nanmean(),nanvar()を用いる
        return (
            np.nanmean(predicted_values, axis=1),
            np.nanvar(predicted_values, axis=1)
        )
