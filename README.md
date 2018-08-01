# Quantile Regression Forests

Meinshausen, Nicolai. "Quantile regression forests." Journal of Machine Learning Research 7.Jun (2006): 983-999.

上記 Quantile Regression Forests の独自実装。

# インストール方法

`> pip install git+https://github.com/tetutaro/quantileregressionforests

# Quantile Regressoin Forests

Quantile Regression Forests は、ほとんど RandomForestRegressor だが、以下の違いがある
1. predict() は、RandomForestRegressor だと各木の予測値の平均だがQuantileRegressionForestsは、各木の予測値を四分位数を用いた外れ値の除外を行った上での平均
2. predict\_avevar() という関数を新設し、外れ値の除外を行った上での平均と分散の両方を返すようにしている

# 四分位数を用いた外れ値の除外

* Q\_{1/4} : 下側四分位数
* Q\_{3/4} : 上側四分位数
* IQR = Q\_{3/4} - Q\_{1/4} : 四分位範囲（interquartile range）
* [Q\_{1/4} - (iqr\_coef * IQR), Q\_{3/4} + (iqr\_coef * IQR)] 範囲外のデータを除外する
    * （論文では [Q\_{0.025}, Q\_{0.975}] を用いている）
* 一般的には iqr\_coef は 1.5 だが、ここではデフォルトで iqr\_coef = 1.2 とする
（cf. https://ja.wikipedia.org/wiki/%E7%AE%B1%E3%81%B2%E3%81%92%E5%9B%B3）

# オプション

ほとんどRandom Forestと同じように使えるが、追加したオプションがある。

* `iqr_coef`: 上記四分位数の係数（default: 1.2）
* `exclude_side`: 上側のみを削除（`upper_only`）、下側のみを削除（`lower_only`）、両方（`both`）（default: `both`）


