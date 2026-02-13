from hossam import *
from pandas import DataFrame, merge
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
from itertools import product

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.inspection import permutation_importance

from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error)

from sklearn.ensemble import RandomForestRegressor

import shap

my_dpi=100

def hs_get_scores(estimator, x_test, y_true):
    if hasattr(estimator, 'named_steps'):
        classname = estimator.named_steps['model'].__class__.__name__
    else:
        classname = estimator.__class__.__name__

    y_pred = estimator.predict(x_test)

    return DataFrame(
        {
            "결정계수(R2)": r2_score(y_true, y_pred),
            "평균절대오차(MAE)": mean_absolute_error(y_true, y_pred),
            "평균제곱오차(MSE)": mean_squared_error(y_true, y_pred),
            "평균오차(RMSE)": np.sqrt(mean_squared_error(y_true, y_pred)),
            "평균 절대 백분오차 비율(MAPE)": mean_absolute_percentage_error(
                y_true, y_pred
            ),
            "평균 비율 오차(MPE)": np.mean((y_true - y_pred) / y_true * 100),
        },
        index=[classname],
    )

def hs_learning_cv(
    estimator,
    x,
    y,
    scoring="neg_root_mean_squared_error",
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
):
    train_sizes, train_scores, cv_scores = learning_curve(
        estimator=estimator,
        X=x,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        shuffle=True,
        random_state=52,
    )

    if hasattr(estimator, 'named_steps'):
        classname = estimator.named_steps['model'].__class__.__name__
    else:
        classname = estimator.__class__.__name__

    # neg RMSE -> RMSE
    train_rmse = -train_scores
    cv_rmse = -cv_scores

    # 평균 / 표준편차
    train_mean = train_rmse.mean(axis=1)
    cv_mean = cv_rmse.mean(axis=1)
    cv_std = cv_rmse.std(axis=1)

    # 마지막 지점 기준 정량 판정
    final_train = train_mean[-1]
    final_cv = cv_mean[-1]
    final_std = cv_std[-1]
    gap_ratio = final_train / final_cv
    var_ratio = final_std / final_cv

    # ------------------
    # 과소적합 기준선 (some_threshold)
    # ------------------
    # 기준모형 RMSE (평균 예측)
    y_mean = y.mean()
    rmse_naive = np.sqrt(np.mean((y - y_mean) ** 2))

    # 분산 기반
    std_y = y.std()

    # 최소 설명력(R^2) 기반
    min_r2 = 0.10
    rmse_r2 = np.sqrt((1 - min_r2) * np.var(y))

    # 최종 threshold (가장 관대한 기준)
    # -> 원래 some_threshold는 도메인 지식 수준에서 이 모델은 최소 어느 정도의 성능은 내야 한다는 기준을 설정하는 것
    some_threshold = min(rmse_naive, std_y, rmse_r2)

    # ------------------
    # 판정 로직
    # ------------------
    if gap_ratio >= 0.95 and final_cv > some_threshold:
        status = "⚠️ 과소적합 (bias 큼)"
    elif gap_ratio <= 0.8:
        status = "⚠️ 과대적합 (variance 큼)"
    elif gap_ratio <= 0.95 and var_ratio <= 0.10:
        status = "✅ 일반화 양호"
    elif var_ratio > 0.15:
        status = "⚠️ 데이터 부족 / 분산 큼"
    else:
        status = "⚠️ 판단 유보"

    # ------------------
    # 정량 결과 표
    # ------------------
    result_df = DataFrame(
        {
            'Train RMSE':[final_train],
            'CV RMSE 평균':[final_cv],
            'CV RMSE 표준편차':[final_std],
            'Train/CV 비율':[gap_ratio],
            'CV 변동성 비율':[var_ratio],
            '판정 결과':[status]
        },
        index=[classname]
    )

    # display(result_df)

    # ------------------
    # 학습곡선 시각화
    # ------------------
    figsize = (1280 / my_dpi, 720 / my_dpi)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=my_dpi)

    sb.lineplot(
        x=train_sizes,
        y=train_mean,
        marker="o",
        markeredgecolor="#ffffff",
        label="Train RMSE",
    )
    
    sb.lineplot(
        x=train_sizes,
        y=cv_mean,
        marker="o",
        markeredgecolor="#ffffff",
        label="CV RMSE",
    )

    ax.set_title('학습곡선 (Learning Curve)', fontsize=12, pad=8)
    ax.set_xlabel("학습 데이터 비율", fontsize=8, labelpad=5)
    ax.set_ylabel("RMSE", fontsize=8, labelpad=5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()

    return result_df

def hs_get_score_cv(estimator, x_test, y_test, x_origin, y_origin, scoring='neg_root_mean_squared_error', cv=5, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1):
    score_df=hs_get_scores(estimator, x_test, y_test)
    cv_df=hs_learning_cv(estimator, x_origin, y_origin, scoring=scoring, cv=cv, train_sizes=train_sizes, n_jobs=n_jobs)
    return merge(score_df, cv_df, left_index=True, right_index=True)

def feature_importance(model, x_train, y_train):
    perm=permutation_importance(
        estimator=model,
        X=x_train,
        y=y_train,
        scoring='r2',
        n_repeats=30,
        random_state=42,
        n_jobs=-1
    )

    perm_df=DataFrame(
        {
            'importance_mean':perm.importances_mean,
            'importance_std':perm.importances_std
        },
        index=x_train.columns
    ).sort_values('importance_mean', ascending=False)

    df=perm_df.sort_values(by='importance_mean', ascending=False)

    figsize=(1280/my_dpi, 600/my_dpi)
    fig, ax=plt.subplots(1, 1, figsize=figsize, dpi=my_dpi)

    sb.barplot(data=df, x='importance_mean', y=df.index)

    ax.set_title('Permutation Importance')
    ax.set_xlabel('ermutation Importance (mean)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()

    return perm_df

def hs_shap_analysis(model, x: DataFrame, plot: bool = True, width: int = 1600, height: int = 800):
    # 1. SHAP Explainer
    explainer = shap.TreeExplainer(model)

    # 2. SHAP 값 계산: shape = [n_samples, n_features]
    shap_values = explainer.shap_values(x)

    # 3. DataFrame 변환
    shap_df = DataFrame(
        shap_values,
        columns=x.columns,
        index=x.index,
    )

    # 4. 요약 통계
    summary_df = DataFrame(
        {
            "feature": shap_df.columns,
            "mean_abs_shap": shap_df.abs().mean().values,
            "mean_shap": shap_df.mean().values,
            "std_shap": shap_df.std().values,
        }
    )

    # 5. 영향 방향 (보수적 표현)
    summary_df["direction"] = np.where(
        summary_df["mean_shap"] > 0,
        "양(+) 경향",
        np.where(summary_df["mean_shap"] < 0, "음(-) 경향", "혼합/미약"),
    )

    # 6. 변동성 지표
    summary_df["cv"] = (
        summary_df["std_shap"]
        / (summary_df["mean_abs_shap"] + 1e-9)
    )

    summary_df["variability"] = np.where(
        summary_df["cv"] < 1,
        "stable",    # 변동성 낮음 - 평균 대비 일관적 영향 의미
        "variable",  # 변동성 큼 - 상황 의존적 영향 의미
    )

    # 7. 중요도 기준 정렬
    summary_df = (
        summary_df
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # 8. 주요 변수 표시 (누적 80%)
    total_importance = summary_df["mean_abs_shap"].sum()
    summary_df["importance_ratio"] = (
        summary_df["mean_abs_shap"] / total_importance
    )
    summary_df["importance_cumsum"] = summary_df["importance_ratio"].cumsum()

    summary_df["is_important"] = np.where(
        summary_df["importance_cumsum"] <= 0.80,
        "core",      # 누적 80% 내 중요 변수 - 모델 핵심 결정 요인 의미 명확
        "secondary", # 누적 80% 초과 변수 - 보조적/상황적 영향 요인 의미
    )

    # 9. 시각화
    if plot:
        shap.summary_plot(shap_values, x, show=False)

        fig = plt.gcf()
        fig.set_size_inches(width / my_dpi, height / my_dpi)

        plt.title("SHAP Summary Plot", fontsize=10, pad=10)
        plt.xlabel("SHAP value", fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=8)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.close()

    return summary_df, shap_values

def hs_shap_dependence_analysis(
    summary_df: DataFrame,
    shap_values,
    x_train: DataFrame,
    include_secondary: bool = False,
    width: int = 1600,
    height: int = 800,
):
    # 1. 주 대상 변수 (Core + Variable)
    main_features = summary_df[
        (summary_df["is_important"] == "core")
        & (summary_df["variability"] == "variable")
    ]["feature"].tolist()

    # 2. 상호작용 후보 변수
    interaction_features = summary_df[
        summary_df["is_important"] == "core"
    ]["feature"].tolist()

    if include_secondary and len(interaction_features) < 2:
        interaction_features.extend(
            summary_df[
                summary_df["is_important"] == "secondary"
            ]["feature"].tolist()
        )

    # 3. 변수 쌍 생성 (자기 자신 제외)
    pairs = []
    for f in main_features:
        for inter in interaction_features:
            # 자기 자신과의 조합은 제외
            if f != inter:
                pairs.append((f, inter))

    # 중요도 순 정렬 (주 변수 기준)
    importance_rank = {}
    for i, row in summary_df.iterrows():
        importance_rank[row["feature"]] = i

    pairs = sorted(
        pairs,
        key=lambda x: importance_rank.get(x[0], 999)
    )

    # 4. dependence plot 일괄 생성
    for feature_name, interaction_name in pairs:
        shap.dependence_plot(
            feature_name,
            shap_values,
            x_train,
            interaction_index=interaction_name,
            show=False,
        )

        # SHAP figure 직접 제어
        fig = plt.gcf()
        fig.set_size_inches(width / my_dpi, height / my_dpi)

        plt.title(
            f"SHAP Dependence Plot: {feature_name} x {interaction_name}",
            fontsize=10,
            pad=10,
        )
        plt.xlabel(feature_name, fontsize=10)
        plt.ylabel(f"SHAP value for {feature_name}", fontsize=10)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=8)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.close()

    return pairs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.inspection import permutation_importance

def hs_feature_importance(model, x_train, y_train, my_dpi=100):
    # 1. 모델 타입에 따른 중요도 산출
    if isinstance(model, XGBRegressor):
        booster = model.get_booster()
        imp = booster.get_score(importance_type="gain")
        imp_sr = pd.Series(imp)
        imp_df = pd.DataFrame(imp_sr, columns=['importance'])
    elif isinstance(model, LGBMRegressor):
        booster=model.booster_
        imp=booster.feature_importance(importance_type='gain')
        imp_df=DataFrame({'importance':imp}, index=model.feature_name_)
    elif isinstance(model, CatBoostRegressor):
        imp=model.get_feature_importance(type='FeatureImportance')
        imp_df=DataFrame({'importance':imp}, index=model.feature_names_)
    else:
        # permutation_importance 사용 시 (best_model이 외부 변수일 경우를 대비해 model 사용)
        result = permutation_importance(
            estimator=model, 
            X=x_train, 
            y=y_train,
            scoring="r2",
            n_repeats=30,
            random_state=42,
            n_jobs=-1
        )
        # 결과 정리
        imp_df = pd.DataFrame({'importance': result.importances_mean}, index=x_train.columns)

    # 2. 중요도 비율(ratio) + 누적 중요도(cumsum) 계산
    # 전체 합이 0일 경우를 대비해 sum()이 0이 아닐 때만 계산
    total_imp = imp_df["importance"].sum()
    imp_df["ratio"] = imp_df["importance"] / total_imp if total_imp != 0 else 0
    
    # ratio 기준으로 내림차순 정렬
    imp_df.sort_values("ratio", ascending=False, inplace=True)
    imp_df["cumsum"] = imp_df["ratio"].cumsum()

    # 3. 시각화 준비
    df = imp_df.copy()
    threshold = 0.9

    height = len(df) * 0.6  # 막대 개수에 따라 높이 조절
    figsize = (1280 / 100, height if height > 5 else 5)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=my_dpi)

    # 바 차트 생성
    sb.barplot(data=df, x="importance", y=df.index, ax=ax)

    # 4. 값 라벨 추가 (f-string 따옴표 수정 완료)
    for i, v in enumerate(df["importance"]):
        # 바깥쪽을 큰따옴표(")로 감싸서 내부의 ['cumsum']과 충돌 방지
        label_text = f"{v:.1f} ({df.iloc[i]['cumsum']*100:.1f}%)"
        ax.text(v + (df["importance"].max() * 0.01), i, label_text, va="center")

    ax.set_title("Feature Importance", fontsize=12, pad=8)
    ax.set_xlabel("importance (cumsum)", fontsize=12)
    ax.set_ylabel(None)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, df["importance"].max() * 1.3) # 텍스트 공간 확보

    # 5. 90% 임계치 수직선(axhline) 표시
    # 누적합이 threshold를 넘는 첫 번째 인덱스 찾기
    indices = np.where(df["cumsum"].values >= threshold)[0]
    if len(indices) > 0:
        cut_idx = indices[0]
        cut_rank = float(cut_idx) + 0.5
        plt.axhline(y=cut_rank, linestyle=":", color="red", alpha=0.8)

    plt.tight_layout()
    plt.show()
    plt.close()

    return imp_df