import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Константы
RANDOM_STATE = 42
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50
PATIENCE = 10
VALIDATION_FOLD = 2  # Фолд для валидации
TOP_FEATURES = 100  # Количество оставляемых признаков

# 1. Загрузка данных
train = pd.read_parquet('train_main_df.parquet')
target = pd.read_csv('train_target.csv')
test = pd.read_parquet('test_main_df.parquet')
train_card = pd.read_parquet('train_card_spending_df.parquet')
test_card = pd.read_parquet('test_card_spending_df.parquet')

# 2. Объединение и подготовка данных
data = train.merge(target, on='id')

# Фильтрация train_card
selected_card_features = [
    # Базовые финансовые показатели
    'sum_tr_all_1', 'sum_tr_all_3', 'sum_tr_all_6',
    'max_tr_pay_1', 'max_tr_pay_3',
    'max_tr_top_up_1',

    # Ключевые категории расходов
    'sum_tr_food_retail_1', 'sum_tr_food_retail_3',
    'sum_tr_publ_util_1m', 'sum_tr_publ_util_3m',
    'sum_tr_relax_1m',
    'sum_tr_internet_1'
]

# Обработка тренировочных данных
train_card_filtered = train_card[['id'] + [col for col in selected_card_features if col in train_card.columns]]
train_card_filtered = train_card_filtered.drop_duplicates(subset=['id'])
data = data.merge(train_card_filtered, on='id', how='left')

# Обработка тестовых данных
test_card_filtered = test_card[['id'] + [col for col in selected_card_features if col in test_card.columns]]
test_card_filtered = test_card_filtered.drop_duplicates(subset=['id'])
test = test.merge(test_card_filtered, on='id', how='left')


# 3. Создание признаков
def create_features(df):
    """Оптимизированная функция создания признаков"""
    # Основные тренды
    for short, long in [(1, 3), (3, 6)]:
        if f'sum_tr_all_{long}' in df.columns and f'sum_tr_all_{short}' in df.columns:
            df[f'trend_sum_{short}_{long}'] = (df[f'sum_tr_all_{long}'] / long) / (
                        df[f'sum_tr_all_{short}'] / short + 1e-6)

    # Финансовые метрики
    if all(col in df.columns for col in ['max_tr_top_up_1', 'max_tr_pay_1', 'sum_tr_all_1']):
        df['savings_capacity'] = (df['max_tr_top_up_1'] - df['max_tr_pay_1']) / (df['sum_tr_all_1'] + 1e-6)

    # Ключевые соотношения расходов
    essential_cols = ['sum_tr_food_retail_1', 'sum_tr_publ_util_1m']
    if all(col in df.columns for col in essential_cols + ['sum_tr_all_1']):
        df['essential_ratio'] = (df[essential_cols].sum(axis=1)) / (df['sum_tr_all_1'] + 1e-6)

    # Важные флаги
    if 'sum_tr_publ_util_1m' in df.columns and 'sum_tr_publ_util_3m' in df.columns:
        df['util_drop_flag'] = (df['sum_tr_publ_util_1m'] < 0.7 * df['sum_tr_publ_util_3m'] / 3).astype(int)

    if 'sum_tr_internet_1' in df.columns:
        df['ecommerce_activity'] = np.log1p(df['sum_tr_internet_1'])

    return df


# Применяем создание признаков к данным
print("Создание производных признаков...")
data = create_features(data)
test = create_features(test)


# 4. Кластеризация
def add_clusters(df, kmeans_model=None):
    cluster_features = [
        'sum_tr_all_1',
        'essential_ratio',
        'savings_capacity',
        'ecommerce_activity'
    ]

    available_features = [f for f in cluster_features if f in df.columns]
    if len(available_features) < 3:
        df['spending_cluster'] = -1
        return df, None

    X_cluster = df[available_features].fillna(0)

    if kmeans_model is None:
        kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE)
        df['spending_cluster'] = kmeans.fit_predict(X_cluster)
        return df, kmeans
    else:
        df['spending_cluster'] = kmeans_model.predict(X_cluster)
        return df, kmeans_model


print("Добавление кластеров...")
data, kmeans_model = add_clusters(data)
test, _ = add_clusters(test, kmeans_model)

# Заполнение пропусков в новых признаках
new_features = [col for col in data.columns if col not in train.columns and col not in ['id', 'target']]
data[new_features] = data[new_features].fillna(0)

test_new_features = [col for col in new_features if col in test.columns]
test[test_new_features] = test[test_new_features].fillna(0)

# ОБУЧЕНИЕ МОДЕЛИ

# Выделение всех признаков (кроме id и target)
features = [col for col in data.columns if col not in ['id', 'target']]
numeric_features = data[features].select_dtypes(include=['int64', 'float64']).columns
categorical_features = data[features].select_dtypes(include=['object', 'category']).columns.tolist()

# Подготовка данных
X = data[features].copy()
X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
X[categorical_features] = X[categorical_features].fillna('MISSING')
y = data['target']


def get_top_features(X, y, cat_features, top_n=100):
    print(f"\nОтбор {top_n} самых важных признаков с помощью Catboost...")
    temp_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        random_seed=RANDOM_STATE,
        cat_features=cat_features,
        verbose=0
    )
    temp_model.fit(X, np.log1p(y))
    feature_importance = temp_model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    top_features = importance_df.head(top_n)['feature'].tolist()
    print(f"Топ-10 самых важных признаков:\n{importance_df.head(10)}")
    return top_features


import json

# Собираем статистики по всем обучающим данным для бота
train_stats = {
    'numerical': X.select_dtypes(include=['number']).median().to_dict(),
    'categorical': {col: 'MISSING' for col in X.select_dtypes(exclude=['number']).columns}
}

with open('train_stats.json', 'w') as f:
    json.dump(train_stats, f, indent=4)
print("Статистики для заполнения сохранены в train_stats.json")

# Применяем отбор признаков
top_features = get_top_features(X, y, categorical_features, TOP_FEATURES)

# Обновляем данные с отобранными признаками
X = X[top_features]
numeric_features = [f for f in top_features if f in numeric_features]
categorical_features = [f for f in top_features if f in categorical_features]
features = top_features

# 4. Создание KFold
cv = KFold(n_splits=N_SPLITS, shuffle=False)

rmsle_scores = []
models = []
oof_predictions = np.zeros(len(X))
feature_importances = pd.DataFrame()
val_predictions = []
val_targets = []

print("Обучение на всех фолдах кроме второго, валидация на втором фолде...")
for fold, (train_index, val_index) in enumerate(cv.split(X)):
    # Для обучения используем все фолды кроме VALIDATION_FOLD
    if fold == VALIDATION_FOLD:
        # Сохраняем валидационные данные
        X_val = X.iloc[val_index]
        y_val = y.iloc[val_index]
        all_val_indices = val_index
        continue

    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        random_seed=RANDOM_STATE,
        cat_features=categorical_features,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS + PATIENCE * fold,
        verbose=0
    )

    model.fit(
        X_train, np.log1p(y_train),
        use_best_model=True
    )

    models.append(model)

    # Делаем предсказания на валидационном фолде
    val_pred = np.expm1(model.predict(X_val))
    val_predictions.append(val_pred)

    # Также делаем OOF предсказания для обучающих фолдов
    oof_pred = np.expm1(model.predict(X.iloc[val_index]))
    oof_predictions[val_index] = oof_pred

    fold_importance = pd.DataFrame({
        'feature': features,
        'importance': model.get_feature_importance(),
        'fold': fold + 1
    })
    feature_importances = pd.concat([feature_importances, fold_importance], axis=0)
    print(f"Fold {fold + 1} | Best iteration: {model.best_iteration_}")

# Усредняем предсказания на валидационном фолде
val_pred = np.mean(val_predictions, axis=0)
val_rmsle = np.sqrt(mean_squared_log_error(y_val, val_pred))
print(f"\nValidation RMSLE на фолде {VALIDATION_FOLD}: {val_rmsle:.5f}")

# Вычисляем OOF score на всех фолдах кроме валидационного
non_val_indices = [i for i in range(len(X)) if i not in all_val_indices]
oof_rmsle = np.sqrt(mean_squared_log_error(y.iloc[non_val_indices], oof_predictions[non_val_indices]))
print(f"OOF RMSLE на обучающих фолдах: {oof_rmsle:.5f}")

# 6. Финальное обучение на всех данных кроме валидационного фолда
print("\nФинальное обучение на всех данных кроме валидационного фолда...")
X_train_full = X.drop(all_val_indices)
y_train_full = y.drop(all_val_indices)

best_iterations = int(np.mean([model.best_iteration_ for model in models])) + 50

final_model = CatBoostRegressor(
    iterations=best_iterations,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    random_seed=RANDOM_STATE,
    cat_features=categorical_features,
    verbose=100
)

final_model.fit(
    X_train_full, np.log1p(y_train_full),
    use_best_model=True
)

# 7. Предсказание на тестовых данных
X_test = test[features].copy()
X_test[numeric_features] = X_test[numeric_features].fillna(X_test[numeric_features].median())
X_test[categorical_features] = X_test[categorical_features].fillna('MISSING')
X_test[categorical_features] = X_test[categorical_features].astype('category')

test_preds = []
for model in models:
    pred = np.expm1(model.predict(X_test))
    test_preds.append(pred)

test_preds.append(np.expm1(final_model.predict(X_test)))
test_pred = np.mean(test_preds, axis=0)

# 8. Сохранение результатов
pd.DataFrame({'id': test['id'], 'target': test_pred}).to_csv('catboost_validation_fold_submission.csv', index=False)
print("\nПредсказания сохранены в catboost_validation_fold_submission.csv")

feature_importances.to_csv('validation_fold_feature_importances.csv', index=False)
pd.DataFrame({'id': data['id'], 'target': data['target'], 'oof_pred': oof_predictions}) \
    .to_csv('catboost_validation_fold_oof_predictions.csv', index=False)