from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pandas as pd
import numpy as np


def random_forest(train_X, train_y, test_X, n_estimators=10, min_samples_leaf=5, criterion='mse', print_model_details=False, gridsearch=True):

    if gridsearch:
        tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                'n_estimators':[10, 50, 100],
                                'criterion':[criterion]}]
        rf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
    else:
        # Create the model
        rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, criterion=criterion)

    # Fit the model
    rf.fit(train_X, train_y)

    if gridsearch and print_model_details:
        print(rf.best_params_)

    if gridsearch:
        rf = rf.best_estimator_

    # Apply the model
    pred_training_y = rf.predict(train_X)
    pred_test_y = rf.predict(test_X)

    if print_model_details:
        print('Feature importance random forest:')
        ordered_indices = [i[0] for i in sorted(enumerate(rf.feature_importances_), key=lambda x:x[1], reverse=True)]

        for i in range(0, len(rf.feature_importances_)):
            print(train_X.columns[ordered_indices[i]], end='')
            print(' & ', end='')
            print(rf.feature_importances_[ordered_indices[i]])

    return pred_training_y, pred_test_y


def prepare_dataset_regression(data_path, label_col="Average PACES", exclude_cols = ["ID", "datetime"], split_ratio=0.7, unknow_users=False):
    data = pd.read_csv(data_path)
    user_dfs = []

    for user in data["ID"].unique():
        user_dfs.append(data[data["ID"]==user])

    if unknow_users:
        train, test = user_dfs[:int(split_ratio*len(user_dfs))], user_dfs[int(split_ratio*len(user_dfs)):]
    else:
        train, test = [], []
        for user_df in user_dfs:
            user_train, user_test = user_df[:int(split_ratio*len(user_df))], user_df[int(split_ratio*len(user_df)):]
            train, test = train+[user_train], test+[user_test]

    label_col = [label_col]
    train, test = pd.concat(train, axis=0), pd.concat(test, axis=0)
    vars = [col for col in train.columns.to_list() if col not in exclude_cols+label_col]
    train_X, train_y = train[vars], train[label_col]
    test_X, test_y = test[vars], test[label_col]

    return train_X, train_y, test_X, test_y


if __name__ == "__main__":
    data_path = "final.csv"
    label_col = "Average PACES"
    exclude_cols = ["ID", "datetime"]
    split_ratio = 0.7
    unknow_users = False # if we want to predict enjoyment for new users instead of the rest part of each user
    train_X, train_y, test_X, test_y = prepare_dataset_regression(data_path, label_col, exclude_cols, split_ratio, unknow_users)
    train_y, test_y = np.ravel(train_y), np.ravel(test_y)

    pred_training_y, pred_test_y = random_forest(
                train_X,
                train_y,
                test_X,
                n_estimators=10,
                min_samples_leaf=5,
                criterion='friedman_mse',
                print_model_details=False, 
                gridsearch=False # to print the best params, enable 'print_model_details'
    )

    print(f"MSE: {metrics.mean_squared_error(test_y, pred_test_y)}")
    print(f"MAE: {metrics.mean_absolute_error(test_y, pred_test_y)}")


