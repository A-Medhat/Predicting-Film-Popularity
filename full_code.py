import copy
import json
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

##############################################################################
pd.set_option('display.max_columns', None)
def tagline_feature_processing():
    new_overview = []
    overview = data['overview']
    regexp = re.compile(r'[a-zA-Z]')
    regexp2 = re.compile(r'[0-9]')
    for sentence in overview:
        if regexp.findall(str(sentence)):
            words = word_tokenize(str(sentence))
            i = 0
            for w in words:
                if regexp2.findall(w):
                    w = re.sub("[0-9]", "", w)
                    words[i] = w
                i+=1
            filtered_words = [w for w in words if w.casefold() not in stop_words]
            no_punct = [''.join(char for char in word if char not in string.punctuation) for word in filtered_words]
            no_punct = [word for word in no_punct if word]
            new_overview.append(' '.join(no_punct))
    new_overview = pd.Series(new_overview)
    x = v.fit_transform(new_overview)
    q = pd.DataFrame(x.toarray(),columns=v.get_feature_names())
    q.to_csv("overview.csv")
def tagline_feature_processing():
    new_tagline = []
    tagline = data['tagline']
    tagline[0] = 'a'
    tagline = tagline.fillna("a")
    for sentence in tagline:
        words = word_tokenize(str(sentence))
        filtered_words = [w for w in words if w.casefold() not in stop_words]
        no_punct = [''.join(char for char in word if char not in string.punctuation) for word in filtered_words]
        no_punct = [word for word in no_punct if word]
        new_tagline.append(' '.join(no_punct))
    new_tagline = pd.Series(new_tagline)
    x = v.fit_transform(new_tagline)
    q = pd.DataFrame(x.toarray(),columns=v.get_feature_names())
    q2 = q.drop(q.iloc[:, 0:68],axis = 1)
    q2.to_csv("tagline.csv")

def replace_nulls_and_zeros_with_avg(df,column_name, avg):

    df_copy = df.copy()
    df_copy[column_name] = df_copy[column_name].replace([0, np.nan], np.nan).fillna(avg)

    return df_copy

################################################################################

def replace_nulls_and_zeros_with_avg_in_y(df, column_name):
    df_copy = df.copy()

    avg = df_copy[column_name].replace([0, np.nan], np.nan).mean()

    df_copy[column_name] = df_copy[column_name].replace([0, np.nan], np.nan).fillna(avg)

    return df_copy

###############################################################################


def preprocess_data(train_data, test_data):
    X_train = train_data.copy()
    X_test = test_data.copy()

    columns_to_replace = ['budget', 'revenue', 'runtime', 'vote_count', 'viewercount']


    for column in columns_to_replace:
        avg = X_train[column].replace([0, np.nan], np.nan).mean()
        X_train = replace_nulls_and_zeros_with_avg(X_train, column, avg)
        X_test = replace_nulls_and_zeros_with_avg(X_test, column, avg)


    X_train = split_date(X_train, 'release_date')
    X_test = split_date(X_test, 'release_date')

    X_train = one_hot_encode(X_train, 'genres')
    X_test = one_hot_encode(X_test, 'genres')


    col_to_move = 'original_language'
    X_train.insert(len(X_train.columns) - 1, col_to_move, X_train.pop(col_to_move))
    X_test.insert(len(X_test.columns) - 1, col_to_move, X_test.pop(col_to_move))

    X_train = replace_null_with_string(X_train, 'original_language', 'en')
    X_test = replace_null_with_string(X_test, 'original_language', 'en')
    X_train = encode_column(X_train, 'original_language')
    X_test = encode_column(X_test, 'original_language')

    columns_to_filter = ['spoken_languages', 'keywords', 'production_countries', 'production_companies']
    for column in columns_to_filter:
        X_train = filter_frequent_items(X_train, column, 10)
        X_train[column] = X_train[column].fillna('[]')
        X_train = one_hot_encode(X_train, column)
       # X_test = filter_frequent_items(X_test, column, 10)
        X_test[column] = X_test[column].fillna('[]')
        X_test = one_hot_encode(X_test, column)
        df1_columns = X_train.columns.tolist()
        df2_columns = X_test.columns.tolist()
        common_columns = list(set(df1_columns) & set(df2_columns))
        X_test=X_test.drop([col for col in df2_columns if col not in common_columns], axis=1)
    #print(X_test)

    # Drop homepage and id columns (unique values)
    X_train = dropcol(X_train, 'homepage')
    X_train = dropcol(X_train, 'id')
    X_test = dropcol(X_test, 'homepage')
    X_test = dropcol(X_test, 'id')

    # Drop status, tagline, title, and overview columns
    X_train = dropcol(X_train, 'status')
    X_train = dropcol(X_train, 'tagline')
    X_train = dropcol(X_train, 'title')
    X_train = dropcol(X_train, 'overview')
    X_train = dropcol(X_train, 'original_title')
    X_test = dropcol(X_test, 'status')
    X_test = dropcol(X_test, 'tagline')
    X_test = dropcol(X_test, 'title')
    X_test = dropcol(X_test, 'overview')
    X_test = dropcol(X_test, 'original_title')

    return X_train, X_test



def correlation_analysis(df):


    # Compute the correlation matrix
    corr_matrix = df.corr()
    print(corr_matrix)
    # Return the correlation matrix
    return corr_matrix


###############################################################################

def one_hot_encode(df, column_name):
    df.loc[:, column_name] = df[column_name].apply(json.loads)
    genre_lists = df[column_name].apply(lambda m: [d['name'] for d in m])

    mlb = MultiLabelBinarizer()
    mlb.fit(genre_lists)

    genre_matrix = mlb.transform(genre_lists)
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_,)
    genre_df['index_col']=df.index
    genre_df.set_index('index_col',inplace=True)
    #print(genre_df)
    df = pd.concat([df, genre_df], axis=1)
    # Drop the original column
    #df = df.dropna(subset=[column_name])
    df = df.drop(columns=[column_name])

    #print(df)
    return df

#############################################################################

def dropcol(df, column_name):
    df = df.drop(columns=[column_name]).copy()
    return df

#############################################################################

def encode_column(df, column):

    le = LabelEncoder()
    df[f'{column}_encoded'] = le.fit_transform(df[column])
    df.drop(column, axis=1, inplace=True)
    return df

#############################################################################

def filter_frequent_items(df, column_name, n):

    filtered_df = df.copy()

    column_data = filtered_df[column_name]

    item_counts = column_data.value_counts()

    frequent_items = item_counts[item_counts >= n].index.tolist()

    filtered_df[column_name] = filtered_df[column_name].where(filtered_df[column_name].isin(frequent_items), other=None)

    return filtered_df

#############################################################################

def replace_null_with_string(df, column_name, string):
    new_df = df.copy()
    new_df[column_name].fillna(string, inplace=True)
    return new_df

#############################################################################

def split_date(df, date_column):

    df[['month', 'day', 'year']] = df[date_column].str.split('/', expand=True)
    df = df.drop(columns=[date_column, 'day'])
    return df

#############################################################################
#############################################################################

def spearman_feature_selection(df, data, target, threshold=0.5):

    import pandas as pd
    from scipy.stats import spearmanr

    original_cols = df.columns.tolist()

    data = pd.DataFrame(data)

    correlations = {}
    for i, column in enumerate(data.columns):
        correlations[original_cols[i]] = spearmanr(data[column], target)[0]

    selected_features = [col for col in correlations.keys() if abs(correlations[col]) > threshold]


    selected_features = df[selected_features]

    #print(selected_features)
    return selected_features

#############################################################################
import numpy as np
from scipy.stats import kendalltau
from sklearn.feature_selection import f_classif


def kendall_anova_feature_selection(X, y, num_features):

    # Compute Kendall's tau correlation between each feature and target variable
    corr = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        corr[i], _ = kendalltau(X[:, i], y)

    # Compute ANOVA F-test statistic between each feature and target variable
    f_values, _ = f_classif(X, y)

    # Compute Kendall-ANOVA score for each feature
    kendall_anova_scores = corr * np.sqrt(f_values)

    # Select top 'num_features' features with highest Kendall-ANOVA score
    selected_features = np.argsort(kendall_anova_scores)[-num_features:]

    return selected_features


import numpy as np
from scipy.stats import kendalltau


def _feature_selection(X, y, num_features):

    n_samples, n_features = X.shape
    feature_scores = np.zeros(n_features)

    for i in range(n_features):
        tau, _ = kendalltau(X[:, i], y)
        feature_scores[i] = abs(tau)

    sorted_indices = np.argsort(feature_scores)[::-1]
    selected_features = sorted_indices[:num_features]

    return selected_features


def anova_feature_selection(X, y, num_features):
    y = np.ravel(y)
    # Calculate ANOVA F-values and p-values for each feature
    f_values, p_values = f_classif(X, y)

    # Rank features by F-value and select top features
    ranked_features = np.argsort(f_values)[::-1]
    selected_indices = ranked_features[:num_features]
    selected_features = X.iloc[:, selected_indices]

    #print(selected_features)
    return selected_features

#############################################################################

def normalize_feature(df):
    copy_df=copy.deepcopy(df)
    scaler=MinMaxScaler()
    scaler.fit(copy_df)
    return pd.DataFrame(scaler.transform(copy_df),columns=df.columns)

#############################################################################

def linear_regression(X_train, X_test, y_train, y_test):

  regr = LinearRegression()
  regr.fit(X_train, y_train)


  y_pred = regr.predict(X_test)

  coef = regr.coef_
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  return regr, coef, mse, r2

#############################################################################

def polynomial_regression(X_train, X_test, y_train, y_test, degree=2):

    poly_features = PolynomialFeatures(degree=degree)

    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    regr = LinearRegression()

    regr.fit(X_train_poly, y_train)
    y_pred = regr.predict(X_test_poly)
    coef = regr.coef_
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return regr, coef, mse, r2

#############################################################################

def ridge_regression(X_train, X_test, y_train, y_test, alpha=1.0):
    regr = Ridge(alpha=alpha)
    y_train = np.ravel(y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return regr, mse, r2

#############################################################################

def random_forest_regression(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None):
    regr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    y_train = np.ravel(y_train)
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return regr, mse, r2
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

#############################################################################

def decision_tree_regression(X_train, X_test, y_train, y_test, max_depth=None):
    regr = DecisionTreeRegressor(max_depth=max_depth)

    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return regr, mse, r2

#############################################################################

def select_k_best(X, y, k):
    y = np.ravel(y)
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()]
    print(X[selected_columns])

    return X[selected_columns]


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def plot_regression(x, y):
    plt.figure()

    for i in range(x.shape[1]):
        xi = x.iloc[:, i]

        a, b = np.polyfit(xi, y, 1)
        plt.scatter(xi, y)

        plt.plot(xi, a * xi + b, color='red')

        plt.xlabel('{}'.format(x.columns[i]))  # Use the column name as the x-axis label
        plt.ylabel('vote_average')

        plt.show()

data = pd.read_csv('movies-regression-dataset.csv')
x=data.iloc[:,:19]

y=data.iloc[:,19:20]

#splitting data

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
t_y = replace_nulls_and_zeros_with_avg_in_y(y_train, 'vote_average')
training_avg = y_train['vote_average'].replace([0, np.nan], np.nan).mean()
correlation_analysis(data)
new_traind_x,new_tested_x=preprocess_data(X_train,X_test)
##################
avg = t_y['vote_average'].replace([0, np.nan], np.nan).mean()
y_test = replace_nulls_and_zeros_with_avg(y_test, 'vote_average', avg)
###################

x_corr=new_traind_x.iloc[:,:7].values
x_corrdf=new_traind_x.iloc[:,:7]
x_an=new_traind_x.iloc[:,7:]
#spearman feature selection for { " numerical input variable" and " categorical output variable " }
selected_feature_corr=spearman_feature_selection(x_corrdf, x_corr, t_y, threshold=0.1)
# anova feature selection for { " categorical input variable" and " numerical output variable " }
selected_feature_an=anova_feature_selection(x_an, t_y, 13)

#selectbest=select_k_best(x_an, t_y, 8)
#concatenation between the selected columns
final_df = pd.concat([selected_feature_corr, selected_feature_an], axis=1)
final_df = dropcol(final_df, 'Português')
final_df = dropcol(final_df, 'Documentary')
#final_df = dropcol(final_df, 'Western')

#final_df = dropcol(final_df, 'United Kingdom')
#final_df = dropcol(final_df, 'War')
#final_df = dropcol(final_df, 'Deutsch')
df1_columns = final_df.columns.tolist()
df2_columns = new_tested_x.columns.tolist()
common_columns = list(set(df1_columns) & set(df2_columns))
new_tested_x=new_tested_x.drop([col for col in df2_columns if col not in common_columns], axis=1)
new_tested_x = new_tested_x.reindex(columns=final_df.columns)
#######################################
plot_regression(final_df,t_y)

regr_linear_regression, coef_linear_regression, mse_linear_regression, r2_linear_regression=linear_regression(final_df, new_tested_x, t_y, y_test)

regr_pol, coef_pol, mse_pol, r2_pol=polynomial_regression(final_df, new_tested_x, t_y, y_test, degree=2)

regr_random_forest, mse_random_forest, r2_random_forest=random_forest_regression(final_df, new_tested_x, t_y, y_test, n_estimators=100, max_depth=None)

regr_decision_tree, mse_decision_tree, r2_decision_tree=decision_tree_regression(final_df, new_tested_x, t_y, y_test, max_depth=None)

regr_ridge_regression, mse_ridge_regression, r2_ridge_regression=ridge_regression(final_df, new_tested_x, t_y, y_test, alpha=1.0)

print("linear regression accuracy : "  )
print(r2_linear_regression)
print("linear regression mse : " )
print(mse_linear_regression)
print("random forest regression accuracy : ")
print(r2_random_forest)
print("random forest regression mse" )
print(mse_random_forest)
print("ridge regression accuracy : ")
print(r2_ridge_regression)
print("ridge regression mse : " )
print(mse_ridge_regression)