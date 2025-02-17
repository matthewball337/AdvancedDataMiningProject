from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifier
from sklearn.inspection import permutation_importance
from tabulate import tabulate

'''
References:

1) Feature importance (Ridge Classifier, Logistic Regression): https://machinelearningmastery.com/calculate-feature-importance-with-python/
2) Permutation feature importance (GaussianNB): https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance
3) Some pieces of code are reused from the Fall 2022 Data Mining project
'''

#Makes run/pass equal to 1 and field goal/punt equal to 0
def decision(dataframe):
    list = []
    
    for row in dataframe['play_type']:
        if row == 'run' or row == 'pass':
            list.append(1)
        else:
            list.append(0)
            
    dataframe['decision'] = list
    
    return dataframe
    
#Maps the team cities and team names
def teams():
    return {
    'Dolphins': 'MIA',
    '49ers' : 'SF',
    'Bears' : 'CHI',
    'Bengals' : 'CIN',
    'Bills' : 'BUF',
    'Broncos' : 'DEN',
    'Browns' : 'CLE',
    'Buccaneers' : 'TB',
    'Cardinals' : 'ARI',
    'Chargers' : 'LAC',
    'Chiefs' : 'KC',
    'Colts' : 'IND',
    'Cowboys' : 'DAL',
    'Eagles' : 'PHI',
    'Falcons' : 'AIL',
    'Giants' : 'NYG',
    'Jaguars' : 'JAX',
    'Jets' : 'NYJ',
    'Lions' : 'DET',
    'Packers' : 'GB',
    'Panthers' : 'CAR',
    'Patriots' : 'NE',
    'Raiders' : 'OAK',
    'Rams' : 'LA',
    'Ravens' : 'BAL',
    'Redskins' : 'WAS',
    'Saints' : 'NO',
    'Seahawks' : 'SEA',
    'Steelers' : 'PIT',
    'Texans' : 'HOU',
    'Titans' : 'TEN',
    'Vikings' : 'MIN'
}

#Creates a training and testing set
def create_X_y(df):
    X = df.drop(['decision'], axis=1)
    y = df['decision']
    return X,y
    
    
#Takes in a model, returns the average over 10 runs. Returns the avg f1 score
def get_average_f1(model,df, bn=False):
    X,y = create_X_y(df)
    f1_sum = 0
    for i in range(10):
        print('Creating training and test data')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        print('Fitting X train and y train')
        model.fit(X_train, y_train)
        print('Getting Results')
        results = model.predict(X_test)
        #potential imbalance, calculate f1 score
        f1 = f1_score(y_test, results)
        print('F1 score is ', f1, ' for trial ', i+1)
        f1_sum += f1
        print()
    print('Average F1 score is ', f1_sum/10)
    print('-----------------------------------------------\n')
    return(f1_sum/10)


def avg_yards_decision_graph(df):
    # Bar chart for yds to go split up by decision
    yds_decision = sns.barplot(data=df, x='decision', y='ydstogo')
    yds_decision.set_title("Average Yards to go by Decision")
    plt.show()


def timeout_decision_graph(df):
    # Scatter plot showcasing the decision density around different remaining team timeouts for
    #the team with the ball vs the defending team
    timeout_diff = sns.scatterplot(data=df, x='posteam_timeouts_remaining', y='defteam_timeouts_remaining', hue='decision')
    timeout_diff.set_title("Timeouts and Decision")
    plt.show()
    
    
def win_probability_graph(df):
    # Bar chart reflecting decision effect on win probability
    yds_wp = sns.barplot(data=df, x='decision', y='wp', hue='decision')
    yds_wp.set_title("Average Win Probability by Decision")
    plt.show()
    
#Graph was added into project under the assumption that it was an important feature
#due to test runs
#unfortunately, that was not the case
def scoring_decision_graph(df):
    # Scatter plot showcasing the decision density around different scoring scenarios for
    #the team with the ball vs the defending team
    score_diff = sns.scatterplot(data=df, x='posteam_score', y='defteam_score', hue='decision')
    score_diff.set_title("Score Differential and Decision")
    plt.show()


#Graphs the average F1 score
def avg_scores_graph(scores):
   df = pd.DataFrame(scores.items())
   model_perf = sns.barplot(data=df, x=0, y=1)
   model_perf.set(xlabel = 'Models', ylabel = 'F score')
   plt.show()
   
def feature_importance_graph(df,scores,params):
    print("Preparing model...")
    X,y = create_X_y(df)
    #Choose a model
    best_model = max(scores, key=scores.get)
    best_params = params[best_model]
    print("Preparing best model with user params")
    if best_model == 'Ridge Classifier':
        if len(best_params) == 0:
            model = RidgeClassifier()
        else:
            model = RidgeClassifier(**best_params)
    elif best_model == 'Naive Bayes':
        model = GaussianNB()
    elif best_model == 'Linear Discriminant Analysis':
        if len(best_params) == 0:
            model = LinearDiscriminantAnalysis()
        else:
            model = LinearDiscriminantAnalysis(**best_params)
    elif best_model == 'Logistic Regression':
        model = LogisticRegressionCV(**best_params)
    elif best_model == 'Bagging':
        model = BaggingClassifier(**best_params)
    elif best_model == 'Gradient Boosting':
        if len(best_params) == 0:
            model = GradientBoostingClassifier()
        else:
            model = GradientBoostingClassifier(**best_params)
            
    #Print out the most important features correlating
    print("Training with data gotten from train test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit(X_train,y_train)
    print("Getting feature importances")
    if best_model == 'Ridge Classifier':
        feature_importance = model.coef_
    elif best_model == 'Naive Bayes':
        feature_importance = permutation_importance(model, X_train, y_train, n_jobs=-1, random_state=0)
    elif best_model == 'Linear Discriminant Analysis':
        feature_importance = model.coef_
    elif best_model == 'Logistic Regression':
        feature_importance = model.coef_[0]
    elif best_model == 'Bagging':
        feature_importance = np.mean([gb.feature_importances_ for gb in model.estimators_], axis=0)
    elif best_model == 'Gradient Boosting':
        feature_importance = model.feature_importances_
    feature_importances = pd.DataFrame({'Features': X_train.columns, 'feature_importance': feature_importance})
    feature_importances = feature_importances.sort_values('feature_importance', ascending=False).head(10)
    print(tabulate(feature_importances, showindex=False, headers=["Features", "Feature Importance"], tablefmt="fancy_grid"))
    with open('results.txt', 'a') as f:
        f.write(str(tabulate(feature_importances, showindex=False, headers=["Features", "Feature Importance"], tablefmt="grid")))
        f.write("\n\n")
    barh = plt.barh(feature_importances['Features'], feature_importances['feature_importance'])
    plt.title('Feature Importances for ' + str(best_model))
    plt.show()
    
    #Seaborn barplot for feature importances
    sns_plot = sns.barplot(data=feature_importances, x='feature_importance', y='Features')
    plt.show()
    
    
def plot_all(df,scores,params):
    avg_yards_decision_graph(df)
    timeout_decision_graph(df)
    win_probability_graph(df)
    scoring_decision_graph(df)
    avg_scores_graph(scores)
    print('Next 2 graphs are feature importance and take a minute to show...')
    feature_importance_graph(df,scores,params)