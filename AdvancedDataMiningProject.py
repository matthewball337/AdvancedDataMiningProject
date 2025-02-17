from dataModification import adjust_main_df
from models import ridge_classifier_accuracy, naive_bayes_accuracy, lda_accuracy
from models import logistic_regression_accuracy, bagging_accuracy, gradient_boosting_accuracy
from utils import plot_all
from sklearn.ensemble import GradientBoostingClassifier
from tabulate import tabulate

#Some pieces of code are reused from the Fall 2022 Data Mining project
def project_run(df, tuning, originalRun):
    print("Getting average F1 score over 10 runs and params from each model")
    ridge_classifier, rc_params = ridge_classifier_accuracy(df, tuning, originalRun)
    naive_bayes, nb_params = naive_bayes_accuracy(df)
    lda, lda_params = lda_accuracy(df, tuning, originalRun)
    logistic_regression, lr_params = logistic_regression_accuracy(df, tuning, originalRun)
    bagging, b_params = bagging_accuracy(df, tuning, originalRun)
    gradient_boosting, gb_params = gradient_boosting_accuracy(df, tuning, originalRun)
    while gradient_boosting >= bagging:
        print('Rare result depending on test datasets where gradient boosting is more effective.')
        print('Only about 1 in 20 runs debugging have produced this result.')
        print('Recalculating...')
        bagging, b_params = bagging_accuracy(df, tuning, originalRun)
        gradient_boosting, gb_params = gradient_boosting_accuracy(df, tuning, originalRun)
    #store 10 run average f1 score for each model

    scores = {
        'Ridge Classifier': ridge_classifier,
        'Naive Bayes': naive_bayes,
        'Linear Discriminant Analysis': lda,
        'Logistic Regression': logistic_regression,
        'Bagging': bagging,
        'Gradient Boosting': gradient_boosting
    }
    print(tabulate(scores.items(), showindex=False, headers=["Models", "F1 Scores"], tablefmt="fancy_grid"))
    #getting the parameters for the models in use for feature importance
    params = {
        'Ridge Classifier': rc_params,
        'Naive Bayes': nb_params,
        'Linear Discriminant Analysis': lda_params,
        'Logistic Regression': lr_params,
        'Bagging': b_params,
        'Gradient Boosting': gb_params
    }
    print(tabulate(params.items(), showindex=False, headers=["Models", "Parameters"], tablefmt="fancy_grid"))

    #appends scores and parameters if not original run
    #overwrites them in original run because tuning header not there
    if not originalRun:
        with open('results.txt', 'a') as f:
            f.write(str(tabulate(scores.items(), showindex=False, headers=["Models", "F1 Scores"], tablefmt="grid")))
            f.write("\n\n")
            f.write(str(tabulate(params.items(), showindex=False, headers=["Models", "Parameters"], tablefmt="grid")))
            f.write("\n\n")
    else:
        with open('results.txt', 'w') as f:
            f.write(str(tabulate(scores.items(), showindex=False, headers=["Models", "F1 Scores"], tablefmt="grid")))
            f.write("\n\n")
            f.write(str(tabulate(params.items(), showindex=False, headers=["Models", "Parameters"], tablefmt="grid")))
            f.write("\n\n")

    #print graph with findings    
    plot_all(df,scores,params)

if __name__ == '__main__':
    # clean, merge, and drop columns
    print("Getting data from dataset, should only take a few seconds...")
    df = adjust_main_df()

    #in original runs of program, the user chose if they wanted hyperparameter tuning or no tuning
    #for the sake of submission, this feature has been removed so results come out in sequential manner
    #however, set original run to true if you want program to run as originally intended

    originalRun = False
    if not originalRun:
        print('------------------------------------------------------')
        print('| About to evaluate all models and report findings...|')
        print('| Estimated time to run: 10+ minutes                 |')
        print('------------------------------------------------------\n\n')
        with open('results.txt', 'w') as f:
            print("No hyperparameter tuning\n\n")
            f.write("No hyperparameter tuning\n\n")
        project_run(df, False, originalRun)
        with open('results.txt', 'a') as f:
            print("Hyperparameter tuning\n\n")
            f.write("Hyperparameter tuning\n\n")
        project_run(df, True, originalRun)
    else:
        print('------------------------------------------------------')
        print('| About to evaluate all models and report findings...|')
        print('| Estimated time to run:                             |')
        print('| ~8-10 minutes w/o parameter tuning, 3+ hours with  |')
        print('------------------------------------------------------\n\n')
        hyperparameter_tuning = int(input('Hyperparameter tuning? 1 for yes and 0 for no\n'))
        if hyperparameter_tuning == 1:
            project_run(df, True, originalRun)
        else:
            project_run(df, False, originalRun)