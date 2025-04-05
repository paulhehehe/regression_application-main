# Import accuracy score
from sklearn.metrics import mean_absolute_error

# # Function to predict and evaluate
def evaluate_model(model, x_test, y_test):
    # Predict the loan eligibility on the testing set
    ytest_pred = model.predict(x_test)

    test_mae = mean_absolute_error(ytest_pred, y_test)


    return test_mae