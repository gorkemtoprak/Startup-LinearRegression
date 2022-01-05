# AUTHOR: Gorkem Toprak (B1705.010008)
# DATE: December 24, 2021 Friday

# You can download the dataset from this link: https://www.kaggle.com/karthickveerakumar/startup-logistic-regression

# Import libraries to be used here
import numpy as np   # We need NumPy to perform calculations
import pandas as pd  # It is the python library we will use for data analysis
import matplotlib.pyplot as plt  # The python library we will use to visualize our data
import seaborn as sea
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    # Just read the specific column from CSV file
    col_list = ["Marketing Spend", "Profit"]

    # Read dataset with specific column using 'usecols' parameter.
    # I created my dataset like this because I want to make it single.
    dataset = pd.read_csv('50_Startups.csv', usecols=col_list)
    dataset.head()  # head() is used to get the first n rows.

    # This is another one of the use col methods. We can use either this or usecols method.
    # dataset = pd.read_csv('50_Startups.csv')
    # dataset = dataset[["Marketing Spend", "Profit"]]

    print("Length of the dataset is : " + str(len(dataset)))
    print("Dataset columns: " + str(dataset.columns))
    print("Types of the columns: " + str(dataset.dtypes))
    print('--------------------------------------')
    print(dataset.info())
    print('--------------------------------------')
    print(dataset.describe())  # It prints with dataset values with 2 digits
    print('--------------------------------------')
    # It prints with dataset empty values if the dataset has.
    print(dataset.isnull().sum(), "\nThere is no empty value in my selected dataset")
    print('-----------------------------------------------')
    # Since there is no String value in the two selected columns, I have commented this part.
    # dataset = pd.get_dummies(dataset, columns=['State'])
    # print(dataset.head())

    # With the iloc method, we can get a specific value of the row and column
    # using the index values assigned to it.
    X = dataset.iloc[:, :-1].values  # independent variable array
    y = dataset.iloc[:, 1].values  # Second column of data frame. And dependent variable vector.
    # In my set, thanks to the usecols method, since I select only 2 columns,
    # our x becomes the first y and the second column. And our X is independent variable, our y is dependent variable

    # train_test_split is a function in Sklearn model selection to split data arrays into two subsets;
    # for training data and test data. With this function, we don't need to split the dataset manually.
    from sklearn.model_selection import train_test_split

    # The first parameter (X, y) is the dataset we choose to use.
    # test_size means that the test data is 0.3 of the data set
    # that is, it determines at what rate we will divide the dataset.
    # random_state means, Python splits this data in different parts each time.
    # If we set a random_state value, it divides by that value each time. So we test with the same test data.
    # For example, my random_state is 0, so every time I run it, my results will change.
    X_train_value, X_test_value, y_train_value, y_test_value = train_test_split(X, y, test_size=0.3, random_state=0)

    # train model learns the correlation and learns how to predict the dependent
    # variables based on the independent variable
    from sklearn.linear_model import LinearRegression
    # Creating an object of the Linear Regression class
    regs = LinearRegression()
    # Fitting linear regression to the our training set
    regs.fit(X_train_value, y_train_value)

    # Given a trained model, it predicts the label of a new dataset.
    y_predict = regs.predict(X_test_value)
    # Predicting the test set results
    y_predict
    # We have our predictions in y_pred. Now lets visualize the data set and the regression line

    # actual values
    y_test_value

    # This is how I printed some test results of the selected dataset.
    print('\nTRAIN SCORE: ' + str(regs.score(X_train_value, y_train_value)))
    print('---------------------------------')
    print('TEST SCORE: ' + str(regs.score(X_test_value, y_test_value)))

    # FOR MODEL EVALUATION
    print("\nMEAN ABSOLUTE ERROR (MAE): " + str(mean_absolute_error(y_test_value, y_predict)))
    print('---------------------------------')
    print("ROOT MEAN SQUARED ERROR (MSE): " + str(mean_squared_error(y_test_value, y_predict)))

    # In order to quantitatively comment on the evaluation of the model,
    # I will be taking the mean of the year and then divide the mean_absolute_error
    # (in this case 20675.88525000875) with the mean of the year.
    eval = 20675.88525000875 / np.mean(dataset['Marketing Spend'])

    # FOR EVALUATION
    print("\nEVALUATION: " + str(eval))

    # The effect of the coefficient total profit on marketing spend
    print("\nCOEFFICIENTS: " + str(regs.coef_))
    print('---------------------------------')
    print("CONSTANTS: " + str(regs.intercept_))

    # Here we can see the actual value and the predicted value returned by our dataset.
    df = pd.DataFrame({'Actual Values': y_test_value.flatten(), 'Predicted Values': y_predict.flatten()})
    print("\n" + str(df))

    # heatmap is defined as a graphical representation of data using colors to visualize the value of the matrix
    # I did these to show my dataset with different graphs.
    sea.heatmap(dataset.corr(), annot=True)
    plt.show()
    # pairplot, plots bidirectional relationships for numeric columns across the entire data frame.
    sea.pairplot(dataset)
    plt.show()

    # Plotting the graph for the TRAIN SET
    plt.scatter(X_train_value, y_train_value, color='red')  # It's plotting the values
    plt.plot(X_train_value, regs.predict(X_train_value), color='blue')  # It's plotting the regression line
    plt.title("Marketing Spend vs Profits (Training set)")  # stating the title of the graph
    plt.xlabel("Marketing Spend")  # adding the name of x-axis
    plt.ylabel("Profits")  # adding the name of y-axis
    plt.show()  # It shows the graph
    # NOTE: The y-coordinate is not y_pred because y_pred is predicted salaries of the test set observations.

    # Plotting the graph for the TEST SET
    plt.scatter(X_test_value, y_test_value, color='red')  # It's plotting the values
    plt.plot(X_train_value, regs.predict(X_train_value), color='blue')  # It's plotting the regression line
    plt.title("Marketing Spend vs Profits (Testing set)")
    plt.xlabel("Marketing Spend")
    plt.ylabel("Profits")
    plt.show()


if __name__ == "__main__":
    main()
