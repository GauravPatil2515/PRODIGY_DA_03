import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_graphviz
import graphviz

try:
    # Check versions of the libraries
    print("pandas version:", pd.__version__)
    print("scikit-learn version:", sklearn.__version__)
    print("graphviz version:", graphviz.__version__)

    # Load the dataset
    df = pd.read_csv('Customer-Churn-Records.csv')
    print("CSV file loaded successfully.")

    # Display the first 10 rows of the dataframe
    print("\nFirst 10 rows of the dataframe:")
    print(df.head(10))

    # Drop unnecessary columns
    columns_to_drop = ['RowNumber', 'Surname', 'Geography', 'Gender', 'Card Type']
    data = df.drop(columns_to_drop, axis=1)
    print("\nData after dropping unnecessary columns:")
    print(data.head())

    # Check for duplicated rows and the shape of the dataframe
    print("\nNumber of duplicated rows:", data.duplicated().sum())
    print("Shape of the dataframe:", data.shape)

    # Calculate the mean values for Balance and IsActiveMember
    threshold_balance = data["Balance"].mean()
    threshold_active = data["IsActiveMember"].mean()
    print("\nThreshold balance:", threshold_balance)
    print("Threshold active member:", threshold_active)

    # Function to create purchase label based on thresholds
    def create_purchase_label(row):
        if row['IsActiveMember'] > threshold_active and row['Balance'] > threshold_balance:
            return 1
        else:
            return 0

    # Apply the function to create the PurchaseLabel column
    data['PurchaseLabel'] = data.apply(create_purchase_label, axis=1)
    print("\nBalance, IsActiveMember, and PurchaseLabel columns:")
    print(data[['Balance', 'IsActiveMember', 'PurchaseLabel']])

    # Sum of the PurchaseLabel column
    print("\nSum of PurchaseLabel:", data['PurchaseLabel'].sum())

    # Define the target variable and features
    y = data['PurchaseLabel']
    X = data.drop(['PurchaseLabel'], axis=1)

    print("\nFeatures (X):")
    print(X.head())
    print("\nTarget (y):")
    print(y.head())

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    print("\nData split into training and testing sets.")

    # Initialize and train the DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    print("\nDecision tree classifier trained.")

    # Predict the target variable for the test set
    y_pred = clf.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy)

    # Print the classification report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Visualize the decision tree
    dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=["No", "Yes"], filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree", format='png', cleanup=True)
    print("\nDecision tree visualization created as 'decision_tree.png'.")

except FileNotFoundError:
    print("Error: The file 'Customer-Churn-Records.csv' was not found.")
except pd.errors.EmptyDataError:
    print("Error: The file 'Customer-Churn-Records.csv' is empty.")
except pd.errors.ParserError:
    print("Error: There was an error parsing the file 'Customer-Churn-Records.csv'.")
except Exception as e:
    print("An error occurred:", e)