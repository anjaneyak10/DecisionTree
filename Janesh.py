import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd



# Read the CSV file and create a DataFrame
iris_tidy = pd.read_csv("/Users/aryanchandwadkar/Downloads/iris_tidy.csv.xls")

# Plot the DataFrame
sns.set(style="whitegrid")
g = sns.FacetGrid(iris_tidy, col="Species", hue="Part", palette="Set1", col_wrap=2)
g.map(plt.scatter, "Measure", "Value", alpha=0.7)
g.add_legend()
plt.show()




# Load the Iris dataset from scikit-learn
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # Features
y = iris.target  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Train the Decision Tree model
clf = DecisionTreeClassifier(random_state=123, criterion='entropy')
clf.fit(X_train, y_train)

# Predict the test set
y_pred = clf.predict(X_test)

# Get feature names as a list
feature_names = X.columns.tolist()

# Convert class names to a list
class_names = iris.target_names.tolist()

# Plot the decision tree
plt.figure(figsize=(15, 10))
plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
plt.title("Decision Tree for Iris Dataset")
plt.show()

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
