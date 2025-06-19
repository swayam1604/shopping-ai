import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

MONTHS_ABBR = {
        "Jan": 0, # No January data in shopping.csv
        "Feb": 1,
        "Mar": 2,
        "Apr": 3, # No April data in shopping.csv
        "May": 4,
        "June": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11
    }

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    # Convert each data point into required format
    # Then append to respective lists
    with open(filename, "r") as file:
        file_reader = csv.DictReader(file)
        for data in file_reader:
            evidence.append([
                int(data['Administrative']),
                float(data['Administrative_Duration']),
                int(data['Informational']),
                float(data['Informational_Duration']),
                int(data['ProductRelated']),
                float(data['ProductRelated_Duration']),
                float(data['BounceRates']),
                float(data['ExitRates']),
                float(data['PageValues']),
                float(data['SpecialDay']),
                MONTHS_ABBR[data["Month"]],
                int(data['OperatingSystems']),
                int(data['Browser']),
                int(data['Region']),
                int(data['TrafficType']),
                1 if data['VisitorType'] == "Returning_Visitor" else 0,
                1 if data['Weekend'] == "TRUE" else 0
            ])
            labels.append(1 if data["Revenue"] == "TRUE" else 0)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    return KNeighborsClassifier(n_neighbors = 1).fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    assert len(labels) == len(predictions)

    # Initialise values
    actual_positive = 0
    predicted_positive = 0

    actual_negative = 0
    predicted_negative = 0

    # Record each instance of actual/true positive/negative
    for i in range(len(labels)):
        if labels[i] == 1:
            actual_positive += 1
        else:
            actual_negative += 1
        
        if predictions[i] == labels[i]:
            if labels[i] == 1:
                predicted_positive += 1
            else:
                predicted_negative += 1
    
    # Calculate the associated values
    sensitivity = predicted_positive / actual_positive
    specificity = predicted_negative / actual_negative

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
