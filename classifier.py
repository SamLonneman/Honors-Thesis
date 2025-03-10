import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_classifier(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file, low_memory=False)
    
    # Omit uncoded rows
    data = data.dropna(subset=['Code'])

    # Omit rows where hands and head are not moving (HoloLens tracking lost)
    data = data[data['HoloRightHandPosX'] != data['HoloRightHandPosX'].shift()]
    data = data[data['HoloRightHandRotX'] != data['HoloRightHandRotX'].shift()]
    data = data[data['HoloLeftHandPosX'] != data['HoloLeftHandPosX'].shift()]
    data = data[data['HoloLeftHandRotX'] != data['HoloLeftHandRotX'].shift()]
    data = data[data['HeadX'] != data['HeadX'].shift()]
    data = data[data['GazeX'] != data['GazeX'].shift()]
    data = data[data['ForwardX'] != data['ForwardX'].shift()]

    # Take only first sample from each code
    # data = data[data['Code'] != data['Code'].shift()]
    
    # Define the columns to be used as features
    feature_columns = [
        # 'AppTime',
        # 'UnixTime',
        # 'TriggerTrial',
        # 'PostTrigger',
        # 'RegionNum',
        # 'RegionName',
        # 'Task',
        # 'Step',
        # 'Idle',
        'HeadX',
        'HeadY',
        'HeadZ',
        'GazeX',
        'GazeY',
        'GazeZ',
        'ForwardX',
        'ForwardY',
        'ForwardZ',
        # 'GazeObject',
        'HoloRightHandPosX',
        'HoloRightHandPosY',
        'HoloRightHandPosZ',
        'HoloRightHandRotX',
        'HoloRightHandRotY',
        'HoloRightHandRotZ',
        'HoloLeftHandPosX',
        'HoloLeftHandPosY',
        'HoloLeftHandPosZ',
        'HoloLeftHandRotX',
        'HoloLeftHandRotY',
        'HoloLeftHandRotZ',
        # 'OptitrackRightHandPosX',
        # 'OptitrackRightHandPosY',
        # 'OptitrackRightHandPosZ',
        # 'OptitrackRightHandRotX',
        # 'OptitrackRightHandRotY',
        # 'OptitrackRightHandRotZ',
        # 'OptitrackLeftHandPosX',
        # 'OptitrackLeftHandPosY',
        # 'OptitrackLeftHandPosZ',
        # 'OptitrackLeftHandRotX',
        # 'OptitrackLeftHandRotY',
        # 'OptitrackLeftHandRotZ',
        # 'Temperature',
        # 'GSR',
        # 'BVP',
        # 'Hr',
        # 'Ibi',
        # 'Code',
        # 'HoloRightHandDistance',
        # 'HoloLeftHandDistance',
        # 'HeadDistance',
        # 'HoloRightHandSpeed',
        # 'HoloLeftHandSpeed',
        # 'HeadSpeed',
        # 'DistanceBetweenHands'
    ]
    features = data[feature_columns]
    
    target = data['Code']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Initialize a decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Evaluate the classifier
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Export feature importance plot
    feature_importances_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': clf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=feature_importances_df)
    plt.title('Feature Importances')
    plt.savefig('feature_importances.png')

    # Export decision tree plot
    plt.figure(figsize=(100,50))
    plot_tree(clf, feature_names=feature_columns, class_names=clf.classes_, filled=True, rounded=True)
    plt.savefig('decision_tree.png')

# Example usage
csv_file = 'P48_Log_NEW_With_Codes_with_features.csv'
classifier = train_classifier(csv_file)
