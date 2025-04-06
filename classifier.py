import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix

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
    'OptitrackRightHandPosX',
    'OptitrackRightHandPosY',
    'OptitrackRightHandPosZ',
    'OptitrackRightHandRotX',
    'OptitrackRightHandRotY',
    'OptitrackRightHandRotZ',
    'OptitrackLeftHandPosX',
    'OptitrackLeftHandPosY',
    'OptitrackLeftHandPosZ',
    'OptitrackLeftHandRotX',
    'OptitrackLeftHandRotY',
    'OptitrackLeftHandRotZ',
    # 'Temperature',
    # 'GSR',
    # 'BVP',
    # 'Hr',
    # 'Ibi',
    # 'Code',
    'HoloRightHandDistance',
    'HoloLeftHandDistance',
    'HeadDistance',
    'HoloRightHandSpeed',
    'HoloLeftHandSpeed',
    'HeadSpeed',
    'DistanceBetweenHands'
]

def plot_confusion_matrix(cm, labels):
    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    
    # Adding labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    # Show the plot
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def load_and_concatenate_data(file_list):
    data_frames = []
    for file in file_list:
        data = pd.read_csv(file, low_memory=False)
        data_frames.append(data)
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data

def train_and_evaluate_model(data, generate_graphics=False):

    # Omit uncoded rows
    data = data.dropna(subset=['Code'])

    # Separate features and target
    features = data[feature_columns]
    target = data['Code']
    
    # Normalize the features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Initialize and train the decision tree classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_leaf_nodes=100,            # Limit tree complexity
        class_weight="balanced",       # Handle class imbalance
        random_state=42,
        n_jobs=-1                      # Use all CPU cores
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # For each fold, fit the model and evaluate
    for fold, (train_idx, test_idx) in enumerate(cv.split(normalized_features, target)):
        X_train, X_test = normalized_features[train_idx], normalized_features[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, clf.classes_)
        print(f"Fold {fold + 1} Classification Report:\n", classification_report(y_test, y_pred))

    # Fit the model on the entire dataset
    clf.fit(normalized_features, target)
    
    # Make predictions on the test set
    # y_pred = clf.predict(X_test)
    
    # Evaluate the model
    # print("Classification Report:\n", classification_report(y_test, y_pred))

    if generate_graphics:
        feature_importances_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': clf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(8, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importances_df)
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importances.png')
        plt.close()

        # Create a pie chart of class distribution
        class_counts = target.value_counts()
        plt.figure(figsize=(5, 5))
        plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Class Distribution')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('class_distribution.png')
        plt.close()

        # Export decision tree plot
        # plt.figure(figsize=(100,50))
        # plot_tree(clf, feature_names=feature_columns, class_names=clf.classes_, filled=True, rounded=True)
        # plt.savefig('decision_tree.png')

    return clf, scaler

def test_model_on_new_data(model, scaler, new_data_csv):
    new_data = pd.read_csv(new_data_csv, low_memory=False)
    new_data = new_data.dropna(subset=['Code'])
    features = new_data[feature_columns]
    features = scaler.transform(features)
    target = new_data['Code']
    predictions = model.predict(features)
    print("Classification Report:\n", classification_report(target, predictions))

# Entry point
data_files = [
    'P35_Log_Features.csv',
    'P48_Log_Features.csv'
]

# Load and concatenate data
data = load_and_concatenate_data(data_files)

# Train and evaluate the model
clf, scaler = train_and_evaluate_model(data, generate_graphics=True)
