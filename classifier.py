import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(normalized_features, target, test_size=0.2, random_state=42)
    
    # Initialize and train the decision tree classifier
    clf = DecisionTreeClassifier(max_leaf_nodes=100, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate the model
    print("Classification Report:\n", classification_report(y_test, y_pred))

    if generate_graphics:
        feature_importances_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': clf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=feature_importances_df)
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importances.png')

        # Export decision tree plot
        plt.figure(figsize=(100,50))
        plot_tree(clf, feature_names=feature_columns, class_names=clf.classes_, filled=True, rounded=True)
        plt.savefig('decision_tree.png')

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
