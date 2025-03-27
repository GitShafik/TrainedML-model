import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns



def main():
    # Step 1: Load normalized data
    data_path = 'normalized-data.csv'
    df = pd.read_csv(data_path)
    
    # Step 2: Create binary label
    df['is_failed'] = df['value'].apply(lambda x: 1 if x < 0 else 0)
    
    # Step 3: Drop non-relevant columns (include any non-numeric columns here)
    columns_to_drop = ['is_failed', 'starttime', 'Source_File']
    X = df.drop(columns=columns_to_drop, axis=1, errors='ignore')  # Use errors='ignore' to avoid issues if columns don't exist
    y = df['is_failed']

    # Step 4: Check for remaining non-numeric columns
    non_numeric = X.select_dtypes(include=['object']).columns
    if not non_numeric.empty:
        print("Warning: Dropping non-numeric columns:", non_numeric.tolist())
        X = X.drop(columns=non_numeric)

    # Step 5: Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 6: Train logistic regression model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Step 7: Output results
    print("Model trained successfully!")
    
    # Step 8: Make predictions on test data
    y_pred = model.predict(X_test)

    # Step 9: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Random Forest Model Evaluation Metrics:")
    print(f" Accuracy:  {accuracy:.2f}")
    print(f" Precision: {precision:.2f}")
    print(f" Recall:    {recall:.2f}")
    print(f" F1 Score:  {f1:.2f}")
    
     # Step 10: Save the trained model
    model_filename = "Randomforest_model_STA.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved to: {model_filename}")
    
     # Step 11: Generate and visualize confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['Pass (0)', 'Fail (1)'],
                yticklabels=['Pass (0)', 'Fail (1)'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

if __name__ == "__main__":
        main()