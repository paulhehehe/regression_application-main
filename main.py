from src.data.make_dataset import load_and_preprocess_data
from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_confusion_matrix
from src.features.build_features import build_features
from src.models.train_model import train_RFmodel
from src.models.predict_model import evaluate_model

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/final.csv"
    df = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    x, y = build_features(df)

    # Train the logistic regression model
    model, x_test, y_test = train_RFmodel(x, y)

    # Evaluate the model
    test_mae = evaluate_model(model, x_test, y_test)
    print(f"Mean_absolute_error: {test_mae}")

    