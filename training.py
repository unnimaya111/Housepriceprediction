import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib

df = pd.read_csv("data/cleaned_house_prices.csv")

X = df.drop(['Price_INR', 'Property_ID'], axis=1)
y = df['Price_INR']
X = pd.get_dummies(X, columns=['Location', 'Property_Type', 'Furnishing'], drop_first=True)
print(X.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}
for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name}")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)
    
# Hyperparameter tuning

parameters = {
    "Random Forest": {
        "n_estimators": [50, 100],
        "max_depth": [10, 20]
    },

    "Gradient Boosting": {
        "n_estimators": [50, 100],
        "learning_rate": [0.1, 0.05],
        "max_depth": [2, 3]
    }
}

best_models = {}
for name in parameters:
    grid = GridSearchCV(
        models[name],
        parameters[name],
        cv=3,
        scoring="neg_mean_absolute_error"
    )

    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    
    print(f"\n{name}")
    print("Best settings:", grid.best_params_)
    print("Best MAE:", -grid.best_score_)
    

# Final evaluation of tuned models

print("\nFINAL TEST RESULTS")

for name, model in best_models.items():
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name}")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)
    
print("\nFINAL TEST RESULTS")

joblib.dump({
    "model": best_models["Gradient Boosting"],
    "features": X.columns.tolist()
}, "data/house_price_model.pkl")

print("Model saved successfully!")

