import numpy as np
import matplotlib.pyplot as plt


class SimpleLinearRegression:
    """
    A Simple Linear Regression model implemented from scratch using NumPy.

    Simple Linear Regression models the relationship between a single
    independent variable (X) and a dependent variable (Y) using the equation:

        Y = slope * X + intercept

    The slope and intercept are calculated using the Ordinary Least Squares (OLS)
    method, which minimizes the sum of squared differences between the actual
    and predicted values.
    """

    def __init__(self):
        """
        Initializes the model with slope and intercept set to None.
        These will be computed once the model is trained via the fit() method.
        """
        self.slope = None
        self.intercept = None

    def fit(self, X, Y):
        """
        Trains the model by computing the slope and intercept using
        the Ordinary Least Squares (OLS) formula.

        OLS Formulas:
            slope     = Σ((Xi - X̄)(Yi - Ȳ)) / Σ((Xi - X̄)²)
            intercept = Ȳ - slope * X̄

        Where:
            X̄ = mean of X
            Ȳ = mean of Y

        Parameters:
            X (array-like): Independent variable (input features), 1D array.
            Y (array-like): Dependent variable (target values), 1D array.

        Returns:
            self: Returns the instance itself so calls can be chained.
        """
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)

        x_mean = np.mean(X)
        y_mean = np.mean(Y)

        numerator   = np.sum((X - x_mean) * (Y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)

        self.slope     = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean

        return self

    def predict(self, X):
        """
        Predicts the output values for a given set of input features
        using the learned slope and intercept.

        Formula:
            Y_pred = slope * X + intercept

        Parameters:
            X (array-like): Input values to predict on, 1D array.

        Returns:
            np.ndarray: Predicted Y values for each input in X.

        Raises:
            ValueError: If the model has not been trained yet (fit not called).
        """
        if self.slope is None or self.intercept is None:
            raise ValueError("Model is not trained yet. Call fit() before predict().")

        X = np.array(X, dtype=float)
        return self.slope * X + self.intercept

    def mean_squared_error(self, Y_actual, Y_predicted):
        """
        Computes the Mean Squared Error (MSE) between actual and predicted values.

        MSE measures the average squared difference between predictions and ground
        truth. Lower values indicate a better fit.

        Formula:
            MSE = (1/n) * Σ(Yi - Ŷi)²

        Parameters:
            Y_actual    (array-like): The true target values.
            Y_predicted (array-like): The predicted values from the model.

        Returns:
            float: The mean squared error value.
        """
        Y_actual    = np.array(Y_actual, dtype=float)
        Y_predicted = np.array(Y_predicted, dtype=float)
        return np.mean((Y_actual - Y_predicted) ** 2)

    def r_squared(self, Y_actual, Y_predicted):
        """
        Computes the R² (Coefficient of Determination) score.

        R² tells us how much of the variance in Y is explained by X.
        - R² = 1.0 → Perfect fit, the model explains all variability.
        - R² = 0.0 → The model is no better than predicting the mean of Y.
        - R² < 0.0 → The model is worse than just predicting the mean.

        Formula:
            SS_res = Σ(Yi - Ŷi)²          ← Residual Sum of Squares
            SS_tot = Σ(Yi - Ȳ)²           ← Total Sum of Squares
            R²     = 1 - (SS_res / SS_tot)

        Parameters:
            Y_actual    (array-like): The true target values.
            Y_predicted (array-like): The predicted values from the model.

        Returns:
            float: R² score between 0 and 1 (or negative if model is very poor).
        """
        Y_actual    = np.array(Y_actual, dtype=float)
        Y_predicted = np.array(Y_predicted, dtype=float)

        ss_res = np.sum((Y_actual - Y_predicted) ** 2)
        ss_tot = np.sum((Y_actual - np.mean(Y_actual)) ** 2)

        return 1 - (ss_res / ss_tot)

    def summary(self, Y_actual, Y_predicted):
        """
        Prints a summary of the trained model including its parameters
        and performance metrics.

        Displays:
            - Slope and Intercept (learned parameters)
            - Mean Squared Error (MSE)
            - R² Score

        Parameters:
            Y_actual    (array-like): The true target values.
            Y_predicted (array-like): The predicted values from the model.

        Returns:
            None
        """
        mse = self.mean_squared_error(Y_actual, Y_predicted)
        r2  = self.r_squared(Y_actual, Y_predicted)

        print("=" * 40)
        print("     Simple Linear Regression Summary")
        print("=" * 40)
        print(f"  Slope       : {self.slope:.4f}")
        print(f"  Intercept   : {self.intercept:.4f}")
        print(f"  MSE         : {mse:.4f}")
        print(f"  R² Score    : {r2:.4f}")
        print("=" * 40)

    def plot(self, X, Y, Y_predicted):
        """
        Plots the original data points alongside the fitted regression line.

        The scatter plot shows the raw data, and the red line shows the
        best-fit line the model has learned.

        Parameters:
            X           (array-like): The input feature values (X-axis).
            Y           (array-like): The actual target values (Y-axis).
            Y_predicted (array-like): The predicted values (regression line).

        Returns:
            None
        """
        plt.figure(figsize=(9, 5))
        plt.scatter(X, Y, color="steelblue", label="Actual Data", s=40, zorder=3)
        plt.plot(X, Y_predicted, color="red", linewidth=2, label="Regression Line")
        plt.title("Simple Linear Regression")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


# ── Example Usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Generate synthetic data: Y = 3X + 5 + noise
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    Y = 3 * X + 5 + np.random.randn(100) * 2

    # Train the model
    model = SimpleLinearRegression()
    model.fit(X, Y)

    # Predict
    Y_pred = model.predict(X)

    # Evaluate
    model.summary(Y, Y_pred)

    # Visualize
    model.plot(X, Y, Y_pred)