# ── Ridge Regression from scratch (supports any number of features) ──────────

def transpose(matrix):
    """
    Transposes a matrix — rows become columns and vice versa.
    Required to compute X^T in the Ridge formula.

    Example:
        [[1, 2],      becomes    [[1, 3],
         [3, 4]]                  [2, 4]]
    """
    return [list(row) for row in zip(*matrix)]


def matmul(A, B):
    """
    Multiplies two matrices A and B together (standard dot product).
    Used to compute X^T * X and X^T * y in the Ridge formula.

    Parameters:
        A: Matrix of shape (m x n)
        B: Matrix of shape (n x p)

    Returns:
        Result matrix of shape (m x p)
    """
    result = [[0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


def add_matrices(A, B):
    """
    Adds two matrices element-wise.
    Used to compute (X^T * X) + (λ * I).
    """
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def identity_matrix(size):
    """
    Creates an identity matrix of the given size (all zeros except 1s on diagonal).
    Used to build λI before adding it to X^T * X.

    The identity matrix ensures we penalize all weights equally in L2.
    """
    I = [[0] * size for _ in range(size)]
    for i in range(size):
        I[i][i] = 1
    return I


def inverse_matrix(matrix):
    """
    Computes the inverse of a square matrix using Gauss-Jordan elimination.
    Works for any n×n matrix, not just 2×2.

    This is the core step in the Ridge formula:
        (X^T X + λI)^-1

    Parameters:
        matrix: A square n×n matrix

    Returns:
        The inverse of the matrix

    Raises:
        ValueError: If the matrix is singular (non-invertible)
    """
    n = len(matrix)

    # Create augmented matrix [matrix | I]
    augmented = [matrix[i][:] + identity_matrix(n)[i] for i in range(n)]

    # Forward elimination and back substitution (Gauss-Jordan)
    for col in range(n):
        # Find pivot row
        pivot = None
        for row in range(col, n):
            if augmented[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            raise ValueError("Matrix is singular and cannot be inverted.")

        # Swap current row with pivot row
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

        # Normalize pivot row
        pivot_val = augmented[col][col]
        augmented[col] = [x / pivot_val for x in augmented[col]]

        # Eliminate all other rows
        for row in range(n):
            if row != col:
                factor = augmented[row][col]
                augmented[row] = [
                    augmented[row][j] - factor * augmented[col][j]
                    for j in range(2 * n)
                ]

    # Extract the right half as the inverse
    return [row[n:] for row in augmented]


def add_intercept(X):
    """
    Prepends a column of 1s to the feature matrix X.
    This allows the model to learn an intercept (bias) term.

    The intercept column of 1s is added BEFORE regularization so that
    the bias weight is NOT penalized by λ — which is the standard
    and mathematically correct behaviour for Ridge Regression.

    Example:
        [[1, 2],      becomes    [[1, 1, 2],
         [3, 4]]                  [1, 3, 4]]
    """
    return [[1] + row for row in X]


def scale_identity_for_ridge(size):
    """
    Creates an identity matrix where the top-left element (intercept position)
    is set to 0, so the intercept is NOT penalized by the λ term.

    Standard Ridge Regression should only penalize slope weights,
    not the intercept. This matrix enforces that correctly.

    Example for size=3:
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    """
    I = identity_matrix(size)
    I[0][0] = 0   # ← Do not penalize the intercept
    return I


def ridge_regression(X, y, lambda_):
    """
    Trains a Ridge Regression model using the closed-form OLS solution
    with L2 regularization.

    Ridge Formula:
        weights = (X^T X + λI)^-1 X^T y

    Where:
        X      = Feature matrix with intercept column prepended
        y      = Target vector
        λ      = Regularization strength (higher = more shrinkage)
        I      = Identity matrix (with intercept position zeroed out)

    How L2 regularization works here:
        Without λ: weights = (X^T X)^-1 X^T y  ← standard OLS
        With λ:    adds λ to the diagonal of X^T X before inverting,
                   which penalizes large weights and shrinks them towards 0.

    Parameters:
        X       (list of lists): Feature matrix, shape (n_samples x n_features)
        y       (list):          Target values, length n_samples
        lambda_ (float):         Regularization strength (λ ≥ 0)

    Returns:
        dict with:
            intercept (float): The bias term (not regularized)
            weights   (list):  The feature weights (regularized)
    """
    # Step 1: Add intercept column (column of 1s)
    X_b = add_intercept(X)

    # Step 2: Transpose X
    X_T = transpose(X_b)

    # Step 3: Compute X^T * X
    XTX = matmul(X_T, X_b)

    # Step 4: Build λI with intercept position zeroed out
    n_features  = len(XTX)
    I           = scale_identity_for_ridge(n_features)
    lambda_I    = [[lambda_ * I[i][j] for j in range(n_features)] for i in range(n_features)]

    # Step 5: Add λI to X^T * X
    XTX_plus_lambdaI = add_matrices(XTX, lambda_I)

    # Step 6: Invert (X^T X + λI)
    XTX_inv = inverse_matrix(XTX_plus_lambdaI)

    # Step 7: Compute X^T * y
    XTy = matmul(X_T, [[v] for v in y])

    # Step 8: Compute final weights = (X^T X + λI)^-1 * X^T y
    result  = matmul(XTX_inv, XTy)
    weights = [w[0] for w in result]

    return {
        "intercept": weights[0],
        "weights":   weights[1:]
    }


def predict(X, model):
    """
    Generates predictions using the learned Ridge Regression weights.

    Formula:
        y_pred = intercept + X * weights

    Parameters:
        X     (list of lists): Feature matrix to predict on
        model (dict):          Output of ridge_regression() containing
                               intercept and weights

    Returns:
        list: Predicted values for each sample in X
    """
    intercept = model["intercept"]
    weights   = model["weights"]
    return [intercept + sum(x[j] * weights[j] for j in range(len(weights))) for x in X]


def mean_squared_error(y_actual, y_predicted):
    """
    Computes Mean Squared Error between actual and predicted values.

    Formula:
        MSE = (1/n) * Σ(yi - ŷi)²

    Lower MSE = better model fit.
    """
    n = len(y_actual)
    return sum((y_actual[i] - y_predicted[i]) ** 2 for i in range(n)) / n


def r_squared(y_actual, y_predicted):
    """
    Computes R² (Coefficient of Determination).

    Formula:
        R² = 1 - (SS_res / SS_tot)

    Where:
        SS_res = Σ(yi - ŷi)²   ← residual error
        SS_tot = Σ(yi - ȳ)²    ← total variance in y

    R² = 1.0 → perfect fit
    R² = 0.0 → no better than predicting the mean
    """
    y_mean = sum(y_actual) / len(y_actual)
    ss_res = sum((y_actual[i] - y_predicted[i]) ** 2 for i in range(len(y_actual)))
    ss_tot = sum((yi - y_mean) ** 2 for yi in y_actual)
    return 1 - (ss_res / ss_tot)


# ── Example Usage ─────────────────────────────────────────────────────────────

X = [
    [1, 1],
    [1, 2],
    [2, 2],
    [2, 3]
]

y       = [6, 8, 9, 11]
lambda_ = 0.5

model  = ridge_regression(X, y, lambda_)
y_pred = predict(X, model)

print("=" * 40)
print("       Ridge Regression Results")
print("=" * 40)
print(f"  Intercept : {model['intercept']:.4f}")
print(f"  Weights   : {[round(w, 4) for w in model['weights']]}")
print(f"  MSE       : {mean_squared_error(y, y_pred):.4f}")
print(f"  R²        : {r_squared(y, y_pred):.4f}")
print("=" * 40)