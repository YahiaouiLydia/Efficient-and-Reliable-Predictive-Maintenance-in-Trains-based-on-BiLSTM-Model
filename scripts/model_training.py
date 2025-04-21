from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model trained.")
    return model

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest model."""
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    print("Random Forest model trained.")
    return model

def train_svr(X_train, y_train, kernel='rbf'):
    """Train a Support Vector Regressor model."""
    model = SVR(kernel=kernel)
    model.fit(X_train, y_train)
    print("SVR model trained.")
    return model

def train_knn(X_train, y_train, n_neighbors=5):
    """Train a K-Nearest Neighbors model."""
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    print("KNN model trained.")
    return model

def train_gradient_boosting(X_train, y_train, n_estimators=100):
    """Train a Gradient Boosting model."""
    model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    print("Gradient Boosting model trained.")
    return model

def build_bilstm_model(input_shape):
    """Build a BiLSTM model."""
    inputs = Input(shape=input_shape)
    bilstm_out = Bidirectional(LSTM(64, kernel_initializer='glorot_uniform'))(inputs)
    output = Dense(1, activation='linear')(bilstm_out)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mean_squared_error',
                  metrics=['mae'])
    print("BiLSTM model built.")
    return model

def train_bilstm_model(model, X_train, y_train, X_test, y_test, epochs=15, batch_size=32):
    """Train a BiLSTM model with early stopping."""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    print("BiLSTM model trained.")
    return model, history