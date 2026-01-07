import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

if os.path.exists('training_data.pkl'):
    Xtr, Xte, ytr, yte, df = joblib.load('training_data.pkl')
    print("Data loaded successfully.")
else:
    raise FileNotFoundError("Run Step 1 in the old notebook to generate 'training_data.pkl' first!")


Xtr_np = Xtr.toarray() if hasattr(Xtr, "toarray") else Xtr
Xte_np = Xte.toarray() if hasattr(Xte, "toarray") else Xte
selector = SelectKBest(score_func=f_regression, k=800)
Xtr_sel = selector.fit_transform(Xtr_np, ytr)
Xte_sel = selector.transform(Xte_np)
manual_cols = ['math_count', 'text_len', 'math_density'] 
X_manual_tr = df.loc[ytr.index, manual_cols].values
X_manual_te = df.loc[yte.index, manual_cols].values

# Concatenate
X_final_tr = np.hstack([Xtr_sel, X_manual_tr])
X_final_te = np.hstack([Xte_sel, X_manual_te])

# ==========================================
# 2. Optuna Optimization
# ==========================================
def objective(trial):
    params = {
        'objective': 'regression',
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 5.0),
        'verbosity': -1,
        'random_state': 7
    }
    model = lgb.LGBMRegressor(**params)
    score = cross_val_score(model, X_final_tr, ytr, cv=5, scoring='r2').mean()
    return score

print("Starting Optuna optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

best_params = study.best_params
with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)
print("Best params saved.")

# ==========================================
# 3. Final Model Training & Saving
# ==========================================
best_regmodel = lgb.LGBMRegressor(**best_params)
best_regmodel.fit(X_final_tr, ytr)
y_final_preds = best_regmodel.predict(X_final_te)

# Metrics
r2 = r2_score(yte, y_final_preds)
rmse = np.sqrt(mean_squared_error(yte, y_final_preds))
mae = mean_absolute_error(yte, y_final_preds)

print("\n" + "="*30)
print(f"FINAL METRICS (R2: {r2:.4f})")
print("="*30)
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")

# Save Model Artifacts
if not os.path.exists('modelReg'):
    os.makedirs('modelReg')

joblib.dump(best_regmodel, 'modelReg/modelreg.pkl')
joblib.dump(selector, 'modelReg/selector.pkl')
joblib.dump(manual_cols, 'modelReg/manual_cols.pkl')

print("\nModels successfully saved to 'modelReg/' folder.")