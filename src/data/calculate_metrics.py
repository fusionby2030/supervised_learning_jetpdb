def calculate_r2(y_true, y_pred): 
    return 1 - ( sum((y_true - y_pred)**2) / sum((y_true - y_true.mean())**2)  )

def calculate_mse(y_true, y_pred): 
    return (sum((y_true - y_pred)**2)) / len(y_pred)

def calculate_mape(y_true, y_pred): 
    epsilon = 1e-7
    return (sum(abs(y_true - y_pred) / (y_true)))  / len(y_true)

def calculate_rmse(y_true, y_pred): 
    return calculate_mse(y_true, y_pred)**(1.0/2)

def calculate_metrics(y_true, y_pred): 
    return calculate_r2(y_true, y_pred), calculate_mse(y_true, y_pred), calculate_mape(y_true, y_pred), calculate_rmse(y_true, y_pred)
