from tensorflow.keras.models import load_model
from data_preprocessing import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

X, y = load_data()
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = load_model('models/resnet50_model.h5')
y_pred = model.predict(X_test).ravel()
y_pred_class = (y_pred > 0.5).astype(int)

print(confusion_matrix(y_test, y_pred_class))
print(classification_report(y_test, y_pred_class))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))