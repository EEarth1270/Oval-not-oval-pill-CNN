from sklearn.metrics import classification_report, classification, accuracy_score
import pandas as pd
import sys

df = pd.read_csv('test_df.csv')
class_label =['OVAL','CAPSULE']
with open('cf.txt', 'w') as f:
    sys.stdout = f
    a = classification_report(df['splshape_text'],df['Predict_shape'],target_names=class_label)
    print(a)
