import os, re, joblib, json
import numpy as np
import scipy.sparse as sp
from flask import Flask, render_template, request

app = Flask(__name__)


clf_model = joblib.load('models/model_best.joblib')
tfidf = joblib.load('models/tfidf.joblib') 
le = joblib.load('models/label_encoder.joblib')
reg_model = joblib.load('modelReg/modelreg.pkl')
selector = joblib.load('modelReg/selector.pkl')

with open('models/keywords.json', 'r') as f:
    data = json.load(f)
    hard_keywords = data['hard_keywords']    
    medium_keywords = data['medium_keywords'] 

hard_topics = [
    'convex hull', 'mobius', 'segment tree', 'flow', 'centroid', 
    'geometry', 'gcd', 'mex', 'dynamic programming', 
    'modulo', 'bitwise', 'graph', 'expected value','permutations',
    'xor','shortest path','grid','query','range','range query'
]
mathsym = ['$', '^', '{', '}', '_', '\\', '=', '<', '>','*']

def upd(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = " ".join(text.split())
    return text

def get_manual_features(combined_text):
    hard_topic_count = 0
    for j in range(len(hard_topics)):
        topic = hard_topics[j]
        if topic in combined_text: 
            hard_topic_count += 1
    is_short_statement = int(len(combined_text) < 300)
    text_len = len(combined_text)
    math_count = 0
    for j in range(len(mathsym)):
        sym = mathsym[j]
        math_count += combined_text.count(sym) 
    math_density = math_count / (len(combined_text) + 1)
    has_high_constraints = int(bool(re.search(r'10\^5|10\^9|1000000007|1e9|1e5', combined_text)))
    high_difficulty_signal_count = sum(1 for word in hard_keywords if word in combined_text)
    medium_signal_count = sum(1 for word in medium_keywords if word in combined_text)
    X_manual = [
        hard_topic_count, 
        math_density, 
        text_len, 
        high_difficulty_signal_count, 
        is_short_statement, 
        has_high_constraints, 
        medium_signal_count
    ]
    X_manual[-1] *= 5 
    
    return np.array(X_manual).reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    p_desc = request.form.get('prob_desc', '')
    i_desc = request.form.get('input_desc', '')
    o_desc = request.form.get('output_desc', '')
    combined_text = upd(f"{p_desc} {i_desc} {o_desc}")
    X_tfidf = tfidf.transform([combined_text])     
    X_manual_7 = get_manual_features(combined_text)  
    X_full_3007 = sp.hstack((X_tfidf, X_manual_7))
    class_idx = clf_model.predict(X_full_3007)
    label = le.inverse_transform(class_idx)[0]
    X_selected_800 = selector.transform(X_full_3007)
    X_final_807 = sp.hstack((X_selected_800, X_manual_7))
    score = reg_model.predict(X_final_807)[0]
    
    return render_template('index.html', 
                           label=label, 
                           score=round(score, 2),
                           p=p_desc, i=i_desc, o=o_desc)

if __name__ == '__main__':
    app.run(debug=True)