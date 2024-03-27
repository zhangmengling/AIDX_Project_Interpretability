from flask import Flask, request, render_template, jsonify, session, redirect, url_for, send_file

import os
import sqlite3
import re
from datetime import timedelta, datetime
import time
import subprocess
import json
import base64
import io
import random
import numpy as np
# import sys
# sys.path.append("./")
from configs import pConfig, dConfig, rConfig

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(hours=1)

DB_PATH = './static/aidx.db'
pattern = r'@'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin')
def signin():
    if 'username' in session:
        return redirect(url_for('home'))
    else:
        return render_template('signin.html')

@app.route('/signin', methods=['GET', 'POST'])
def signincheck():
    username = request.json['username']
    password = request.json['password']

    # Check username and password
    if username.strip() == '' and password.strip() == '':
        # Authentication failed
        return jsonify({'success': False})
    else:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if re.search(pattern, username):
            query = "SELECT * FROM users WHERE email = ?"
        else:
            query = "SELECT * FROM users WHERE username = ?"
        cursor.execute(query, (username,))
        user = cursor.fetchone()
        conn.close()
        if user is None or user[3] != password:
            # Authentication failed
            return jsonify({'success': False})
        else:
            # Authentication successful
            session['username'] = user[1]
            session['logged_in'] = True  # 设置会话中的登录状态为True
            session.permanent = True  # 设置会话为永久有效
            return jsonify({'success': True})

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/signup', methods=['GET', 'POST'])
def signupcheck():
    username = request.json['username']
    email = request.json['email']
    password = request.json['password']

    # Check username and password
    if username.strip() == '' or email.strip() == '' or password.strip() == '': 
        # Authentication failed
        return jsonify({'success': False})
    elif re.search(pattern, username) or not re.search(pattern, email) or len(password) < 10:
        # Authentication failed
        return jsonify({'success': False})
    else:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        query1 = "SELECT * FROM users WHERE email = ?"
        cursor.execute(query1, (email,))
        user1 = cursor.fetchone()
        query2 = "SELECT * FROM users WHERE username = ?"
        cursor.execute(query2, (username,))
        user2 = cursor.fetchone()
        if user1 or user2:
            # Authentication failed
            conn.close()
            return jsonify({'success': False})
        else:
            # Authentication successful
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
            conn.commit()
            conn.close()
            return jsonify({'success': True})

@app.route('/signout')
def signout():
    session.pop('username', None)
    session.pop('logged_in', None)  # 从会话中移除登录状态
    return redirect(url_for('index'))

@app.route('/home')
def home():
    if 'username' in session:
        logged_in_username = session.get('username')
        return render_template('home.html', logged_in_username=logged_in_username)
    else:
        return redirect(url_for('signin'))

@app.route('/home', methods=['GET', 'POST'])
def searchhome():
    inputid = request.json['inputid']
    # username = request.json['username']
    username = session.get('username')
    cases = search(inputid, username)
    return jsonify({'success': True, 'cases': cases})

def search(inputid, username, dimension=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if inputid == '':
        query = "SELECT * FROM cases WHERE username = ?"
        if dimension is not None:
            query += " AND dimension = ?"
            cursor.execute(query, (username, dimension))
        else:
            cursor.execute(query, (username, ))
    else:
        if inputid[0] == 'C':
            query = "SELECT * FROM cases WHERE case_id = ? AND username = ?"
            if dimension is not None:
                query += " AND dimension = ?"
                cursor.execute(query, (inputid, username, dimension))
            else:
                cursor.execute(query, (inputid, username))
        elif inputid[0] == 'M':
            query = "SELECT * FROM cases WHERE model_id = ? AND username = ?"
            if dimension is not None:
                query += " AND dimension = ?"
                cursor.execute(query, (inputid, username, dimension))
            else:
                cursor.execute(query, (inputid, username))
        elif inputid[0] == 'D':
            query = "SELECT * FROM cases WHERE dataset_id = ? AND username = ?"
            if dimension is not None:
                query += " AND dimension = ?"
                cursor.execute(query, (inputid, username, dimension))
            else:
                cursor.execute(query, (inputid, username))
    cases = cursor.fetchall()
    conn.close()
    return cases

@app.route('/robustness')
def robustness():
    if 'username' in session:
        logged_in_username = session.get('username')
        return render_template('robustness.html', logged_in_username=logged_in_username)
    else:
        return redirect(url_for('signin'))

@app.route('/robustness', methods=['GET', 'POST'])
def postrobustness():
    operation = request.json['operation']
    username = session.get('username')
    dimension = "Robustness"
    if operation == 'search':
        inputid = request.json['inputid']
        cases = search(inputid, username, dimension)
        return jsonify({'success': True, 'cases': cases})
    elif operation == 'diagnose':
        requestData = request.get_json()
        caseId = requestData['caseInput']
        if caseId == '':
            caseId = str(time.time()).replace(".",'')
        caseId = 'C_' + caseId
        if 'fakepath' in requestData['modelFile']:
            modelFile = 'M_' + requestData['modelFile'].split('\\')[-1]
        else:
            modelFile = 'M_' + requestData['modelFile']
        if 'fakepath' in requestData['datasetFile']:
            dataFile = 'D_' + requestData['datasetFile'].split('\\')[-1]
        else:
            dataFile = 'D_' + requestData['datasetFile']
        dataSize = requestData['sizeSelect']
        # perturbation = requestData['perturbationSelect']
        # parameterSet = requestData['parameterSetSelect']
        # config = dataSize + "; " + perturbation + "; " + parameterSet
        config = "Sample Size: " + dataSize
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        while True:
            try:
                query = "INSERT INTO cases (case_id, model_id, dataset_id, username, dimension, config, start_time) VALUES (?, ?, ?, ?, ?, ?, ?)"
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute(query, (caseId, modelFile, dataFile, username, dimension, config, current_time))
                conn.commit()
                conn.close()
                break  # 插入成功，跳出循环
            except sqlite3.IntegrityError:
                caseId = 'C_' + str(time.time()).replace(".",'')

        program_b_args = ['/usr/local/opt/python@3.9/bin/python3.9', '../../TEA/aid_tutorial/diagnose.py', '-c', caseId, '-m', requestData['modelFile'].split("\\")[-1], '-d', requestData['datasetFile'].split("\\")[-1], '-ds',
                          dataSize]
        subprocess.Popen(program_b_args)

        return jsonify({'success': True})

@app.route('/rreport', methods=['GET'])
def rreport():
    if 'username' in session:
        logged_in_username = session.get('username')
        # logged_in_username = 'python'
        case_id = request.args.get('caseId')  # 获取查询参数中的 caseId 值
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        query = "SELECT * FROM cases WHERE case_id = ?"
        cursor.execute(query, (case_id,))
        case = cursor.fetchone()
        query = "SELECT * FROM robustness WHERE case_id = ?"
        cursor.execute(query, (case_id,))
        summary = cursor.fetchone()
        query = "SELECT * FROM corruption WHERE case_id = ?"
        cursor.execute(query, (case_id,))
        corruption = cursor.fetchone()
        query = "SELECT * FROM adversarial WHERE case_id = ?"
        cursor.execute(query, (case_id,))
        adversarial = cursor.fetchone()
        conn.close()
        results = {'corruption': corruption, 'adversarial': adversarial}

        # case = case + ("The model has been well tested, the results show that it is vulnerable against transferable adversarial attack.",)
        session['case_id'] = case_id
        session['dataset_file'] = case[3].split("\\")[-1].split(".zip")[0]
        cachefiles('corruption', case_id)
        cachefiles('adversarial', case_id)
        adv_labels = {'corruption': session['corruption']['label'], 'adversarial': session['adversarial']['label']}

        references = {'corruption': get_reference("corruption"), 'adversarial': get_reference("adversarial")}

        return render_template('robustnessreport.html', logged_in_username=logged_in_username, case=case, summary=summary, results=results, adv_labels=adv_labels, references=references)  # 返回报告页面
    else:
        return redirect(url_for('signin'))

@app.route('/fairness')
def fairness():
    if 'username' in session:
        logged_in_username = session.get('username')
        return render_template('fairness.html', logged_in_username=logged_in_username)
    else:
        return redirect(url_for('signin'))

@app.route('/security')
def security():
    if 'username' in session:
        logged_in_username = session.get('username')
        return render_template('security.html', logged_in_username=logged_in_username)
    else:
        return redirect(url_for('signin'))

@app.route('/explainability')
def explainability():
    if 'username' in session:
        logged_in_username = session.get('username')
        return render_template('explainability.html', logged_in_username=logged_in_username)
    else:
        return redirect(url_for('signin'))

@app.route('/explainability', methods=['GET', 'POST'])
def postexplainability():
    operation = request.json['operation']
    username = session.get('username')
    dimension = "Explainability"
    if operation == 'search':
        inputid = request.json['inputid']
        cases = search(inputid, username, dimension)
        return jsonify({'success': True, 'cases': cases})
    elif operation == 'diagnose':
        requestData = request.get_json()
        caseId = requestData['caseInput']
        if caseId == '':
            caseId = str(time.time()).replace(".",'')
        caseId = 'C_' + caseId
        if 'fakepath' in requestData['modelFile']:
            modelFile = 'M_' + requestData['modelFile'].split('\\')[-1]
        else:
            modelFile = 'M_' + requestData['modelFile']
        if 'fakepath' in requestData['datasetFile']:
            dataFile = 'D_' + requestData['datasetFile'].split('\\')[-1]
        else:
            dataFile = 'D_' + requestData['datasetFile']
        if 'layerSelect' in requestData:
            layerSize = requestData['layerSelect']
            config = "Layers: " + layerSize
        
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            while True:
                try:
                    query = "INSERT INTO cases (case_id, model_id, dataset_id, username, dimension, config, start_time) VALUES (?, ?, ?, ?, ?, ?, ?)"
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    cursor.execute(query, (caseId, modelFile, dataFile, username, dimension, config, current_time))
                    conn.commit()
                    conn.close()
                    break  # 插入成功，跳出循环
                except sqlite3.IntegrityError:
                    caseId = 'C_' + str(time.time()).replace(".",'')

            program_b_args = ['/usr/local/opt/python@3.9/bin/python3.9', '../../Interpretability_testing/interpretability/api.py', '-c', caseId, '-m', requestData['modelFile'].split("\\")[-1], '-d', requestData['datasetFile'].split("\\")[-1], '-l', layerSize]
            subprocess.Popen(program_b_args)
        else:
            clusterSize = requestData['clusterSelect']
            maxTokens = requestData['maxTokens']
            config = "Clusters: " + clusterSize + "; MaxTokens: " + maxTokens

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            while True:
                try:
                    query = "INSERT INTO cases (case_id, model_id, dataset_id, username, dimension, config, start_time) VALUES (?, ?, ?, ?, ?, ?, ?)"
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    cursor.execute(query, (caseId, modelFile, dataFile, username, dimension, config, current_time))
                    conn.commit()
                    conn.close()
                    break  # 插入成功，跳出循环
                except sqlite3.IntegrityError:
                    caseId = 'C_' + str(time.time()).replace(".",'')

            program_b_args = ['/usr/local/opt/python@3.9/bin/python3.9', '../../aisg_demo/backend/api.py', '-c', caseId, '-m', requestData['modelFile'].split("\\")[-1], '-d', requestData['datasetFile'].split("\\")[-1], '-k', clusterSize, '-t', maxTokens]
            subprocess.Popen(program_b_args)

        return jsonify({'success': True})

@app.route('/ereport', methods=['GET'])
def ereport():
    if 'username' in session:
        logged_in_username = session.get('username')
        # logged_in_username = 'python'
        case_id = request.args.get('caseId')  # 获取查询参数中的 caseId 值
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        query = "SELECT * FROM cases WHERE case_id = ?"
        cursor.execute(query, (case_id,))
        case = cursor.fetchone()

        session['case_id'] = case_id
        session['dataset_file'] = case[3].split("\\")[-1].split(".zip")[0]

        return render_template('explainabilityreport.html', logged_in_username=logged_in_username, case=case)  # 返回报告页面, , summary=summary, results=results, references=references
    else:
        return redirect(url_for('signin'))

def cachefiles(attack, case_id):
    if attack == 'corruption':
        sub_attack = 'gni'
        files = getfiles(sub_attack, case_id)
        session['corruption'] = files
    elif attack == 'adversarial':
        sub_attack = 'fgsm'
        files = getfiles(sub_attack, case_id)
        session['adversarial'] = files

def getfiles(sub_attack, case_id):
    advPath = os.path.join(pConfig.adv_path, f'{sub_attack}_{case_id}')
    adv_files =[]
    for img_file in os.listdir(advPath):
        if img_file.endswith('.png'):
            adv_files.append(img_file)
    file = random.choice(adv_files)
    advFile = os.path.join(advPath, file)

    noisePath = os.path.join(pConfig.noise_path, f'{sub_attack}_{case_id}')
    noiseFile = os.path.join(noisePath, file)

    originalPath = os.path.join(pConfig.data_path, session['dataset_file'], 'test/images')
    originalFile = os.path.join(originalPath, file.split("_")[0] + ".png")

    files = {"original":originalFile, "noise":noiseFile, "adv":advFile, "label":file.split("_")[1]}
    return files

def get_reference(table):
    # 连接到数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 执行SQL查询，获取列1的值
    query = "SELECT * FROM " + table
    cursor.execute(query)
    values = cursor.fetchall()
    values = np.array(values)
    conn.close()

    references = []
    for i in range(2,6):
        # 对列1的值进行排序
        sorted_values = sorted(values[:,i])

        # 计算70%和90%分位点的位置
        n = len(sorted_values)
        index_70th = int(n * 0.7)
        index_90th = int(n * 0.9)

        # 获取70%和90%分位点对应的值
        value_70th = sorted_values[index_70th]
        value_90th = sorted_values[index_90th]

        # 打印结果
        references.append([value_70th,value_90th])
    return references

@app.route('/get_image/<image_type>', methods=['GET'])
def get_image(image_type):
    attack = image_type.split("_")[0]
    file_type = image_type.split("_")[1]
    filename = session[attack][file_type]
    return send_file(filename, mimetype='image/png')

@app.route('/get_model', methods=['GET'])
def get_model():
    filename = pConfig.explanation_path + session["case_id"] + ".png"
    return send_file(filename, mimetype='image/png')

@app.route('/change_images', methods=['POST'])
def change_images():
    attack = request.json['attack']
    cachefiles(attack, session["case_id"])
    adv_label = session[attack]['label']
    return jsonify(success=True, adv_label=adv_label)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    upload_file(request, 'modelFile')
    return jsonify(success=True)

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    upload_file(request, 'datasetFile')
    return jsonify(success=True)

def upload_file(request, file_name):
    if file_name == 'modelFile':
        UPLOAD_FOLDER = './static/compressed/models/'
    else:
        UPLOAD_FOLDER = './static/compressed/datasets/'
    if file_name not in request.files:
        return redirect(request.url)

    file = request.files[file_name]

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        return 'File uploaded successfully.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1111)
