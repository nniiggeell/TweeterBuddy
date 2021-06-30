# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 21:08:56 2021

@author: njtj1
"""


from flask import Flask, request, jsonify
from flask_cors import CORS
import fake_news_model as model
import fake_news_input as func

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/api', methods=['GET'])
def hello_world():
    
    d = {}
    d['Query'] = str(func.input_func(request.args['Query']))
    return jsonify(d);

if __name__ == '__main__':
    app.run()