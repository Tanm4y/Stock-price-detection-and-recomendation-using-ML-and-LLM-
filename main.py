from flask import Flask, render_template, request, redirect, session, url_for, jsonify 
import mysql.connector as ms 
import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt 
import base64 
from io import BytesIO 
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler 
from datetime import timedelta 
import yfinance as yf 
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

import pymysql

# Connect to the database using pymysql
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="ANish@25",
    database="stock"
)

# Check if the connection is successful
if conn.open:
    print("Successfully connected to the database.")
else:
    print("Failed to connect to the database.")

# Close the connection after use
conn.close()


mc = conn.cursor()

app = Flask(__name__) 
stock_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK'] 
start_date = '2020-01-01' 
end_date = '2024-11-10' 
folder_path = 'C:\Users\anish\OneDrive\Documents\Desktop\Anish\B1+TB1-FACE - FACE (APT) - ACAD\stock_data'  # Adjust this path based on your system 
os.makedirs(folder_path, exist_ok=True) 
 
def predict_next_close(symbol): 
    try: 
        data = pd.read_csv(f'D:/Anish/B1+TB1-FACE - FACE (APT) - ACAD/stock_data/{symbol}_predictions.csv') 
        print(f"Prediction data for {symbol} loaded successfully.") 
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce') 
         
        latest_prediction = data.sort_values(by='Date').iloc[-1] 
        predicted_price = latest_prediction['Predicted_Close'] 
        prediction_date = latest_prediction['Date'] 
         
        return { 
            "symbol": symbol, 
            "date": prediction_date.strftime('%Y-%m-%d'), 
            "predicted_price": round(predicted_price, 2) 
        } 
    except Exception as e: 
        print(f"Error in fetching prediction for {symbol}: {e}") 
        return {"error": str(e)} 
 

class StockAdvisor:
    def init(self, csv_path):
        # Load CSV data
        self.df = pd.read_csv(csv_path)
        
        # Setup the model
        self.model = ChatOpenAI(
            model="meta-llama/llama-3.2-3b-instruct:free",
            openai_api_key="your-api-key",
            openai_api_base="https://openrouter.ai/api/v1",
        )
        
        # Initialize workflow
        self.workflow = StateGraph(state_schema=MessagesState)
        self.setup_workflow()
        
    def get_stock_data(self, query):
        """Retrieve relevant stock data based on query"""
        try:
            if 'symbol' in query.lower():
                symbol = query.split()[-1].upper()
                return self.df[self.df['Symbol'] == symbol].to_dict('records')
            else:
                return self.df[self.df.astype(str).apply(lambda x: x.str.contains(query, case=False)).any(axis=1)].head().to_dict('records')
        except Exception as e:
            return f"Error retrieving data: {str(e)}"

    def setup_workflow(self):
        # Create prompt template with context
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a stock investment advisor. 
            Analyze the provided stock data and give recommendations. 
            Use the data provided in the context for your analysis.
            If specific data is not available, mention that in your response."""),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Context: {context}")
        ])

        # Define the model call function
        def call_model(state):
            # Get the latest query
            query = state["messages"][-1].content
            
            # Retrieve relevant stock data
            context = self.get_stock_data(query)
            
            # Generate response using prompt template
            response = self.prompt.pipe(self.model).invoke({
                "messages": state["messages"],
                "context": str(context)
            })
            
            return {"messages": response}

        # Set up the workflow graph
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", call_model)
        
        # Add memory
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

    def get_chatbot_response(self, user_query):
        config = {"configurable": {"thread_id": "abc123"}}
        input_messages = [HumanMessage(content=user_query)]
        try:
            output = self.app.invoke({"messages": input_messages}, config)
            return output["messages"][-1].content  # Return the chatbot response
        except Exception as e:
            return f"Error: {str(e)}"

# Instantiate StockAdvisor
advisor = StockAdvisor('portfolio_data.csv')

# Flask routes
@app.route('/chat')
def chat_page():
    return render_template("chat.html")

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json.get("message")  # Get user message from JSON
    response = advisor.get_chatbot_response(user_message)  # Get chatbot response from StockAdvisor
    return jsonify({"response": response})

@app.route('/') 
def main_page(): 
    return render_template("landing.html") 
@app.route('/signup') 
def signup_page(): 
    return render_template("signup.html") 
@app.route('/dashboard', methods=['POST']) 
def enter_details(): 
    if request.method == 'POST': 
        uname = request.form.get('username') 
        passwd = request.form.get('password') 
        email = request.form.get('email') 
         
        if not uname or not passwd or not email: 
            err = "All fields are required." 
            return render_template("login.html", err=err) 
         
        mc.execute("SELECT uname FROM users WHERE username=%s", (uname,)) 
        result = mc.fetchall() 
        conn.commit()    
        if result: 
            err = 'Username already exists' 
            return render_template("login.html", err=err) 
        else: 
            mc.execute("INSERT INTO users (username, passwd, email) VALUES (%s, %s, %s)", (uname, passwd, 
email)) 
            conn.commit() 
            return render_template("option.html", result=result) 
 
@app.route('/login') 
def login_page(): 
    return render_template("login.html") 
 
@app.route('/start/dashboard', methods=['POST']) 
def dashboard_page(): 
    if request.method == 'POST': 
        global uname 
        uname = request.form['username'] 
        passwd = request.form['password'] 
        mc.execute("SELECT * FROM user WHERE username=%s AND passwd=%s", (uname, passwd)) 
        result = mc.fetchall() 
        conn.commit() 
         
        if result != []: 
            return render_template("option.html", result=result) 
        else: 
            err = "Invalid username or password!" 
            return render_template("login.html", err=err) 
 
@app.route('/predict', methods=['GET', 'POST']) 
def predict(): 
    error = "" 
    prediction = None 
     
    if request.method == 'POST': 
        symbol = request.form.get('symbol') 
         
        if symbol not in stock_symbols: 
            error = "Symbol not supported" 
        else: 
            prediction = predict_next_close(symbol) 
            if 'error' in prediction: 
                error = prediction['error'] 
            else: 
                prediction = { 
                    "symbol": prediction["symbol"], 
                    "date": prediction["date"], 
                    "predicted_price": prediction["predicted_price"] 
                } 
     
    return render_template("predict.html", error=error, prediction=prediction) 
 
 
@app.route('/home') 
def home(): 
    return render_template('option.html') 

@app.route('/chat')
def chat_page():
    return render_template("chat.html")

# @app.route('/get_response', methods=['POST'])
# def get_response():
#     user_message = request.json.get("message")  
#     response = generate_chatbot_response(user_message)
#     return jsonify({"response": response})

# def generate_chatbot_response(message):

#     # Example placeholder respythoponse
#     return f"You said: {message}"

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5001, debug=True)