# Stock Market Prediction Using Natural Language Processing and LSTM

## Project Overview
This project demonstrates how **natural language processing (NLP)** combined with a **neural network architecture** can be used to model the evolution of the **S&P 500 stock market index**. It integrates **machine learning techniques** such as **text categorization using word embedding** and **sentiment analysis** to extract meaningful features from news headlines. These features, along with financial indicators, are used in a **recurrent neural network (RNN)** with an **LSTM (Long Short-Term Memory)** layer to predict stock index prices.

## Key Features
1. **Text Categorization**:
   - Headlines are categorized using word embeddings to group them into relevant themes such as Business and Economy, Politics, and Entertainment and Sports.
   
2. **Sentiment Analysis**:
   - Sentiment analysis is applied to assess the positivity, negativity, or neutrality of headlines, providing additional input to the model.

3. **Technical Indicators**:
   - Financial indicators like MACD, RSI, Bollinger Bands, and moving averages are incorporated to enhance predictive accuracy.

4. **Recurrent Neural Network**:
   - An LSTM-based model is implemented to learn from sequential data, combining headline sentiments, categories, and financial indicators.

5. **Performance Metrics**:
   - Model predictions are evaluated using:
     - **Root Mean Squared Error (RMSE)**
     - **Mean Absolute Error (MAE)**
     - **Mean Absolute Percentage Error (MAPE)**

## Results
- Models incorporating **headline sentiments** outperform those without sentiment data.
- The inclusion of **business & economy categories** leads to superior performance, with better generalization on unseen data and more accurate stock price predictions.
- The predictions are quantified using minimal error rates across the evaluation metrics.

## Keywords
- Natural Language Processing
- Word Embedding
- Sentiment Analysis
- Neural Networks
- LSTM
- Stock Forecasting


## File Structure
- `data/`: Input datasets (financial indicators, headlines). 
- `scripts/`: Python scripts for preprocessing, modeling, and evaluation.
- `README.md`: Project documentation.
- `requirements.txt`: Dependencies for the project.

Download the word2vec pre-trained Google News corpus word vector model from  https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?usp=sharing&resourcekey=0-wjGZdNAUop6WykTtMip30g and add to data folder.
