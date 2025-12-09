# Suicide Detection API

This Python project implements a FastAPI-based API for classifying potentially suicidal text messages using machine learning. The project achieves an accuracy of 93.33%, leveraging natural language processing (NLP) and machine learning techniques to flag messages that may indicate a suicidal tendency, allowing for timely intervention.

## Project Overview

Mental health is a critical issue today, and identifying individuals who may be at risk of self-harm is essential. This project offers a FastAPI service to classify text messages as "suicide" or "non-suicide" based on the content of the message.

The service uses a simple logistic regression model trained on a dataset of labeled messages, achieving an accuracy of 93.33%. By integrating this API into various communication platforms or apps, you can flag urgent cases and provide appropriate help or direct individuals to support systems.

## Dataset

The dataset used in this project contains text messages labeled as "suicide" or "non-suicide" and can be found on [Kaggle](https://www.kaggle.com/). This dataset is used to train the machine learning model, enabling the classification of text messages based on language patterns associated with suicidal thoughts.

- **Link to dataset**: [Suicide Watch Dataset on Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)

## Technologies Used

- **FastAPI**: A modern web framework for building APIs with Python 3.7+, known for its speed and ease of use.
- **Scikit-learn**: A powerful library for machine learning in Python, used here for training the logistic regression model.
- **NLTK (Natural Language Toolkit)**: A toolkit for working with human language data in Python, used for text preprocessing and NLP tasks.
- **Matplotlib and Pandas**: Powerful tools for data analysis, used here for exploratory data analysis and data cleaning.

## Features

- **API Endpoints**: Exposes endpoints to classify text messages as suicide or non-suicide.
- **High Accuracy**: Achieves a classification accuracy of 93.33%.
- **Lightweight**: Uses a simple logistic regression model, making it fast and easy to integrate into applications.

## Installation

To get started with the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/steelandflesh2/suicide-detection.git
   cd suicide-detection
   ```

2. **Install dependencies**:
   Create a virtual environment (recommended) and install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the FastAPI server**:
   To start the API server locally, run:

   ```bash
   fastapi dev api_wrapper/main.py
   ```

   This will start the FastAPI application on `http://127.0.0.1:8000`.

## API Endpoints

### Classify Message

**POST** `/api`

* **Description**: Classify a text message as "suicide" or "non-suicide"
* **Request Body** (JSON):

  ```json
  {
    "text": "I'm feeling so lost and hopeless right now."
  }
  ```
* **Response**:

  ```json
  {
    "prediction": "suicide"
  }
  ```

### Example of using the API

Once the server is running, you can test the classification endpoint using tools like `curl`, `Postman`, or even directly in your Python code.

Example with `requests` in Python:

```python
import requests

url = "http://127.0.0.1:8000/api"
data = {
    "message": "I don't know if I can keep going anymore."
}

response = requests.post(url, json=data)
result = response.json()

if result['prediction'] == 'suicide':
    print("Warning: This message may indicate suicidal thoughts. Seek help immediately.")
else:
    print("This message seems non-suicidal.")
```

## Project Structure

```
suicide-detection/
│
├── api_wrapper/
│   ├── main.py                # FastAPI app and API endpoints
│   ├── model.pkl              # Model pickled file
│   ├── vectorizer.pkl         # TF-IDF vectorizer pickled file
│   └── label_encoder.pkl      # Label encoder pickled file
│
├── dataset.csv                # Dataset for training
|
├── notebook.ipynb             # Jupyter notebook
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── LICENSE                    # License file
```

## Improvements

There are several areas where the project can be further enhanced:

1. **Hyperparameter Tuning**: Experiment with different machine learning algorithms and hyperparameters to improve accuracy.
2. **Advanced NLP**: Implement more advanced techniques like stemming.
3. **Model Evaluation**: Evaluate the model on additional metrics (precision, recall, F1-score) to better understand its performance on imbalanced datasets.
4. **Scalability for API**: Make the API abuse-prone, implement rate-limiting and scale it to handle load.
5. **Explore other algorithms/neural networks**: Logistic regression is used due to its lightweight nature in this implementation, however advanced neural networks like RNNs, transformers etc could be used as well.

## Contribution

Contributions are welcome! If you’d like to improve this project or add new features, please feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

* The dataset was provided by [Nikhileswar Komati](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch).
