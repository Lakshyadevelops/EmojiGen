

# Emoji Fi-Gen

## Introduction
This project provides an interactive interface to find and generate emojis based on input text. The interface consists of an input box for entering text and two buttons: `Find` and `Generate`. When a user submits text, the system processes it through a BERT model and calculates cosine similarity to match emojis from a predefined dataset. If matching emojis are found, the top five emojis are displayed. By clicking the `Generate` button, the user can generate an emoji using an API call to a fine-tuned model on a specific emoji dataset.

## Features
- **Text Input**: Input box to enter text for emoji search.
- **Submit Button**: Processes the input text and displays matching emojis.
- **Generate Button**: Generates an emoji using a fine-tuned model via API call.

## Installation

### Prerequisites
- streamlit
- replicate
- pillow
- transformers
- gensim==3.8.3
- emoji-data==0.1.6
- scikit-learn==1.4.2
- scipy==1.10.0

### Setup
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Lakshyadevelops/EmojiGen.git
   ```
2. **Create a virtual environment**
   ```bash
   conda create -n emojigen python=3.9
   conda activate emojigen
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit Application**
   ```bash
   streamlit run main.py
   ```

## Usage

1. **Open the Application**
   Open your web browser and go to `http://localhost:8501`.

2. **Input Text**
   Enter the text you want to analyze in the input box.

3. **Find Text**
   Click the `Find` button. The text will be processed, and the top four matching emojis will be displayed if found.

4. **Generate Emoji**
   Click the `Generate` button to generate an emoji using the fine-tuned model via API call.

## How It Works

1. **Text Processing**
   - The input text is passed through a pre-trained BERT model to obtain embeddings.
   
2. **Cosine Similarity Calculation**
   - Cosine similarity is calculated between the text embeddings and the embeddings of emojis in the dataset.
   
3. **Emoji Matching**
   - The top five matching emojis based on cosine similarity are displayed.

4. **Emoji Generation**
   - Upon clicking the `Generate` button, an API call is made to a server where a fine-tuned model generates an emoji based on the input text.


## Contribution
Feel free to fork this repository and contribute by submitting a pull request.

## License
This project is licensed under the MIT License.

---

