

# Emoji Finder and Generator

## Introduction
This project provides an interactive interface to find and generate emojis based on input text. The interface consists of an input box for entering text and two buttons: `Submit` and `Next`. When a user submits text, the system processes it through a BERT model and calculates cosine similarity to match emojis from a predefined dataset. If matching emojis are found, the top four emojis are displayed. By clicking the `Next` button, the user can generate an emoji using an API call to a fine-tuned model on a specific emoji dataset.

## Features
- **Text Input**: Input box to enter text for emoji search.
- **Submit Button**: Processes the input text and displays matching emojis.
- **Next Button**: Generates an emoji using a fine-tuned model via API call.

## Installation

### Prerequisites
- Python 3.8 or higher
- Flask
- Transformers (HuggingFace)
- NumPy
- Requests

### Setup
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/emoji-finder-generator.git
   cd emoji-finder-generator
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask Application**
   ```bash
   export FLASK_APP=app.py
   flask run
   ```

## Usage

1. **Open the Application**
   Open your web browser and go to `http://localhost:5000`.

2. **Input Text**
   Enter the text you want to analyze in the input box.

3. **Submit Text**
   Click the `Submit` button. The text will be processed, and the top four matching emojis will be displayed if found.

4. **Generate Emoji**
   Click the `Next` button to generate an emoji using the fine-tuned model via API call.

## How It Works

1. **Text Processing**
   - The input text is passed through a pre-trained BERT model to obtain embeddings.
   
2. **Cosine Similarity Calculation**
   - Cosine similarity is calculated between the text embeddings and the embeddings of emojis in the dataset.
   
3. **Emoji Matching**
   - The top four matching emojis based on cosine similarity are displayed.

4. **Emoji Generation**
   - Upon clicking the `Next` button, an API call is made to a server where a fine-tuned model generates an emoji based on the input text.

## API Details
The `Next` button triggers an API call to generate the emoji. Ensure the API endpoint is correctly configured in your application.

### Sample API Call
```python
import Replicate from 'replicate';
const replicate = new Replicate();

const input = {
    prompt: "A TOK emoji of a man",
    apply_watermark: false
};

const output = await replicate.run("fofr/sdxl-emoji:dee76b5afde21b0f01ed7925f0665b7e879c50ee718c5f78a9d38e04d523cc5e", { input });
console.log(output)
//=> ["https://replicate.delivery/pbxt/a3z81v5vwlKfLq1H5uBqpVm...
```

## Contribution
Feel free to fork this repository and contribute by submitting a pull request.

## License
This project is licensed under the MIT License.

---

