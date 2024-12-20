# Lang GPT

Lang GPT is a Streamlit-based application that allows users to input URLs, process their content into vectorized embeddings, and query the data using OpenAI's GPT-3.5-turbo model. This application is designed for information retrieval and question-answering tasks using advanced natural language processing.

## Features
- Load articles or books from up to three URLs.
- Split text into smaller chunks for efficient processing.
- Generate vector embeddings for the text using OpenAI's embedding model.
- Save and load vectorized data for future use.
- Query the vectorized data using OpenAI's GPT model.

## Prerequisites
- Python 3.7+
- OpenAI API Key (add it to a `.env` file as `OPENAI_API_KEY`)

## Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Add your OpenAI API Key to a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=sk-<your_api_key>
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run answers_from_links.py
   ```
3. Use the sidebar to input up to three URLs and click the "Submit" button to process the data.
4. Enter your query in the text box labeled "ASK:" and receive answers along with the sources.

## File Structure
- `answers_from_links.py`: Main application script.
- `.env`: File to store environment variables such as the OpenAI API Key.
- `requirements.txt`: List of required Python dependencies.
- `vector_index_data.pkl`: Pickle file to store processed vectorized data.

## Technologies Used
- **Streamlit**: For building the web interface.
- **LangChain**: For managing chains and embeddings.
- **OpenAI**: For generating embeddings and processing queries.
- **FAISS**: For efficient vector storage and retrieval.
- **SeleniumURLLoader**: For loading and scraping web content.

## Notes
- Ensure the URLs provided are accessible and contain readable content.
- Handle sensitive information like API keys securely and avoid sharing them publicly.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to create a pull request or raise an issue.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
Special thanks to the creators of LangChain, Streamlit, and OpenAI for providing the tools that make this project possible.
