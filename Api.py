from flask import Flask, request, jsonify
import os
from DataAnalystGpt import DataAnalystGpt
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

@app.route('/24f1002354-submit-question', methods=['POST'])
def api():
    assert OPENAI_API_KEY, "OPENAI_API_KEY must be set in .env file"

    if 'questions.txt' not in request.files:
        return jsonify({'error': 'questions.txt is required'}), 400
    
    # Read the question
    question_file = request.files['questions.txt']
    question_str = question_file.read().decode('utf-8') + "\n"
    
    # Prepare a list of uploaded files (excluding questions.txt)
    attachments = {
        key: file
        for key, file in request.files.items()
        if key != 'questions.txt'
    }
    
    # Use the agent in a context manager
    with DataAnalystGpt(api_key=OPENAI_API_KEY) as agent:
        # Save attachments to agent's tempDir and note their paths
        for name, file in attachments.items():
            save_path = os.path.join(agent.tempDir, name)
            file.save(save_path)
            # Add a note to the question string about the file path
            question_str += f"\n[Attachment `{name}` saved to `{save_path}`]"
        
        # Ask the question
        try:
            answer = agent.askQuestion(question_str)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify(answer)

# WSGI entrypoint for httpd/mod_wsgi
application = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9004, debug=False)