import inspect
import json
import os
import shutil
import tempfile
import openai

import openai
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition

# --- Decorator for LLM-usable functions ---
llm_functions_registry = []
def llm_function(fn):
    """Decorator to mark a function as usable by the LLM and register it for function description."""
    fn._llm_function = True
    llm_functions_registry.append(fn)
    return fn

class AgenticGpt:
    DEFAULT_MODEL = "gpt-4o"
    MAX_TOKENS_PER_MESSAGE = 120000 # TODO - adjust based on model capabilities

    api_key: str
    model: str

    __tempDir: str
    finalAnswer: list|dict|None
    context: bool    

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        if not self.api_key:
            raise ValueError("OpenAI API key not provided.")
        self.client = openai.OpenAI(api_key=self.api_key)

    def __enter__(self):
        self.__tempDir = tempfile.mkdtemp(prefix="gpt_functions_")
        self.context = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.__tempDir, ignore_errors=True)
        self.context = False

    def converse(self, messages: list, functions: list[FunctionDefinition]|None = None, responseFormat = None) -> ChatCompletionMessage:
        """
        Handle a conversation with the OpenAI API, optionally using tools.
        """
        assert self.context, "AgenticGtp must be used within a context manager."

        try:
            toolsParameter: list[ChatCompletionToolParam]|openai.NotGiven = [
                ChatCompletionToolParam(
                    type="function",
                    function=functionDefinition
                ) for functionDefinition in functions
            ] if functions else openai.NotGiven()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=toolsParameter,
                response_format=responseFormat if responseFormat else openai.NotGiven()
            )
            return response.choices[0].message
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
        
    def oneshot(self, prompt: str, systemPrompt: str|None = None, responseFormat = None) -> str:
        assert self.context, "AgenticGtp must be used within a context manager."

        messages = []
        if systemPrompt:
            messages.append({"role": "system", "content": systemPrompt})
        messages.append({"role": "user", "content": prompt})

        completion = self.converse(messages, responseFormat=responseFormat)

        if not completion or not hasattr(completion, 'content'):
            raise ValueError("OpenAI response is empty or malformed.")
        responseContent = completion.content
        if not isinstance(responseContent, str):
            raise ValueError("OpenAI response content is not a string.")
        return responseContent
    
    def doDescribeFile(
        self, filePath: str, textQuery: str, systemPrompt=None, responseFormat=None
    ) -> str:
        """
        Describe an image or PDF file based on a text query.
        Args:
            file_path (str): Path to the file.
            text_query (str): Text query to ask about the file (e.g. "What is in this image?").
            systemPrompt (str): Optional system prompt to guide the LLM.
            responseFormat: Optional response format for the OpenAI API.
        Returns:
            str: Description of the file based on the text query.
        """
                # Canonicalize the file path
        filePath = os.path.abspath(filePath)
        # Ensure the file path is within the temp directory
        assert filePath.startswith(self.tempDir), "File path must be within the temp directory."
        # Get base64 encoded data
        import base64
        # Build base64 url string with file type
        import mimetypes
        mimeType, _ = mimetypes.guess_type(filePath)
        if mimeType is None:
            mimeType = 'application/octet-stream'
        data_url = f"data:{mimeType};base64,"
        with open(filePath, "rb") as f:
            image_data = f.read()
        data_url += base64.b64encode(image_data).decode("utf-8")

        # Call OpenAI API with the text query and image data
        # Different call format for images and other files
        if mimeType.startswith('image/'):
            messageContent = [
                {"type": "text", "text": textQuery},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
        else:
            messageContent = [
                {
                    "type": "file",
                    "file": {
                        "filename": os.path.basename(filePath),
                        "file_data": data_url
                    }
                },
                {"type": "text", "text": textQuery}
            ]

        messages = [
            {
                "role": "user",
                "content": messageContent
            }
        ]

        if systemPrompt:
            messages.insert(0, {"role": "system", "content": systemPrompt})

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages, # type: ignore
            response_format=responseFormat if responseFormat else openai.NotGiven()
        )
        response = response.choices[0].message
        if not response or not hasattr(response, 'content'):
            raise ValueError("OpenAI response is empty or malformed.")
        responseContent = response.content
        if not isinstance(responseContent, str):
            raise ValueError("OpenAI response content is not a string.")
        return responseContent

    @llm_function
    def describeFile(self, filePath: str, textQuery: str) -> str:
        """
        Describe an image or PDF file based on a text query.
        Args:
            filePath (str): Path to the file.
            textQuery (str): Text query to ask about the file (e.g. "What is in this image?").
        Returns:
            str: Description of the file based on the text query.
        """
        return self.doDescribeFile(
            filePath,
            textQuery
        )
    
    @classmethod
    def describeFunctions(cls):
        """
        Returns a list of function descriptions for all @llm_function methods, using their docstrings and signatures.
        The 'self' parameter is omitted from the function descriptions passed to the LLM.
        """
        functions = []
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if getattr(method, '_llm_function', False):
                sig = inspect.signature(method)
                doc = inspect.getdoc(method) or ""
                # Omit 'self' from parameters
                params = {k: {"type": "string"} for k in sig.parameters if k != 'self'}
                for k, v in sig.parameters.items():
                    if k == 'self':
                        continue
                    if v.annotation == list:
                        params[k]["type"] = "array"
                        params[k]["items"] = {"type": "number"}
                    elif v.annotation == bool:
                        params[k]["type"] = "boolean"
                        params[k]["default"] = v.default if v.default is not inspect.Parameter.empty else False
                functions.append({
                    "name": name,
                    "description": doc,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": [k for k, v in sig.parameters.items() if k != 'self' and v.default is inspect.Parameter.empty]
                    }
                })
        return functions
    
    def askQuestion(self, question: str):
        """
        Start the LLM function-use loop to answer the given question. Returns the final answer when the LLM calls provideFinalAnswer.
        """
        assert self.context, "AgenticGtp must be used within a context manager."
        self.finalAnswer = None
        # Prepare the system and user messages
        messages: list = [
            {"role": "system", "content": """
You are a computer operations specialist.
             
Use the available functions to answer the user's questions in the requested format.
When you are finished, call provideFinalAnswer with your answer.

Try to make as few calls as possible, and only call functions when necessary. Return a final answer as soon as you have all of the requested information.
"""},
            {"role": "user", "content": question}
        ]
        functions = self.describeFunctions()
        while self.finalAnswer is None:
            if len(messages) > 100:
                raise RuntimeError("Too many messages in the conversation, stopping to avoid infinite loop.")

            completion = self.converse(messages, functions=functions)
            messages.append(completion)

            # Check if the response contains a tool call
            if completion.tool_calls:
                for tool_call in completion.tool_calls:
                    fn_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    if hasattr(self, fn_name):
                        try:
                            result = getattr(self, fn_name)(**args)
                        except Exception as e:
                            result = {"error": str(e)}
                    else:
                        result = {"error": f"Unknown function: {fn_name}"}
                    messages.append({
                        "role": "tool",
                        'tool_call_id': tool_call.id,
                        "content": str(result)
                    })
            else:
                # Instruct the LLM to perform a function call
                messages.append({
                    "role": "user",
                    "content": "Please call a function to continue processing the question."
                })
            # Loop again to let LLM decide next step

        return self.finalAnswer
    
    @property
    def tempDir(self):
        assert self.context, "AgenticGtp must be used within a context manager."
        return self.__tempDir
    
    @llm_function
    def provideFinalAnswer(self, answer: list|dict):
        """
        Called by the LLM when it is finished processing all questions.

        Args:
            answer: The final answer/answers (object or list). Any value in the answer that is a string and a valid PNG file path will be loaded, base64-encoded, and replaced in the answer. Other values are left unchanged.

        Returns:
            dict: Confirmation and the type of answer stored.

        Usage for LLM:
            - To return a PNG image, include the file path as a value in the answer object or array. The agent will detect file paths, load and base64-encode the file, and replace the value with the encoded string.
            - To return a JSON or text answer, include the value directly in the answer.

        IMPORTANT FORMATTING NOTES FOR LLM:
            - If the answer is a number (e.g., 3), return just that, not as a longer string or spelled-out word (e.g., return 3, not "three" or "The answer is 3").
            - If the answer is a decimal, do not round it unless specifically requested. Return the full precision available.

        """
        import base64
        import os
        if isinstance(answer, str):
            answer = json.loads(answer)
        def encode_file_if_path(val):
            try:
                if isinstance(val, str):
                    canonicalPath = os.path.abspath(val)
                    if (
                        os.path.isfile(val) and
                        val.startswith(self.__tempDir) and
                        # Is a .png file
                        val.lower().endswith('.png')
                    ):
                        with open(canonicalPath, "rb") as f:
                            return 'data:image/png;base64,' + base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                pass
            return val
            
        def process(obj):
            if isinstance(obj, dict):
                return {k: process(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [process(v) for v in obj]
            else:
                return encode_file_if_path(obj)
        processedAnswer = process(answer)
        assert isinstance(processedAnswer, (list, dict)), "Final answer must be a list or dict."
        self.finalAnswer = processedAnswer
        return {"status": "final answer received", "type": "object_with_files_base64"}