import json
import os
import duckdb
from matplotlib import pyplot as plt
import pandas as pd
import requests
from retry import retry
from AgenticGpt import AgenticGpt, llm_function

class DataAnalystGpt(AgenticGpt):
    """
    A specialized class for data analysis tasks using GPT.
    Inherits from AgenticGpt to leverage its capabilities.
    """
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)

    @llm_function
    def extractDataFromUrl(self, url: str, dataDescription: str) -> dict:
        """
        Download data from the given URL and use GPT to extract the relevant data as described.
        Saves the extracted data (as JSON) to a Parquet file in a temp directory.
        Args:
            url (str): The URL to download from.
            dataDescription (str): Description of the data to extract (e.g., 'table of cars released by year').
        Returns:
            dict: Metadata about the saved file (file path, url, dataDescription, columns)
        """
        # Guess the MIME type of the URL
        import mimetypes
        mime_type, _ = mimetypes.guess_type(url)
        # If the URL is a PDF or an image, download it to the tempDir
        if mime_type and (
            mime_type.startswith('application/pdf') or
            mime_type.startswith('image/')
        ):
            filePath = f"{self.tempDir}/data_{abs(hash(url))}.{mime_type.split('/')[-1]}"
            resp = requests.get(url)
            with open(filePath, 'wb') as f:
                f.write(resp.content)
        else:
            # Assume the URL is an HTML page
            resp = requests.get(url)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            pageContent = soup.get_text(separator="\n", strip=True)
            filePath = f"{self.tempDir}/data_{abs(hash(url))}.txt"
            # Write with utf-8 encoding to handle non-ASCII characters
            with open(filePath, 'w', encoding='utf-8') as f:
                f.write(pageContent)
        
        extractedDataInfo = self.extractDataFromFile(
            filePath,
            dataDescription
        )
        extractedDataInfo['url'] = url
        return extractedDataInfo

    @retry(tries=2, delay=1, backoff=2)
    @llm_function
    def extractDataFromFile(self, filePath: str, dataDescription: str) -> dict:
        """
        Use GPT to extract the relevant data as described from a PDF, image, or txt file.
        Saves the extracted data (as JSON) to a Parquet file in a temp directory.
        Args:
            filePath (str): Path to the file to extract data from (PDF or image).
            dataDescription (str): Description of the data to extract (e.g., 'table of cars released by year').
        Returns:
            dict: Metadata about the saved file (filePath, url, dataDescription, columns)
        """
        # Canonicalize the file path
        filePath = os.path.abspath(filePath)
        # Ensure the file path is within the temp directory
        assert filePath.startswith(self.tempDir), "File path must be within the temp directory."

        systemPrompt = f"""
You are a data extraction agent. The user will first provide a "target data description", and then an HTML document. Extract the relevant data as a JSON array of objects (one per row, with keys as column names). The data may be found in an HTML table, or may be unstructured text within the document. Include all relevant data in the output, do not truncate or skip any rows.
"""

        responseFormat = {
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "name": "ExtractedRows",
                "description": "A JSON object containing a 'rows' key with an array of row objects.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "A JSON encoded string describing each column in the row."
                            }
                        }
                    },
                    "required": ["rows"],
                    "additionalProperties": False
                }
            }
        }

        # Guess the MIME type of the file
        import mimetypes
        mime_type, _ = mimetypes.guess_type(filePath)
        if not mime_type:
            raise ValueError(f"Could not determine MIME type for file: {filePath}")

        # Text MIME type
        if mime_type.startswith('text/'):
            # Try reading as utf-8, fallback to latin-1 if decoding fails
            try:
                with open(filePath, 'r', encoding='utf-8') as f:
                    fileContent = f.read()
            except UnicodeDecodeError:
                with open(filePath, 'r', encoding='latin-1') as f:
                    fileContent = f.read()

            prompt = f"""
    Target data description:
    `````
    {dataDescription}
    `````

    Data text contents
    `````
    {fileContent}
    file_content
    """

            # Make sure the prompt is not too long
            from tiktoken import encoding_for_model
            enc = encoding_for_model(self.model)
            token_count = len(enc.encode(prompt))
            # Truncate the prompt to self.MAX_TOKENS_PER_MESSAGE
            if token_count > self.MAX_TOKENS_PER_MESSAGE:
                prompt = enc.decode(enc.encode(prompt)[:self.MAX_TOKENS_PER_MESSAGE])

            responseContent = self.oneshot(
                prompt,
                systemPrompt=systemPrompt,
                responseFormat=responseFormat
            )
        elif mime_type.startswith('application/pdf') or mime_type.startswith('image/'):
            responseContent = self.doDescribeFile(
                filePath,
                f"Target data description:\n`````\n{dataDescription}\n`````",
                systemPrompt=systemPrompt,
                responseFormat=responseFormat
            )
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")

        # Parse the response content as JSON
        extracted = json.loads(responseContent)
        rows = [json.loads(row) for row in extracted['rows']]
        df = pd.DataFrame(rows)
        columns = list(df.columns)
        file_path = f"{self.tempDir}/data_{abs(hash(filePath))}.parquet"
        df.to_parquet(file_path)
        return {"filePath": file_path, "dataDescription": dataDescription, "columns": columns}

    @llm_function
    def runDuckDbQuery(self, query: str) -> dict:
        """
        Run a DuckDB SQL query and return the results as a list of dicts.
        Args:
            query (str): The SQL query to run.
        Returns:
            dict|str: Query results (list of dicts) if the number of results is less than 20, or a CSV file path containing the output if more than 20 results.

        IMPORTANT NOTES AND INSTRUCTIONS:
        - Use STRPTIME to convert strings to DATE/TIMESTAMP fields in DuckDB.
        - Common typo: `JULIANDAY` is not a function but `JULIAN` is.
        - When querying multiple files, avoid using '*' in the file path if possible. It is better to make multiple specific queries instead of a wildcard query.
        - Any output CSVs will have no headers.

        AGGREGATION FUNCTIONS:
            #### `any_value(arg)`
            #### `arbitrary(arg)`
            #### `arg_max(arg, val)`
            #### `arg_max(arg, val, n)`
            #### `arg_max_null(arg, val)`
            #### `arg_min(arg, val)`
            #### `arg_min(arg, val, n)`
            #### `arg_min_null(arg, val)`
            #### `array_agg(arg)`
            #### `avg(arg)`
            #### `bit_and(arg)`
            #### `bit_or(arg)`
            #### `bit_xor(arg)`
            #### `bitstring_agg(arg)`
            #### `bool_and(arg)`
            #### `bool_or(arg)`
            #### `count()`
            #### `count(arg)`
            #### `countif(arg)`
            #### `favg(arg)`
            #### `first(arg)`
            #### `fsum(arg)`
            #### `geomean(arg)`
            #### `histogram(arg)`
            #### `histogram(arg, boundaries)`
            #### `histogram_exact(arg, elements)`
            #### `histogram_values(source, col_name, technique, bin_count)`
            #### `last(arg)`
            #### `list(arg)`
            #### `max(arg)`
            #### `max(arg, n)`
            #### `max_by(arg, val)`
            #### `max_by(arg, val, n)`
            #### `min(arg)`
            #### `min(arg, n)`
            #### `min_by(arg, val)`
            #### `min_by(arg, val, n)`
            #### `product(arg)`
            #### `string_agg(arg)`
            #### `string_agg(arg, sep)`
            #### `sum(arg)`
            #### `weighted_avg(arg, weight)`
            #### `corr(y, x)`
            #### `covar_pop(y, x)`
            #### `covar_samp(y, x)`
            #### `entropy(x)`
            #### `kurtosis_pop(x)`
            #### `kurtosis(x)`
            #### `mad(x)`
            #### `median(x)`
            #### `mode(x)`
            #### `quantile_cont(x, pos)`
            #### `quantile_disc(x, pos)`
            #### `regr_avgx(y, x)`
            #### `regr_avgy(y, x)`
            #### `regr_count(y, x)`
            #### `regr_intercept(y, x)`
            #### `regr_r2(y, x)`
            #### `regr_slope(y, x)`
            #### `regr_sxx(y, x)`
            #### `regr_sxy(y, x)`
            #### `regr_syy(y, x)`
            #### `sem(x)`
            #### `skewness(x)`
            #### `stddev_pop(x)`
            #### `stddev_samp(x)`
            #### `var_pop(x)`
            #### `var_samp(x)`
        """
        con = duckdb.connect()
        try:
            # Execute the query
            df = con.execute(query).df()
            if df.shape[0] > 20:
                # If more than 20 results, save to CSV file
                file_path = f"{self.tempDir}/query_results_{abs(hash(query))}.csv"
                df.to_csv(file_path, index=False, header=False)
                columnNames = list(df.columns)
                return {"file_path": file_path, "row_count": df.shape[0], "columns": columnNames}
            else:
                return {"results": df.to_dict(orient="records"), "row_count": df.shape[0], "columns": list(df.columns)}
        except Exception as e:
            return {"error": str(e)}
        finally:
            con.close()

    @llm_function
    def saveCsv(self, x: list, y: list) -> str:
        """
        Save two lists as a CSV file (with no headers) in the temp directory.
        Args:
            x (list): X-axis values.
            y (list): Y-axis values.
        Returns:
            str: File path to the saved CSV file.
        Raises:
            ValueError: If x and y are not of the same length.
        """
        if len(x) != len(y):
            raise ValueError("x and y must be of the same length.")
        df = pd.DataFrame({'x': x, 'y': y})
        file_path = f"{self.tempDir}/data_{abs(hash(str(x)+str(y)))}.csv"
        df.to_csv(file_path, index=False, header=False)
        return file_path

    @llm_function
    def drawScatterplot(self, csv_source_file: str, x_axis_label: str = 'X', y_axis_label: str = 'Y', regression: bool = True, regression_style: str = 'r--') -> str:
        """
        Draw a scatterplot of x vs y, with an optional regression line. You can either provide x and y as lists, or a CSV file path containing two columns to plot.
        Args:
            csv_source_file (str): Path to a CSV file containing two columns to plot. If provided and x/y are not given, loads data from this file.
            x_axis_label (str): Label for the X-axis.
            y_axis_label (str): Label for the Y-axis.
            regression (bool): Whether to draw a regression line.
            regression_style (str): Matplotlib line style for the regression line (e.g., 'r--' for dotted red).
        Returns:
            str: File path to the saved PNG image.
        Raises:
            ValueError: If neither x/y nor a valid parquet_file is provided.
        """
        if csv_source_file and not csv_source_file.endswith('.csv'):
            raise ValueError("csv_source_file must be a valid CSV file path ending with .csv")

        import pandas as pd
        if csv_source_file is not None:
            df = pd.read_csv(csv_source_file, header=None)
            if df.shape[1] < 2:
                raise ValueError("CSV file must have at least two columns.")
            x_vals = df.iloc[:, 0].tolist()
            y_vals = df.iloc[:, 1].tolist()
        else:
            raise ValueError("Must provide either x and y lists, or a csv_file with two columns.")
        plt.figure(figsize=(6,4))
        plt.scatter(x_vals, y_vals)
        if regression and len(x_vals) > 1:
            import numpy as np
            m, b = np.polyfit(x_vals, y_vals, 1)
            plt.plot(x_vals, [m*xi + b for xi in x_vals], regression_style, label='Regression')
            plt.legend()
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
         # Save the plot to a file
         # Use a hash of the x and y values to create a unique file name
        file_path = f"{self.tempDir}/scatterplot_{abs(hash(str(x_vals)+str(y_vals)+regression_style))}.png"
        plt.savefig(file_path, format='png')
        plt.close()
        return file_path

