Files information
requirements.txt  -- Contains all required modules to execute the code
app.py            -- Streamlit based app to have a quick demo
app_api.py        -- Contains code for initializing api's for \url_parser and \query

Changes that need to be done:
    Change the db_filepath to your respected path to store the vector database for further loading
    Change model name as your requirement (llama, llama3.2, mistral, gemma,...)

How to run
    1. Install all the module present in the requirements.txt file
    2. if you wanted to have a quick demo run app.py using the command `streamlit run app.py`
    3. For api's run the app_api.py code with the command `uvicorn app_api:app --reload`
    4. After starting the api's you can use the below two commands for using \url_parser and \query
        for \url_parser  -- ` curl -X POST "http://127.0.0.1:8000/url-parser" -H "Content-Type: application/json" -d "{\"url\": \"https://your_url\"}" `
        for \query       -- ` curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d "{\"query\": \"What is the article talking about?\"}" `