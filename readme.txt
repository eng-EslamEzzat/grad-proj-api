1- Install the requrements, open cmd on the same directory and run this commands:

    pip install -r requirements


2- Run the application:

    uvicorn main:app --reload


3-  Request the POST API on "http://127.0.0.1:8000/prediction/word"

        {
            file: --uploadedFile (only video)
        }

    Request the POST API on "http://127.0.0.1:8000/prediction/number"

        {
            file: --uploadedFile (only image)
        }

    Request the POST API on "http://127.0.0.1:8000/prediction/alphabet"

        {
            file: --uploadedFile (only image)
        }