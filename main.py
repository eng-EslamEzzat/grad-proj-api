from fastapi import FastAPI, File, UploadFile
from prediction_script_words import predict_video
from prediction_script_numbers import predict_number
from prediction_script_alphabet import predict_alphabet
import os
app = FastAPI()


@app.post("/prediction/word")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(os.path.join('data', file.filename), 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    result = predict_video(os.path.join('data', file.filename))

    return result

@app.post("/prediction/number")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(os.path.join('data', file.filename), 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    result = predict_number(os.path.join('data', file.filename))
    return result


@app.post("/prediction/alphabet")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(os.path.join('data', file.filename), 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    result = predict_alphabet(os.path.join('data', file.filename))
    return result