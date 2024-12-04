from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from realtime_detection import process_frame_for_prediction
import uvicorn

app = FastAPI()

# Enable CORS for frontend app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global variable to track the concatenated text and last character
predicted_text = ""
last_character = None

# API to process a single video frame and return the predicted character
@app.post("/predict_frame")
async def predict_frame(frame: UploadFile = File(...)):
    global predicted_text, last_character

    # Read uploaded file and process it for prediction
    content = await frame.read()
    predicted_character = process_frame_for_prediction(content)

    # Handle special actions like Clear, Space, and Backspace
    if predicted_character == "Clear":
        predicted_text = ""
    elif predicted_character == "Space":
        predicted_text += " "
    elif predicted_character == "Back Space":
        predicted_text = predicted_text[:-1]
    elif predicted_character:
        predicted_text += predicted_character

    last_character = predicted_character

    return {
        "predicted_text": predicted_text,
        "predicted_character": predicted_character
    }

# API to reset the backend state
@app.get("/reset_capture")
def reset_capture():
    global predicted_text, last_character
    predicted_text = ""
    last_character = None
    return {"message": "Capture reset successfully"}

# API to clear the current text
@app.get("/clear_text")
def clear_text():
    global predicted_text
    predicted_text = ""
    return {"message": "Text cleared successfully"}

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
