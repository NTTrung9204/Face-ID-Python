from flask import Flask, request, send_file
from flask_cors import CORS
from utils.server_utils import ServerUtils

app = Flask(__name__)
CORS(app)

@app.route('/save_image', methods=['POST'])
def save_image():
    label = request.form.get('label')
    image = request.files.get('image')

    if not label:
        return{
            "message": "Label is None!",
            "status": "error"
        }

    if not image:
        return{
            "message": "No image received!",
            "status": "error"
        }
    
    print("Label:", label)
    print("Image received:", image.filename)

    try:
        ServerUtils.save_image(label, image)
        return {
            "message": "Save image successfully!",
            "status": "success"
        }
    except:
        return{
            "message": "Interal Server Error!",
            "status": "error"
        }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
