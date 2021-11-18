from flask import Flask,request, render_template

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

@app.route("/" ,methods=['GET'])
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def upload():
        
    if request.method == 'POST':
        imagefile=request.files['file']
        img_pa="E:\\Placements\\flask\\templates\\" + imagefile.filename
        imagefile.save(img_pa)

        model1 = load_model('C:\\Users\\Admin\\Downloads\\vggmodelfinal (1).h5')
        img1 =image.load_img(img_pa,target_size=(227,227))
        img1 = image.img_to_array(img1)
        img1 = img1/255.0
        img1 = np.expand_dims(img1,axis=0)
        pred1 = np.argmax(model1.predict(img1),axis=1)
        if pred1[0]==1:
            result="Contains Crack"
        else: 
            result="Doesn't contain crack"
        return result
    return None
    
    
if __name__ == "__main__":
    app.run(debug=True)