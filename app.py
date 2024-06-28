from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {
    "0": "cow_1",
    "1": "cow_10",
    "2": "cow_101",
    "3": "cow_110",
    "4": "cow_112",
    "5": "cow_113",
    "6": "cow_124",
    "7": "cow_126",
    "8": "cow_127",
    "9": "cow_129",
    "10": "cow_13",
    "11": "cow_140",
    "12": "cow_157",
    "13": "cow_158",
    "14": "cow_168",
    "15": "cow_17",
    "16": "cow_179",
    "17": "cow_190",
    "18": "cow_196",
    "19": "cow_2",
    "20": "cow_201",
    "21": "cow_210",
    "22": "cow_212",
    "23": "cow_217",
    "24": "cow_223",
    "25": "cow_224",
    "26": "cow_235",
    "27": "cow_24",
    "28": "cow_246",
    "29": "cow_247",
    "30": "cow_248",
    "31": "cow_249",
    "32": "cow_252",
    "33": "cow_255",
    "34": "cow_257",
    "35": "cow_258",
    "36": "cow_259",
    "37": "cow_261",
    "38": "cow_262",
    "39": "cow_263",
    "40": "cow_264",
    "41": "cow_265",
    "42": "cow_266",
    "43": "cow_267",
    "44": "cow_268",
    "45": "cow_269",
    "46": "cow_27",
    "47": "cow_270",
    "48": "cow_271",
    "49": "cow_273",
    "50": "cow_274",
    "51": "cow_275",
    "52": "cow_279",
    "53": "cow_280",
    "54": "cow_283",
    "55": "cow_285",
    "56": "cow_286",
    "57": "cow_287",
    "58": "cow_288",
    "59": "cow_289",
    "60": "cow_292",
    "61": "cow_293",
    "62": "cow_295",
    "63": "cow_296",
    "64": "cow_298",
    "65": "cow_299",
    "66": "cow_300",
    "67": "cow_301",
    "68": "cow_302",
    "69": "cow_304",
    "70": "cow_309",
    "71": "cow_311",
    "72": "cow_312",
    "73": "cow_314",
    "74": "cow_315",
    "75": "cow_32",
    "76": "cow_320",
    "77": "cow_327",
    "78": "cow_328",
    "79": "cow_331",
    "80": "cow_333",
    "81": "cow_334",
    "82": "cow_336",
    "83": "cow_337",
    "84": "cow_34",
    "85": "cow_342",
    "86": "cow_343",
    "87": "cow_349",
    "88": "cow_35",
    "89": "cow_352",
    "90": "cow_40",
    "91": "cow_5",
    "92": "cow_55",
    "93": "cow_58",
    "94": "cow_64",
    "95": "cow_68",
    "96": "cow_69",
    "97": "cow_74",
    "98": "cow_79",
    "99": "cow_8",
    "100": "cow_80",
    "101": "cow_83",
    "102": "cow_84",
    "103": "cow_88",
    "104": "cow_90"
}


model = load_model('best_model.h5')

model.make_predict_function()

from PIL import Image
import numpy as np

def predict_label(img_path):
    img = Image.open(img_path)
    img = img.convert('L')  # Convert RGB to grayscale
    img = img.resize((128, 128))  # Resize to match model input size
    img_array = np.expand_dims(np.array(img), axis=0)  # Convert image to numpy array and add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize
    prediction = model.predict(img_array)
    predicted_class_index = prediction.argmax(axis=-1)[0]
    return dic[str(predicted_class_index)]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)