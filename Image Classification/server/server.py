from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/classify_image',methods=['GET','POST'])
def image_classify():
    image_data=request.form['image_data']

    resopnse = jsonify(util.classify_image(image_data))

    resopnse.headers.add('Access-Control-Allow-Origin','*')

    return resopnse


if __name__ == "__main__":
    print("starting server for image classification")
    util.load_saved_artifacts()
    app.run(port=5000)
