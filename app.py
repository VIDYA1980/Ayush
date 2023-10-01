import flask
from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import gunicorn
  
import psycopg2  # pip install psycopg2 
import psycopg2.extras
import joblib
import cv2
import numpy as np
from skimage import feature
import pandas as pd
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.spatial import distance


def create_app():
    app = Flask(__name__)
    return app

app = create_app()
print("Flask Version:", flask.__version__)

app.secret_key = "nikhil2004"
     
DB_HOST = "dpg-cka4cbaa8h2s738ha7gg-a.oregon-postgres.render.com"
DB_NAME = "smartindiahackathon"
DB_USER = "nikhil"
DB_PASS = "EcgrKuOwalOpS55vOt49oEFlBy1SUfNf"
     
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
if(conn):
    print("Connection established")
else:
    print("Error")
  
UPLOAD_FOLDER = 'static/uploads/'
TEST_FOLDER = 'static/tests/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEST_FOLDER'] = TEST_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
  
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
encoder = joblib.load('encoder.joblib')
model = load_model('dl_model.h5')
scaler = joblib.load("scaler.joblib")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def class_name(val):
    classes = {'a1': 'Land Caltrops (Bindii)', 'a2': 'Sweet Flag', 'a3': 'Common Wireweed', 'a4': 'Velvet Bean', 'a5': 'Coatbuttons',
               'a6': 'Crown Flower', 'a7': 'Shaggy Button Weed', 'a8': 'Avaram', 'a9': 'Benghal Dayflower', 'a10': 'Indian CopperLeaf', 
               'a11': 'Mexican Mint', 'a12': 'Indian Thornapple', 'a13': 'Punarnava', 'a14': 'Ivy Gourd', 'a15': 'Mexican Prickly Poppy', 
               'a16': 'Tinnevelly Senna', 'a17': 'Bristly Wild Grape', 'a18': 'Square Stalked Vine', 'a19': 'Bellyache Bush (Green)', 
               'a20': 'Prickly Chaff Flower', 'a21': 'Malabar Catmint', 'a22': 'Indian Stinging Nettle', 'a23': 'Sweet Basil', 
               'a24': 'Indian Sarsaparilla', 'a25': 'Small Water Clover', 'a26': 'Madagascar Periwinkle', 'a27': 'Indian Jujube', 
               'a28': 'Kokilaksha', 'a29': 'Trellis Vine', 'a30': 'Rosary Pea', 'a31': 'Stinking Passionflower', 
               'a32': 'Heart-Leaved Moonseed', 'a33': 'Green Chireta', 'a34': 'Cape Gooseberry', 'a35': 'Big Caltrops', 
               'a36': 'Balloon Vine', 'a37': 'Holy Basil', 'a38': 'Black-Honey Shrub', 'a39': 'Nalta Jute', 'a40': 'Country Mallow', 
               'a41': 'Asthma Plant', 'a42': 'Madras Pea Pumpkin', 'a43': 'Butterfly Pea', 'a44': 'Mountain Knotgrass', 
               'a45': 'Purple Fruited Pea Eggplant', 'a46': 'Spiderwisp', 'a47': 'Panicled Foldwing', 'a48': 'Purple Tephrosia', 
               'a49': 'Night Blooming Cereus', 'a50': 'Indian Wormwood', 's1': 'Betel Leaves', 's2': 'Amaranthus Red', 
               's3': 'Mint Leaves', 's4': 'Chinese Spinach', 's5': 'Lettuce Tree', 's6': 'Palak', 
               's7': 'Black Night Shade', 's8': 'Dwarf Copperleaf (Green)', 's9': 'Indian pennywort', 's10': 'Fenugreek Leaves', 's11': 'Celery', 
               's12': 'Coriander Leaves', 's13': 'Dwarf Copperleaf (Red)', 's14': 'Balloon Vine', 's15': 'Lagos Spinach', 's16': 'Mustard', 's17': 'Amaranthus Green', 's18': 'Lambs Quarters', 
               's19': 'Water Spinach', 's20': 'Malabar Spinach (Green)', 's21': 'Giant Pigweed', 's22': 'Curry Leaf', 
               's23': 'False Amarnath', 's24': 'Gongura', 's25':'Siru Keerai'}
    return classes[val]


def predict(data):
    scaled_data = scaler.transform(data)
    pred = model.predict(scaled_data)
    predicted_class_index = np.argmax(pred[0])
    prediction = encoder.inverse_transform([predicted_class_index])[0]
    inter_pred = str(prediction)  # Convert the prediction to a string
    return inter_pred

def create_dataset(img_path):
    names = ['area', 'perimeter', 'physiological_length', 'physiological_width', 'aspect_ratio', 'rectangularity', 'circularity',
             'mean_r', 'mean_g', 'mean_b', 'stddev_r', 'stddev_g', 'stddev_b',
             'contrast', 'correlation', 'inverse_difference_moments', 'entropy'
            ]
    data = []  

    main_img = cv2.imread(img_path)
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (25, 25), 0)
    ret_otsu, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((50, 50), np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    if len(contours) > 0:
        for cnt in contours:
            M = cv2.moments(cnt)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if area != 0:  
                rectangularity = w * h / area
                circularity = ((perimeter) ** 2) / area
            else:
                rectangularity = 0
                circularity = 0
    else:
        rectangularity = 0
        circularity = 0

    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0
        
    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)
        
    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)
        
    # Calculate LBP texture features
    lbp = feature.local_binary_pattern(gs, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    vector = [area, perimeter, w, h, aspect_ratio, rectangularity, circularity,
            red_mean, green_mean, blue_mean, red_std, green_std, blue_std,
            hist[0], hist[1], hist[2], hist[3]
            ]
        
    data.append(vector)
    
    df = pd.DataFrame(data, columns=names)
    return df 

@app.route('/')
def land():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/faq')
def faq():
    return render_template("faq.html")

@app.route('/try-us')
def try_us():
    return render_template("try_us.html")

@app.route('/leaf-detective')
def leaf_detective():
    return render_template("leafdetect.html")

@app.route('/green_check',methods=['POST'])
def green_check():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
 
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['TEST_FOLDER'], filename)
        file.save(filepath)

        # ------------------------------------------------ Prediction ------------------------------------------------------------------------
        numerical_data = create_dataset(filepath)
        pred = predict(numerical_data)
        prediction = class_name(pred)


        # ----------------------------------------------- Authenticity ------------------------------------------------------------------------
        print(pred)
        cursor.execute('''SELECT area, perimeter, physiological_length, physiological_width, aspect_ratio,
                  rectangularity, circularity, mean_r, mean_g, mean_b, stddev_r, stddev_g, stddev_b,
                  contrast, correlation, inverse_difference_moments, entropy 
                  FROM plant_data WHERE plant_number LIKE %s''', (pred,))
        matching_rows = cursor.fetchall()

        if not matching_rows:
            print("No matching data found in the database")
        else:
            similarities = []  # List to store similarities for all matching rows

        # Convert the matching rows to NumPy arrays for efficient calculation
        reference_values = np.array([list(map(float, row)) for row in matching_rows])

        # Assuming 'numerical_data' is a Pandas DataFrame with your predicted data
        predicted_values = np.array(numerical_data.values).flatten()

        for ref_row in reference_values:
            # Calculate the Euclidean distance between the feature vectors
            euclidean_distance = distance.euclidean(ref_row, predicted_values)

            # Calculate similarity (assuming smaller Euclidean distance implies higher similarity)
            max_distance = np.sqrt(sum(x ** 2 for x in ref_row))  # Maximum possible Euclidean distance
            similarity = 1 - (euclidean_distance / max_distance)

            similarities.append(similarity)

        # Calculate and print the median similarity
        authencity_val = np.median(similarities) * 100

        if(authencity_val > 60):
            authenticity = True
            auth_statement = "Yes, this is an original leaf"
        else:
            authenticity = False
            auth_statement = "No, this is not an original leaf"
        cursor.execute('''INSERT INTO authentic(plant_class, auth_value, authenticity) VALUES (%s, %s, %s)''', (prediction, authencity_val, authenticity))
        conn.commit()
        os.remove(filepath)
        return render_template('leafdetect.html',authenticity=auth_statement,filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/prediction',methods=['POST'])
def prediction():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
 
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # ------------------------------------------------ Prediction ------------------------------------------------------------------------
        numerical_data = create_dataset(filepath)
        pred = predict(numerical_data)
        prediction = class_name(pred)

        # --------------------------------------------- Querying History-------------------------------------------------------------------------------
        cursor.execute('''INSERT INTO history(plant_class) VALUES (%s)''', (prediction,))
        conn.commit()

        # ---------------------------------------------- Details of the plant/herb --------------------------------------------------------------------

        cursor.execute('''SELECT scientific_name FROM details WHERE plant_name LIKE %s''', ('%' + prediction + '%',))
        scientific = cursor.fetchone()
        if scientific:
            scientific_name = scientific[0]
        else:
            scientific_name = "Scientific Name not found"
        cursor.execute('''SELECT advantage FROM details WHERE plant_name LIKE %s''', ('%' + prediction + '%',))
        advantage_result = cursor.fetchone()
        if advantage_result:
            advantage = advantage_result[0]
        else:
            advantage = "Advantages not found"
        cursor.execute('''SELECT general_location FROM details WHERE plant_name LIKE %s''', ('%' + prediction + '%',))
        location = cursor.fetchone()
        if location:
            general_location = location[0]
        else:
            general_location = "Locations not found"
        cursor.execute('''SELECT web_link FROM details WHERE plant_name LIKE %s''', ('%' + prediction + '%',))
        web = cursor.fetchone()
        if web:
            web_link = web[0]
        else:
            web_link = "Link not available"

        #conn.commit()
        flash('Image successfully uploaded and displayed')
        print(prediction)
        return render_template('try_us.html', filename=filename, plant_name=prediction,scientific_name=scientific_name,
                               advantages=advantage,general_location=general_location,web_link=web_link)  # Pass prediction to the template
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
    

@app.route('/prediction')
def show_prediction():
    prediction = request.args.get('prediction')
    return render_template('try_us.html', prediction=prediction)

@app.route('/green-check')
def show_green():
    green_check = request.args.get('authencity')
    return render_template('leafdetect.html', authenticity=green_check)

@app.route('/map')
def map_show():
    return render_template("test.html")

if __name__ == "main":
    app.run(debug=True)