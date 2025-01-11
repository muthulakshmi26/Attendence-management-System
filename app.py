import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging
import traceback

#### Defining Flask App
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(0)
except:
    cap = None

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,ID,Time')

#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

#### extract the face from an image
def extract_faces(img):
    if img is not None:  # Check if the image is not empty
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv', parse_dates=['Time'])
    df['Date'] = datetoday2  # Add Date column with the formatted date
    df['Time'] = pd.to_datetime(df['Time']).dt.strftime('%H:%M:%S')


    # Check if the person was present or absent based on time (before or after 10:00 AM)
    df['Status'] = df['Time'].apply(lambda t: 'Absent' if datetime.strptime(t, "%H:%M:%S").hour >= 10 else 'Present')
    names = df['Name']
    IDs = df['ID']
    times = df['Time']
    dates = df['Date']
    statuses = df['Status']
    l = len(df)
    return names, IDs, times, dates, statuses, l

#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['ID']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

# New function to generate the attendance graph for an individual user
def show_user_graph(name):
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    user_df = df[df['Name'] == name]

    if user_df.empty:
        return None

    plt.figure(figsize=(10, 6))
    plt.bar(user_df['Time'], user_df['ID'])
    plt.xlabel('Time')
    plt.ylabel('ID')
    plt.title(f'Attendance for {name} on {datetoday2}')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convert the graph to base64 format
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return graph_data


# Our main page
@app.route('/')
def home():
    names, rolls, times, dates, statuses, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, dates=dates, statuses=statuses, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/start', methods=['GET'])
def start():
    try:
        if 'face_recognition_model.pkl' not in os.listdir('static'):
            return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2,
                                   mess='There is no trained model in the static folder. Please add a new face to continue.')

        ret = True
        while ret:
            ret, frame = cap.read()
            faces = extract_faces(frame)
            if len(faces) > 0:  # Check if faces are detected
                (x, y, w, h) = faces[0]  # Access the first detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

        # Update the attendance data and extract it
        names, rolls, times, dates, statuses, l = extract_attendance()

        return render_template('home.html', names=names, rolls=rolls, dates=dates, times=times, statuses=statuses, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2)

    except Exception as e:
        traceback.print_exc()
        return str(e)

# This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    while 1:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, dates, statuses, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, dates=dates, statuses=statuses, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)

# This function will show the attendance graph for an individual user
@app.route('/view_graph/<username>', methods=['GET'])
def view_graph(username):
    graph_data = show_user_graph(username)
    return render_template('graph.html', username=username, graph_data=graph_data)

### Our main function which runs the Flask App
if __name__ == '__main__':
    if cap is None:
        print("Camera not available. Exiting...")
    else:
        app.run(debug=True)
