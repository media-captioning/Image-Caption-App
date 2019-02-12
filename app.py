# import necessary libraries
import os
from flask import (
   Flask,
   render_template,
   jsonify,
   request)
from flask_sqlalchemy import SQLAlchemy

# sys.path.append(os.path.abspath("./model"))

# from Load import *
# tokenizer = load(open('tokenizer.pkl', 'rb'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '') or "sqlite:///db/db.sqlite"
db = SQLAlchemy(app)

class Gis(db.Model):
    __tablename__ = 'gisdata'

    id = db.Column(db.Integer, primary_key=True)
    image=db.Column(db.String)

    def __repr__(self):
        return '<Gis %r>' % (self.name)

@app.before_first_request
def setup():
  # Recreate database each time
  # db.drop_all()
  db.create_all()

@app.route("/")
def log():
   return render_template("login.html")

@app.route("/auth/google")
def googleauth():
   return render_template("home.html")


@app.route("/auth/facebook")
def faceauth():
   return render_template("home.html")

@app.route("/auth/instagram")
def register():
   return render_template("home.html")


@app.route("/send", methods=["GET", "POST"])
def send():
   if request.method == "POST":
       image=request.form["image"]

       caption = Gis(image=image)
       db.session.add(caption)
       db.session.commit()

       return "Receive data"
       # return render_template("home.html")

@app.route("/api/data")
def list_captions():
   results = db.session.query(Gis.image).all()

   captions = []
   for result in results:
       captions.append({
           "image": result[0]
       })
   return jsonify(captions)


@app.route('/uploads', methods=['GET', 'POST'])
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        if request.files.get('file'):
            # read the file
            path = request.files['file']
            generate_caption(path)
            return jsonify(output)

@app.route('/predictions', methods=['GET', 'POST'])
# output = []
def predict_file(path):

    data = {"success": False}
    if request.method == 'POST':
        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(path)

            # return jsonify(path)
            generate_caption(path)
            return jsonify(output)

if __name__ == "__main__":
  app.run(debug=True)
