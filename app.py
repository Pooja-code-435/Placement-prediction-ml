from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load saved model and label encoders
with open("place-rf-model.pkl", "rb") as file:
    place_model = pickle.load(file)

with open("lb-activities.pkl", "rb") as file:
    lb_act = pickle.load(file)
    

with open("lb1-training.pkl", "rb") as file:
    lb1_tra = pickle.load(file)
    

with open("lb2-status.pkl", "rb") as file:
    lb2_sta = pickle.load(file)


def predict_placement(cgpa=8.0, Internships=1, Projects=2, Workshopscertifications=1,
                      AptitudeTestScore=70, SoftSkillsRating=7, ExtracurricularActivities="yes",
                      PlacementTraining="yes", SSC_Marks=85, HSC_Marks=80):

    lst = []

    # Add numeric features
    lst.append(float(cgpa))
    lst.append(int(Internships))
    lst.append(int(Projects))
    lst.append(int(Workshopscertifications))
    lst.append(float(AptitudeTestScore))
    lst.append(int(SoftSkillsRating))
    lst.append(float(SSC_Marks))
    lst.append(float(HSC_Marks))

    PlacementTraining = str(PlacementTraining).strip().lower()
    if PlacementTraining not in lb1_tra.classes_:
        return "Error: Invalid value for Placement Training"
    placement_training_encoded = lb1_tra.transform([PlacementTraining])

    ExtracurricularActivities = str(ExtracurricularActivities).strip().capitalize()
    if ExtracurricularActivities not in lb_act.classes_:
        return "Error: Invalid value for Extracurricular Activities"
    extracurricular_encoded = lb_act.transform([ExtracurricularActivities])
    
    
    lst += list(placement_training_encoded)
    lst += list(extracurricular_encoded)

    # Predict
    result = place_model.predict([lst])
    print("Raw model output:", result)

    return "Student is likely to be placed" if result[0] == 1 else "Student is unlikely to be placed"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contact.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            cgpa = float(request.form.get("cgpa"))
            internships = int(request.form.get("internships"))
            projects = int(request.form.get("projects"))
            certificate = int(request.form.get("certificate"))
            aptitude_score = float(request.form.get("aptitude_score"))
            soft_skills = int(request.form.get("soft_skills"))
            extracurricular = request.form.get("Extracurricular").strip().lower()
            training = request.form.get("Training").strip().lower()
            ssc_marks = float(request.form.get("ssc_marks"))
            hsc_marks = float(request.form.get("hsc_marks"))
        except (ValueError, AttributeError):
            return render_template("predict.html", prediction="Error: Invalid input values. Please check your entries.")

        result = predict_placement(
            cgpa=cgpa,
            Internships=internships,
            Projects=projects,
            Workshopscertifications=certificate,
            AptitudeTestScore=aptitude_score,
            SoftSkillsRating=soft_skills,
            ExtracurricularActivities=extracurricular,
            PlacementTraining=training,
            SSC_Marks=ssc_marks,
            HSC_Marks=hsc_marks
        )

        print(result)
        return render_template("predict.html", prediction=result)

    return render_template("predict.html")


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=8000)
