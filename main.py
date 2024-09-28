from tkinter import * 
import numpy as np
import pandas as pd
import joblib 


# Load Model 
model = joblib.load("SVC_Model.joblib")


# Load Data

precautaion = pd.read_csv("Dataset/precautions_df.csv")
workout = pd.read_csv("Dataset/workout_df.csv")
description = pd.read_csv("Dataset/description.csv")
medications = pd.read_csv("Dataset/medications.csv")
dieat = pd.read_csv("Dataset/diets.csv")


# Function 

# ===============Custom and helping Function================

# ===================Helper Function=====================

def helper(dis):
    # Description about Disease 
    
    descr = description[description["Disease"] == dis]["Description"]
    descr = " ".join([w for w in descr])

    # Precuations about Disease

    pre = precautaion[precautaion["Disease"] == dis][["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]]
    pre = [col for col in pre.values]

    # Meditations About Disease

    med = medications[medications["Disease"] == dis]["Medication"]
    med = [medi for medi in med.values]

    #  Dieat About Disease

    diet = dieat[dieat["Disease"] == dis]["Diet"]
    diet = [die for die in diet.values]

    wrkout = workout[workout["disease"] == dis]["workout"]

    return descr, pre, med, diet, wrkout
    
    




symptoms_dict = {'itching': 0,'skin_rash': 1,'nodal_skin_eruptions': 2,'continuous_sneezing': 3,'shivering': 4,'chills': 5,'joint_pain': 6,'stomach_pain': 7,'acidity': 8,
    'ulcers_on_tongue': 9,'muscle_wasting': 10,'vomiting': 11,  'burning_micturition': 12,  'spotting_urination': 13,  'fatigue': 14,  'weight_gain': 15,
    'anxiety': 16,  'cold_hands_and_feets': 17,  'mood_swings': 18,  'weight_loss': 19,  'restlessness': 20,  'lethargy': 21,  'patches_in_throat': 22,
    'irregular_sugar_level': 23,'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31,
    'yellowish_skin': 32,   'dark_urine': 33,   'nausea': 34,   'loss_of_appetite': 35,   'pain_behind_the_eyes': 36,   'back_pain': 37,   'constipation': 38,
    'abdominal_pain': 39,  'diarrhoea': 40,   'mild_fever': 41,  'yellow_urine': 42,  'yellowing_of_eyes': 43,  'acute_liver_failure': 44,  'fluid_overload': 45,
    'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,'malaise': 48,'blurred_and_distorted_vision': 49,   'phlegm': 50,   'throat_irritation': 51,   'redness_of_eyes': 52,
    'sinus_pressure': 53,  'runny_nose': 54,   'congestion': 55,   'chest_pain': 56,   'weakness_in_limbs': 57,   'fast_heart_rate': 58,    'pain_during_bowel_movements': 59,  
    'pain_in_anal_region': 60,    'bloody_stool': 61, 'irritation_in_anus': 62,  'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,    'obesity': 67,  
    'swollen_legs': 68,    'swollen_blood_vessels': 69,    'puffy_face_and_eyes': 70,  'enlarged_thyroid': 71,    'brittle_nails': 72,    'swollen_extremeties': 73,
    'excessive_hunger': 74,   'extra_marital_contacts': 75,   'drying_and_tingling_lips': 76,   'slurred_speech': 77,    'knee_pain': 78,   'hip_joint_pain': 79,   'muscle_weakness': 80,   'stiff_neck': 81,
    'swelling_joints': 82,  'movement_stiffness': 83,  'spinning_movements': 84,  'loss_of_balance': 85,  'unsteadiness': 86,   'weakness_of_one_body_side': 87,
    'loss_of_smell': 88,  'bladder_discomfort': 89,   'foul_smell_of_urine': 90,   'continuous_feel_of_urine': 91,   'passage_of_gases': 92,   'internal_itching': 93,
    'toxic_look_(typhos)': 94,   'depression': 95,   'irritability': 96,   'muscle_pain': 97,   'altered_sensorium': 98,   'red_spots_over_body': 99,   'belly_pain': 100,   'abnormal_menstruation': 101,
    'dischromic_patches': 102,    'watering_from_eyes': 103,    'increased_appetite': 104,    'polyuria': 105,    'family_history': 106,   'mucoid_sputum': 107,  
    'rusty_sputum': 108,   'lack_of_concentration': 109,  'visual_disturbances': 110,   'receiving_blood_transfusion': 111,   'receiving_unsterile_injections': 112,   'coma': 113,  
    'stomach_bleeding': 114,   'distention_of_abdomen': 115,   'history_of_alcohol_consumption': 116,
    'fluid_overload_1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,    'palpitations': 120,   'painful_walking': 121,'pus_filled_pimples': 122,   'blackheads': 123,    'scurring': 124, 
    'skin_peeling': 125,  'silver_like_dusting': 126,  'small_dents_in_nails': 127,  'inflammatory_nails': 128,
    'blister': 129,  'red_sore_around_nose': 130,  'yellow_crust_ooze': 131
}


disease_dict = { 15: 'Fungal infection',4: 'Allergy',16: 'GERD',9: 'Chronic cholestasis',14: 'Drug Reaction',33: 'Peptic ulcer disease',1: 'AIDS',12: 'Diabetes',17: 'Gastroenteritis', 6: 'Bronchial Asthma',
    23: 'Hypertension',30: 'Migraine',7: 'Cervical spondylosis',32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid',
    40: 'Hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold',
    34: 'Pneumonia', 13: 'Dimorphic hemorrhoids (piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia',
    31: 'Osteoarthritis', 5: 'Arthritis', 0: '(vertigo) Paroxysmal Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

# model prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))

    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1

    return disease_dict[model.predict([input_vector])[0]]




# Initialize all windows to None globally
disease_window = None
Descr_window = None
Prec_window = None
Medic_window = None
Dieat_window = None
Work_window = None

# Disease Window
def Disease_predict(predicted_disease):
    global disease_window
    disease_window = Toplevel()
    disease_window.geometry("500x300+200+50")
    disease_window.title("Disease Window")
    disease_window.config(bg = "#055B7F")
    
    disease_frame = Frame(disease_window, bg = "#055B7F")
    disease_frame.pack(padx = 10, pady = 20, fill = "x", expand = True)

    descr_label = Label(disease_frame, text="Disease: " + predicted_disease, font=("arial", 12, "bold"), bg="#055B7F", fg="white", wraplength=450, justify = "left")
    descr_label.pack(pady=10)


    
    disease_window.mainloop()

# Description Window
def Descr_predict(descr):
    global Descr_window
    Descr_window = Toplevel()
    Descr_window.geometry("500x300+200+50")
    Descr_window.title("Description Window")
    Descr_window.config(bg = "#055B7F")
    
    descr_frame = Frame(Descr_window, bg = "#055B7F")
    descr_frame.pack(padx = 10, pady = 20, fill = "x", expand = True)

    descr_label = Label(descr_frame, text="Description: " + descr, font=("arial", 12, "bold"), bg="#055B7F", fg="white", wraplength=450, justify = "left")
    descr_label.pack(pady=10)
    # Check if the previous windows exist before destroying
    if disease_window and disease_window.winfo_exists():
        disease_window.destroy()

    if Prec_window and Prec_window.winfo_exists():
        Prec_window.destroy()

    if Medic_window and Medic_window.winfo_exists():
        Medic_window.destroy()

    if Dieat_window and Dieat_window.winfo_exists():
        Dieat_window.destroy()

    if Work_window and Work_window.winfo_exists():
        Work_window.destroy()

    Descr_window.mainloop()

# Precaution Window
def Prec_predict(pre):
    global Prec_window
    Prec_window = Toplevel()
    Prec_window.geometry("500x300+200+50")
    Prec_window.title("Precaution Window")
    Prec_window.config(bg = "#055B7F")
    
    Prec_frame = Frame(Prec_window, bg = "#055B7F")
    Prec_frame.pack(padx = 10, pady = 20, fill = "x", expand = True)

     # Dynamically generate the precautions list with indices
    precautions_list = ""
    index = 1
    for pre_i in pre:  # Assuming 'pre' is a list of precaution strings
        precautions_list += f"{index}: {pre_i}\n"
        index += 1

    pre_label = Label(Prec_frame, text=f"Precautions:\n" +precautions_list, font=("arial", 12, "bold"), bg="#055B7F", fg="white", wraplength=450, justify = "left")
    pre_label.pack(pady=10)

    # Check if the previous windows exist before destroying
    if disease_window and disease_window.winfo_exists():
        disease_window.destroy()

    if Descr_window and Descr_window.winfo_exists():
        Descr_window.destroy()

    if Medic_window and Medic_window.winfo_exists():
        Medic_window.destroy()

    if Dieat_window and Dieat_window.winfo_exists():
        Dieat_window.destroy()

    if Work_window and Work_window.winfo_exists():
        Work_window.destroy()

    Prec_window.mainloop()

# Medication Window
def Medic_predict(med):
    global Medic_window
    Medic_window = Toplevel()
    Medic_window.geometry("500x300+200+50")
    Medic_window.title("Medication Window")
    Medic_window.config(bg = "#055B7F")
    
    Medic_frame = Frame(Medic_window, bg = "#055B7F")
    Medic_frame.pack(padx = 10, pady = 20, fill = "x", expand = True)

    # Dynamically generate the medication list with indices
    medication_list = ""
    index = 1
    for med_i in med:  # Assuming 'med' is a list of medication strings
        medication_list += f"{index}: {med_i}\n"
        index += 1
    

    # Display the medication text
    med_label = Label(Medic_frame, text=f"Medication:\n" + medication_list, font=("arial", 12, "bold"), bg="#055B7F", fg="white", wraplength=450, justify = "left")
    med_label.pack(pady=10)


    # Check if the previous windows exist before destroying
    if disease_window and disease_window.winfo_exists():
        disease_window.destroy()

    if Prec_window and Prec_window.winfo_exists():
        Prec_window.destroy()

    if Descr_window and Descr_window.winfo_exists():
        Descr_window.destroy()

    if Dieat_window and Dieat_window.winfo_exists():
        Dieat_window.destroy()

    if Work_window and Work_window.winfo_exists():
        Work_window.destroy()

    Medic_window.mainloop()

# Work Out Window
def Work_predict(wrkout):
    global Work_window
    Work_window = Toplevel()
    Work_window.geometry("500x300+200+50")
    Work_window.title("Work Out Window")
    Work_window.config(bg = "#055B7F")
    
    Work_frame = Frame(Work_window, bg = "#055B7F")
    Work_frame.pack(padx = 10, pady = 20, fill = "x", expand = True)

     # Dynamically generate the workout list with indices
    workout_list = ""
    index = 1
    for wrkout_i in wrkout:  # Assuming 'wrkout' is a list of workout strings
        workout_list += f"{index}: {wrkout_i}\n"
        index += 1
    

    # Display the workout text
    workout_label = Label(Work_frame, text="Workout: \n" + workout_list, font=("arial", 12, "bold"), bg="#055B7F", fg="white", wraplength=450, justify = "left")
    workout_label.pack(pady=10)

    # Check if the previous windows exist before destroying
    if disease_window and disease_window.winfo_exists():
        disease_window.destroy()

    if Prec_window and Prec_window.winfo_exists():
        Prec_window.destroy()

    if Medic_window and Medic_window.winfo_exists():
        Medic_window.destroy()

    if Dieat_window and Dieat_window.winfo_exists():
        Dieat_window.destroy()

    if Descr_window and Descr_window.winfo_exists():
        Descr_window.destroy()

    Work_window.mainloop()

# Diet Window
def Dieat_predict(diet):
    global Dieat_window
    Dieat_window = Toplevel()
    Dieat_window.geometry("500x300+200+50")
    Dieat_window.title("Diet Window")
    Dieat_window.config(bg = "#055B7F")
    
    Dieat_frame = Frame(Dieat_window,bg = "#055B7F")
    Dieat_frame.pack(padx = 10, pady = 20, fill = "x", expand = True)

     # Dynamically generate the diet list with indices
    diet_list = ""
    index = 1
    for diet_i in diet:  # Assuming 'diet' is a list of diet strings
        diet_list += f"{index}: {diet_i}\n"
        index += 1

    # Display the diet text
    diet_label = Label(Dieat_frame, text="Diet:\n" + diet_list, font=("arial", 12, "bold"), bg="#055B7F", fg="white", wraplength=450,justify="left")
    diet_label.pack(pady=10)

    # Check if the previous windows exist before destroying
    if disease_window and disease_window.winfo_exists():
        disease_window.destroy()

    if Prec_window and Prec_window.winfo_exists():
        Prec_window.destroy()

    if Medic_window and Medic_window.winfo_exists():
        Medic_window.destroy()

    if Descr_window and Descr_window.winfo_exists():
        Descr_window.destroy()

    if Work_window and Work_window.winfo_exists():
        Work_window.destroy()

    Dieat_window.mainloop()


def Predict():

    symptoms = ent1.get()
    user_symptoms = [s.strip() for s in symptoms.split(",")]
    user_symptoms = [sym.strip("[]' ") for sym in user_symptoms]
    predicted_disease = get_predicted_value(user_symptoms)
    descr, pre, med, diet, wrkout = helper(predicted_disease)

    my_precaution = []
    for i in pre[0]:
        my_precaution.append(i)

             

    

    disease_btn = Button(app, text = "Disease", font = ("Robot",16, "bold"), bg = "orange", fg = "white",bd = 4, cursor = "hand2", command =lambda: Disease_predict(predicted_disease))
    disease_btn.place(x = 60, y = 580)

    description_btn = Button(app, text = "Description", font = ("Robot",16, "bold"), bg = "blue", fg = "white",bd = 4, cursor = "hand2", command =lambda: Descr_predict(descr))
    description_btn.place(x = 200, y = 580)

    precaution_btn = Button(app, text = "Precaution", font = ("Robot",16, "bold"), bg = "DeepPink", fg = "white",bd = 4, cursor = "hand2", command =lambda: Prec_predict(my_precaution))
    precaution_btn.place(x = 390, y = 580)

    medication_btn = Button(app, text = "Medication", font = ("Robot",16, "bold"), bg = "SeaGreen", fg = "white",bd = 4, cursor = "hand2", command =lambda: Medic_predict(med))
    medication_btn.place(x = 580, y = 580)

    dieat_btn = Button(app, text = "Work Out", font = ("Robot",16, "bold"), bg = "Crimson", fg = "white",bd = 4, cursor = "hand2", command =lambda: Work_predict(wrkout))
    dieat_btn.place(x = 760, y = 580)

    workout_btn = Button(app, text = "Dieats", font = ("Robot",16, "bold"), bg = "GoldenRod", fg = "white",bd = 4, cursor = "hand2", command =lambda: Dieat_predict(diet))
    workout_btn.place(x = 940, y = 580)


    



app = Tk()
app.geometry("1100x700+50+30")
app.title("Medicine Recommendation System")
app.resizable(0,0)
app.config(bg = "#055B7F")


heading = Label(app, text = "Medicine AI Base Recommendation System",font = ("Robot",30, "bold"), fg="white", bg="#013a5c", justify = "left")
heading.pack(fill = "x", ipady=20)

frame = Frame(app, bg = "black", width = 1000, height = 300, pady = 50)
frame.place(x = 50, y = 150)

lbl1 = Label(frame, text = "Select Symptoms: ",font = ("Robot",14, "bold"), fg="white", bg="black")
lbl1.place(x = 20, y = 20)

ent1 = Entry(frame, font = ("arial",16, "bold"), width = 76)
ent1.place(x = 30, y = 60)


predict_btn = Button(frame, text = "Predict", font = ("Robot",16, "bold"), bg = "red", fg = "white", width = 70, command = Predict)
predict_btn.place(x = 30, y = 140)




app.mainloop()