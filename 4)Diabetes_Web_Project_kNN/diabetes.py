import pickle 
with open("scaler.pkl","rb") as file: #scaler.pkl dosyasını okuyoruz. / we read the scaler.pkl file.
    load_sc=pickle.load(file)
#%%
with open("model.pkl","rb") as file: #model.pkl dosyasını okuyoruz. / we read the model.pkl file.
    load_model = pickle.load(file)
#%%
import warnings
warnings.filterwarnings('ignore') #uyarı mesajlarını kapatıyoruz. / we turn off warning messages.
#%%
import numpy as np
new_predict = load_model.predict(load_sc.transform(np.array([[1,148,72,20,1,33.6,0.427,20]]))) #yeni veri ile tahmin yapılıyor. / prediction is made with new data.
print(new_predict)
#%%
from flask import Flask,request,render_template 
app = Flask(__name__)
@app.route("/" ,methods=["GET","POST"])     # web sayfası oluşturuyoruz. / we create a web page.

def mltahmin(): # web sayfasında kullanıcıdan veri alıp, tahmin yapılıyor. / we get data from the user on the web page and make a prediction.
    tahmin = None
    tahmin1 =None
    if request.method=="POST":
        Pregnancies=float(request.form["Pregnancies"])
        Glucose=float(request.form["Glucose"])
        BloodPressure=float(request.form["BloodPressure"])
        SkinThickness=float(request.form["SkinThickness"])
        Insulin=float(request.form["Insulin"])
        BMI=float(request.form["BMI"])
        DiabetesPedigreeFunction=float(request.form["DiabetesPedigreeFunction"])
        Age=float(request.form["Age"])
        
        kullanici_verisi = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]) #kullanıcıdan alınan verileri diziye çeviriyoruz. / we convert the data taken from the user into an array.
        tahmin1 = load_model.predict(load_sc.transform(kullanici_verisi)) #kullanıcıdan alınan veri ile tahmin yapılıyor. / prediction is made with the data taken from the user.
    return render_template("web.html", tahmin=tahmin1) #web sayfasına tahmin sonucunu yazdırıyoruz. / we print the prediction result on the web page.

if __name__=="__main__": #uygulamayı çalıştırıyoruz. / we run the application.
    app.run(port=5001) #uygulamanın çalışacağı port numarasını belirtiyoruz. / we specify the port number on which the application will run.
