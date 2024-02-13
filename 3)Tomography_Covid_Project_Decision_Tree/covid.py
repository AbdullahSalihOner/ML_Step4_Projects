import numpy as np
import PIL.Image as img
import os
import pandas as pd
#%%
# Dosya yollarını değişkenlere atadık. / We assigned file paths to variables.
covidli = "COVID/"
covidsiz = "non-COVID/"
test = "deneme/test.png"
#%%
def dosya(yol):
    return [os.path.join(yol, f) for f in os.listdir(yol)]
# covidli ve covidsiz klasörlerindeki dosyaları listeye atadık. / we put the files in the covidli and covidsiz folders into the list.

#%%
def veri_donustur(klasor_adi,sinif_adi): # covidli ve covidsiz klasörlerindeki dosyaları okuyup, normalleştirme işlemi yapıyoruz. / we read the files in the covidli and covidsiz folders and normalize them.
    goruntuler = dosya(klasor_adi)  
    goruntu_sinif = []
    
    for goruntu in goruntuler:
        goruntu_oku = img.open(goruntu).convert('L')    #resmi okuyup, gri tonlamaya çeviriyoruz. / we read the image and convert it to grayscale.
        goruntu_boyutlandirma = goruntu_oku.resize((28,28)) #resmi 28x28 boyutuna getiriyoruz. / we resize the image to 28x28.
        goruntu_normallestirme = np.array(goruntu_boyutlandirma)/255.0  #normalleştirme işlemi yapıyoruz. / we normalize the image.
        goruntu_donusturme = goruntu_normallestirme.flatten() #resmi düzleştiriyoruz. / we flatten the image.
        
        if sinif_adi == "covidli": #covidli ve covidsiz resimlerin etiketlerini ekliyoruz. / we add labels to covidli and covidsiz images.
            veriler = np.append(goruntu_donusturme,[0]) 
        elif sinif_adi == "covid_olmayan": 
            veriler = np.append(goruntu_donusturme,[1]) 
        else:
            continue
        
        goruntu_sinif.append(veriler) #etiketlenmiş resimleri listeye atıyoruz. / we put labeled images into the list.
    return goruntu_sinif
#%%

covidli_veri = veri_donustur(covidli,"covidli") #covidli ve covidsiz resimleri okuyup, normalleştirme işlemi yapıyoruz. / we read the covidli and covidsiz images and normalize them.
covidli_df = pd.DataFrame(covidli_veri)
covidli_olmayan_veri = veri_donustur(covidsiz,"covid_olmayan")
covidli_olmayan_df = pd.DataFrame(covidli_olmayan_veri) 

tum_veri = pd.concat([covidli_df,covidli_olmayan_df]) #covidli ve covidsiz resimleri birleştiriyoruz. / we combine covidli and covidsiz images.

#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 

giris = np.array(tum_veri)[:,:784]
cikis = np.array(tum_veri)[:,784]
#%%
giristrain, giristest, cikistrain, cikistest = train_test_split(giris, cikis, test_size=0.2, random_state=32) #verileri eğitim ve test olarak ayırıyoruz. / we split the data into training and testing.
model = DecisionTreeClassifier(max_depth=5) #decision tree modeli oluşturuyoruz. / we create a decision tree model.
clf = model.fit(giristrain,cikistrain) #modeli eğitiyoruz. / we train the model.
cikis_tahmin = clf.predict(giristest) #modeli test ediyoruz. / we test the model.
print("Accuracy:",metrics.accuracy_score(cikistest, cikis_tahmin))   #doğruluk oranını yazdırıyoruz. / we print the accuracy rate.
#%%

goruntu_oku1 = img.open(test).convert('L')  # test resmini okuyup, gri tonlamaya çeviriyoruz. / we read the test image and convert it to grayscale.
goruntu_boyutlandirma1 = goruntu_oku1.resize((28,28)) #test resmini 28x28 boyutuna getiriyoruz. / we resize the test image to 28x28.
goruntu_normallestirme1 = np.array(goruntu_boyutlandirma1)/255.0 #normalleştirme işlemi yapıyoruz. / we normalize the test image.
goruntu_donusturme1 = goruntu_normallestirme1.flatten() #test resmini düzleştiriyoruz. / we flatten the test image.
goruntu_donusturme1 = goruntu_donusturme1.reshape(1,-1) #test resmini yeniden boyutlandırıyoruz. / we resize the test image again.

print("Tahmin:",clf.predict(goruntu_donusturme1)) #test resmi için tahmin yapıyoruz. / we make a prediction for the test image.

if clf.predict(goruntu_donusturme1) == 0: 
    print("covidli")
    metin = "covidli"
if clf.predict(goruntu_donusturme1) == 1:
    print("covidsiz")
    metin = "covidsiz"    

# %%

import cv2
resim = cv2.imread(test)    
cv2.putText(resim, metin, (10,235), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),2)
cv2.imshow("tahmin", resim)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(cikistest, cikis_tahmin)  
print(cm)
