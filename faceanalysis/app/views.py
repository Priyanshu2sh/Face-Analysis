from django.shortcuts import render, HttpResponseRedirect
from.models import Upload
from django.contrib import messages
from django.conf import settings
import os
import base64
from django.core.files.base import ContentFile

# Create your views here.
def home(request):
    if request.method == 'POST':

        if request.POST.get('canvas_image_data'):
            canvas_image_data = request.POST.get('canvas_image_data')
        
            # Extract image data from base64 string
            format, imgstr = canvas_image_data.split(';base64,') 
            ext = format.split('/')[-1]
            img_data = ContentFile(base64.b64decode(imgstr), name=f"canvas_image.{ext}")
        
            upload=Upload(image=img_data)
            upload.save()
            uploaded_image_path = os.path.join(settings.MEDIA_ROOT, 'images', img_data.name)
            result = face_analysis(uploaded_image_path)
            print(result)
            print(type(result))
            if result=='none':
                messages.error(request,'No faces detected in the image!')
                return HttpResponseRedirect('/')
            else:
                return render(request, 'app/result.html',{'result':result})

        elif request.FILES.get('imagefile'):
            uploaded_image=request.FILES['imagefile']
            upload=Upload(image=uploaded_image)
            upload.save()
            uploaded_image_path = os.path.join(settings.MEDIA_ROOT, 'images', uploaded_image.name)
            result = face_analysis(uploaded_image_path)
            if result=='none':
                messages.error(request,'No faces detected in the image!')
                return HttpResponseRedirect('/')
            else:
                return render(request, 'app/result.html',{'result':result})

        else:
            messages.warning(request,'You have not uploaded any image')
            return HttpResponseRedirect('/')

    else:
        return render(request, 'app/home.html')
    

def face_analysis(uploaded_image_path):
    import numpy as np
    import tensorflow as tf
    import random

    # Set random seed for numpy
    np.random.seed(42)

    # Set random seed for Python random module
    random.seed(42)

    # Set random seed for TensorFlow
    tf.random.set_seed(42)

    import cv2
    from keras.models import Model
    from keras.layers import Dense, Dropout, Flatten
    from keras.applications.vgg16 import VGG16
    from keras.preprocessing.image import img_to_array

    # Load the face detection cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image
    image_path = uploaded_image_path
    image = cv2.imread(image_path)

    # Check if the image is not empty
    if image is not None:
        # Convert image to grayscale for face detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if only one face is detected
        if len(faces) >= 1:
            # Preprocess the image
            image = cv2.resize(image, (224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = image / 255.0

            # Load the VGG16 model (without the top layers)
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

            # Add custom layers to the model
            x = base_model.output
            x = Flatten()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(7, activation='sigmoid')(x)

            # Create the final model
            model = Model(inputs=base_model.input, outputs=x)

            # Freeze the base model layers
            for layer in base_model.layers:
                layer.trainable = False

            # Compile the model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Make predictions
            predictions = model.predict(image)[0]

            # Define helper functions to get the descriptive words
            def get_sentiment_description(score):
                if score < 0.4:
                    return "SENTIMENTS RESULT - Very Low sensitivity - Your sensitivity level is very low"
                elif 0.4 <= score <= 0.6:
                    return "SENTIMENTS RESULT - Normal sensitivity - Your sensitivity level is normal"
                else:
                    return "SENTIMENTS RESULT - High sensitivity - Your sensitivity level is high"

            def get_mental_activity_description(score):
                if score < 0.4:
                    return "MENTAL ACTIVITY RESULT - Low confidence - Your confidence level is lower than normal"
                elif 0.4 <= score <= 0.6:
                    return "MENTAL ACTIVITY RESULT - Moderate confidence - Your confidence level is moderate"
                else:
                    return "MENTAL ACTIVITY RESULT - High confidence - Your confidence level is higher than normal"

            def get_sport_description(score):
                if score < 0.4:
                    return "SPORT RESULT - Not sportif - You are not sportif"
                elif 0.4 <= score <= 0.6:
                    return "SPORT RESULT - Moderate sportif - You have moderate sportif traits"
                else:
                    return "SPORT RESULT - Very sportif - You are very sportif"

            def get_competence_description(score):
                if score < 0.4:
                    return "COMPETENCE RESULT - Low competence - Your competence level is lower than normal"
                elif 0.4 <= score <= 0.6:
                    return "COMPETENCE RESULT - Normal competence - You have normal competence"
                else:
                    return "COMPETENCE RESULT - High competence - You have high competence"

            def get_forgiveness_description(score):
                if score < 0.4:
                    return "FORGIVENESS RESULT - Not very tolerant - You are not very tolerant"
                elif 0.4 <= score <= 0.6:
                    return "FORGIVENESS RESULT - Moderately tolerant - You have moderate tolerance"
                else:
                    return "FORGIVENESS RESULT - Very tolerant - You are very tolerant"

            def get_self_reliance_description(score):
                if score < 0.4:
                    return "SELF-RELIANCE RESULT - Not very self-reliant - You are not very self-reliant"
                elif 0.4 <= score <= 0.6:
                    return "SELF-RELIANCE RESULT - Moderately self-reliant - You have moderate self-reliance"
                else:
                    return "SELF-RELIANCE RESULT - Very self-reliant - You are very self-reliant"

            def get_generosity_description(score):
                if score < 0.4:
                    return "GENEROSITY RESULT - Not very generous - You are not very generous"
                elif 0.4 <= score <= 0.6:
                    return "GENEROSITY RESULT - Moderately generous - You have moderate generosity"
                else:
                    return "GENEROSITY RESULT - Very generous - You are very generous"

            # Print the results
            print("SENTIMENTS RESULT")
            print(get_sentiment_description(predictions[0]))

            print("MENTAL ACTIVITY RESULT")
            print(get_mental_activity_description(predictions[1]))

            print("SPORT RESULT")
            print(get_sport_description(predictions[2]))

            print("COMPETENCE RESULT")
            print(get_competence_description(predictions[3]))

            print("FORGIVENESS RESULT")
            print(get_forgiveness_description(predictions[4]))

            print("SELF-RELIANCE RESULT")
            print(get_self_reliance_description(predictions[5]))

            print("GENEROSITY RESULT")
            print(get_generosity_description(predictions[6]))

            output = [get_sentiment_description(predictions[0]), get_mental_activity_description(predictions[1]), get_sport_description(predictions[2]), get_competence_description(predictions[3]), get_forgiveness_description(predictions[4]), get_self_reliance_description(predictions[5]), get_generosity_description(predictions[6])]
            return output
        else:
            output = 'none'
            return output
    else:
        output = 'none'
        return output