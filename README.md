# Emusify

**Emusify is a real-time mood-based music recommendation system that runs in the background and plays music according to a user's mood.**

<img src="https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/002/038/759/datas/gallery.jpg">

## 💡 Inspiration

In this era, music platforms provide easy access to large amounts of music. They are working continuously to improve music organization and search management thereby addressing the problem of choice and simplifying exploring new music pieces. Recommendation systems gain more and more popularity and help people to select appropriate music for all occasions. However, there is still a gap in personalization and emotion-driven recommendations. Music has a great influence on humans and is widely used for relaxing, mood regulation, destruction from stress and diseases, to
maintain mental and physical work. A recommendation system is targeted to help people with music selection for different life situations and maintain their mental and physical conditions.

## ⚙️ What it does?

This system will continuously run in background and calculate and predict the mood of the user for a stipulated time period and will play a song according to the mood of the user, for example, if the user is in a calm state or neutral state then a calming, meditation song will be played, script will be continuously running in background so if any emotion changes after 5-6 min then again a new song will be played.

## 🔧 How we built it?

**Technology Stack**

 ● OpenCV 

● Machine Learning

Emusify is a real-time mood-based music recommendation system using machine learning and keras dataset.

This project has three main parts:

❖ Data Collection 

❖ Data Training

❖ Data Testing

This project uses Mediapipe which will return all the landmark key
points. Mediapipe Holistic is one of the pipelines which contains optimized
face, hands, and pose components that allow for holistic tracking, thus
enabling the model to simultaneously detect hand and body poses along
with face landmarks.
We have to run the data collection script every time for all the emotions and
in this project, I have collected approx 1000 images of 6 particular
emotions as well as keras emotion dataset.

After collecting all data we have to load the data then send it for the dense
neural network training part.

I have loaded the model and labels in the testing script.
Finally, the prediction of emotion can be seen on the screen according to
the emotion made by the user.
Now part comes for the prediction of songs on the basis of the emotions of
the user, so for this, we need to create a local directory of songs.

Inside the song folder, we need to add folders of emotion and 5-6 songs
with respect to that emotion inside that folder.

<img src="https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/002/039/019/datas/gallery.jpg" height=200>

<img src="https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/002/039/050/datas/gallery.jpg"  height=200>

Whenever the user comes in front of the webcam then we will add a time of
approx 15-20 sec to calculate the emotions of the user.
This is going to work like we will create an array of the particular emotion
and whichever emotion is dominant, the value of that particular emotion
counter will increase and a respective song will be played.

**For example, if the user looks sad for a prolonged period of time then a happy song will be played to cheer up the user. 12 After the first song is played then the script will be running in the background and will be calculating the emotions for like 5-6 minutes then again a song will be played according to the most dominant emotion.**

## Challenges we ran into 🙁

I ran into challenges during data collection, capturing 990 images of a particular emotion is kind of hectic task and while doing it sometimes a sad emotion changed to happy lol.

## Accomplishments that we're proud of 😇

We are proud that in this short period of time we were able to make a recommendation system and that to working very efficiently.

We can also edit the code for capturing multiple people's emotions and collaborative song recommendations on the basis of multiple people's emotions. 

## What we learned 🤔

I learned a lot about new modules and packages of python like MediaPipe Selfie Segmentation for removal of a human face from the background and also the Keras library. 

## What's next for Emusify 📲

This project is only suitable for the people sitting in front of a computer's webcam at a distance of 1 meter and if the distance increases more then the prediction accuracy is reduced such as neutral and sad emotions are not captured well and the prediction becomes wrong. 

To improve this we need to have a high-definition camera as well as a proper training data set of approx 6000-10000 images of different people making different emotions. We can also try different models for training and testing the data.
