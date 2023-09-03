import json
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import openai
import requests

openai.api_key='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'
st.image('Image to be used on dashboard.jpg')
# Streamlit interface
st.title("Russia Ukraine War - Emotion Detection Dashboard")
st.write("This dashboard predicts the emotion behind a text based on models trained on Twitter and Reddit data.")

user_input = st.text_area("Text:")
button_click=False


if st.button("Predict") and len(user_input)>10:
    button_click=True


if button_click:
    button_click=False
    with st.spinner('Processing.....'):
        data = requests.get(f'https://us-central1-ukrainerussiawartweets.cloudfunctions.net/get_predict?text={user_input}')
        text_data = data.content.decode('utf-8')

        json_data = json.loads(text_data)
        prediction, probs = json_data['prediction'],json_data['probabilities']
        classes = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness','surprise']
        emoji_classes={"anger":" 🤬",
                "disgust":"🤢",
                "fear":"😨",
                "joy":"😀",
                "neutral":"😐",
                "sadness":"😭",
                "surprise":"😲"}
        
        # Getting sorted indices based on probabilities
        sorted_indices = np.argsort(probs)[::-1]
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_classes = [i+emoji_classes[i] for i in sorted_classes]
        sorted_probs = [probs[i] for i in sorted_indices]
        print(prediction)
        response = openai.Completion.create(
            model='text-davinci-003',
            prompt=f'''
            The emotion of a twitter/Reddit user is predicted to be {prediction} with regards to 
            Russia Vs Ukraine conflict based on sentiment analysis, using bullet points give some 
            insights as to why he could be {prediction}
            ''',
            n=1,
            max_tokens=500
        )
    st.write(f"Predicted emotion: **{prediction}**")

    prob_table = {"Emotion Class": sorted_classes, "Probability": [f"{prob:.4f}" for prob in sorted_probs]}
    st.table(prob_table)

    plt.figure(figsize=(10,5))
    plt.bar(sorted_classes, sorted_probs)
    plt.xlabel('Emotion Class')
    plt.ylabel('Probability')
    plt.title('Predicted Probabilities for each Emotion Class')
    st.pyplot(plt)
    st.header('Recommendations')
    st.success(response['choices'][0]['text'])



