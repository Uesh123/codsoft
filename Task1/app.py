import gradio as gr
import pandas as pd
import numpy as np
from joblib import load

def predict_Survived(
         Pclass , Sex , Age , SibSp , Parch , Fare , Embarked              
):
    
# load model
    model = load('TITANIC SURVIVAL PREDICTION.joblib')

    # Create a dict array from the parameters
    data = {
        'Pclass': [Pclass],
        'Sex': [Sex],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Fare': [Fare],
        'Embarked': [Embarked],
    }
    Xinp = pd.DataFrame(data)
    print(Xinp)

    # Predict the Survived
    Survived = model.predict(Xinp)

    # return the level
    return Survived[0]

# Create the gradio interface

ui = gr.Interface(
    fn = predict_Survived,
    inputs = [
        gr.inputs.Textbox(placeholder='Pclass', default="3", numeric=True,label='Pclass'), 
        gr.inputs.Textbox(placeholder='Sex', default="1",numeric=True,label='Sex'), 
        gr.inputs.Textbox(placeholder='Age', default="34.5",numeric=True,label='Age'),
        gr.inputs.Textbox(placeholder='SibSp', default="0",numeric=True,label='SibSp'),
        gr.inputs.Textbox(placeholder='Parch', default="0",numeric=True,label='Parch'),
        gr.inputs.Textbox(placeholder='Fare', default="7.8292",numeric=True,label='Fare'),
        gr.inputs.Textbox(placeholder='Embarked', default="0",numeric=True,label='Embarked'),
        
    ],
    outputs = "text",
   
    
)

if __name__ == "__main__":      
    ui.launch(share=False)