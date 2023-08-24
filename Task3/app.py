import gradio as gr
import pandas as pd
import pickle

def predict_Sale(TV, Radio, Newspaper):
    try:
        # Load model
        pickled_model = pickle.load(open('Sales Prediction.pkl', 'rb'))
    except Exception as e:
        return "Error: Model could not be loaded."

    # Create a dict array from the parameters
    data = {
        'TV': [float(TV)],
        'Radio': [float(Radio)],
        'Newspaper': [float(Newspaper)],
    }
    Xinp = pd.DataFrame(data)

    # Predict the Sale
    Sale = pickled_model.predict(Xinp)

    # Return the prediction
    return str(Sale[0])

# Create the gradio interface
ui = gr.Interface(
    fn=predict_Sale,
    inputs=[
        gr.inputs.Textbox(placeholder='TV', default="230.1", numeric=True, label='TV'),
        gr.inputs.Textbox(placeholder='Radio', default="37.8", numeric=True, label='Radio'),
        gr.inputs.Textbox(placeholder='Newspaper', default="69.2", numeric=True, label='Newspaper'),
    ],
    outputs="text",
    # Enable live updates without requiring a restart
    title="Sales Predictor",
    examples=[
        ['230.1', '37.8', '69.2'],
        ['283.6', '42.0', '66.2'],
    ],
)

if __name__ == "__main__":
    ui.launch(share=False)
