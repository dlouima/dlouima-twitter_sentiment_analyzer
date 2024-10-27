
from setup import *

# Starting the Streamlit application
def main():
    st.title("PlayMood AI Player Sentiment Analysis")

    # Path to the CSV file 
    file_path = "twitter_comment_2.csv"
    
     # create visaulization
    data = loading_dataset(file_path)
    st.session_state['data'] = data
    st.write("### Dataset Summary")
    st.write(data.describe())  
        
    # Visualize sentiment distribution
    st.write("### Dataset Sentiment Distribution Summary ")
    sentiment_distribution_plot(data)

    # Preprocess data and train the model only once
    if 'model' not in st.session_state or 'vectorizer' not in st.session_state:
        X, y, vectorizer = preprocess_data(st.session_state['data'])
        model = train_model(X, y)
        st.session_state['model'] = model
        st.session_state['vectorizer'] = vectorizer
       

    # User input for sentiment prediction
    st.write("### Predict Sentiment for New Comments")
    user_input = st.text_area("Enter a comment here for sentiment prediction:")

    # Only run prediction when the button is clicked
    if st.button("Predict Sentiment") and user_input:
        prediction = predict_sentiment(st.session_state['model'], st.session_state['vectorizer'], [user_input])
        st.write(f"The Predicted Sentiment is : **{prediction[0]}**")
        

if __name__ == '__main__':
    main()