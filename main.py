
from setup import *

# Starting the Streamlit application
def main():
    st.title("PlayMood AI Player Sentiment Analysis")
    
    # Path to the CSV file 
    file_path = "twitter_comment_2.csv"
    
    # create visaulization 
    data = loading_dataset(file_path)
    st.write("### Dataset Summary")
    st.write(data.describe())  

     # Visualize sentiment distribution
    st.write("### Dataset Sentiment Distribution Summary ")
    sentiment_distribution_plot(data)
       
    # Preprocess and train the model
    X, y, vectorizer = preprocess_data(data)
    model = train_model(X, y)
    
    # User input for sentiment prediction
    st.write("### Predict Sentiment for New Comments")
    user_input = st.text_area("Enter a comment here for sentiment prediction:")
    
    if st.button("Predict Sentiment"):
        if user_input:
            prediction = predict_sentiment(model, vectorizer, [user_input])
            st.write(f"The Predicted Sentiment is : **{prediction[0]}**")

if __name__ == '__main__':
    main()
