import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# New data bank, migrated to JSON for cleaner storage of data about the senior high school

json_file = "bank.json"

try:
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
except FileNotFoundError:
    st.markdown(f"Error: The file {json_file} not found.")
except json.JSONDecodeError:
    st.markdown(f"Error: Failed to decode JSON from the file '{json_file}'. Check file formatting.")

# Old Data Bank of my Senior High School
questions = [
    "Where is the senior high school located?",
    "What is the mascot of the senior high school?",
    "Who is the founder of the senior high school?",
    "When was the senior high school founded?",
    "Who is the current president?",
    "How can I contact the school?",
    "What are the enrollment requirements?",
    "Give me directions to your campus.",
    "How do I get to your Senior High School from my current location?"
]

answers = [
    "Our AU-JRC Senior High School is located at Gov. Pascual Avenue, Malabon City.",
    "Arellano University's mascot is the Chief, named in honor of Cayetano S. Arellano, the first Chief Justice of the Philippines.",
    "The AU-JRC was founded by the late Florentino Cayco Sr., the first Filipino Undersecretary of Public Instruction and illustrious educator",
    "The AU-JRC was established in the year 1950. Side note, it was previously an extension of the Elisa Esguerra Campus, which was formerly known as Gregorio Sancianco - High School Campus. The GS-HSC was closed down in the 1980s and AU developed a new campus a few city blocks south in 2017, it was renamed as the Arellano University - Elisa Esguerra Campus. The old campus was renamed as the Jose Rizal Campus after the separation.",
    "The current president of the school is Mr. Francisco Paulino V. Cayco, the CEO of Arellano University, whom oversees all campuses of AU.",
    "You can contact the school via email at hs.joserizal@arellano.edu.ph or visit their Contact Us page here: \nhttps://arellano.edu.ph/contact/jose-rizal-campus-au-malabon/",
    "To enroll, you need your previous report card (F-138), PSA birth certificate (Original Copy), 2x2 picture with white background (3 pieces), and a certificate of good moral character. You may also find more information about this here: \nhttps://www.arellano.edu.ph/basic-education/senior-high-school/",
    "You can easily navigate to our campus using Google Maps! [Click here for directions from your current location](https://www.google.com/maps/dir/?api=1&destination=Arellano+University+Jose+Rizal+Campus+Malabon).",
    "You can easily navigate to our Senior High School using Google Maps! [Click here for directions from your current location](https://www.google.com/maps/dir/?api=1&destination=Arellano+University+Jose+Rizal+Campus+Malabon)."
]

# TF-IDF Calculation approach (Term Frequency - Inverse Document Frequency)
vectorizer = TfidfVectorizer()
# Setting my questions into a mathematical matrix for the system to be able to calculate for it
question_vectors = vectorizer.fit_transform(data['questions'])

def bot_response(user_input):
    # Transformed user input into a vector to make sure that it matches the data type of the question matrix
    user_vector = vectorizer.transform([user_input])

    # Calculating the similarities between the user input vector to the questions matrix vector using cosine similarity
    # This calculation returns an array of scores between 0.0 and 1.0 to the variable
    similarities = cosine_similarity(user_vector, question_vectors)

    # This just gets the highest score, closest similarity, in the array
    closest_match_index = np.argmax(similarities)
    highest_score = similarities[0, closest_match_index]

    # Returns answer to user, using a minimum requirement of 30% to ensure some accuracy to the chatbot
    if highest_score > 0.3:
        thought_process = f"\n[System Log: Query matched with '{data['questions'][closest_match_index]}' | Confidence: {highest_score * 100:.1f}%]"
        return data['answers'][closest_match_index], thought_process
    else:
        return "I do apologize. I am only designed to answer basic queries about the Senior High School. Perhaps try rephrasing the question?", ""

# Design - used Streamlit
st.title("Arellano University - Jose Rizal Campus Chatbot")
st.write("Welcome to the AU-JRC Senior High School chatbot, Deputy Chief Cayto!")

with st.chat_message("assistant"):
    st.markdown("Hi! I'm Cayto, your Senior High School assistant from AU - JRC.")
    st.markdown("Let me know what seems to be your query!")

# Initialized the session state for messages if it doesn't exist yet
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main loop for user input and chatbot response, also to display the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input interaction
if user_query := st.chat_input("Ask Deputy Chief Cayto.."):
    # Display user query
    st.chat_message("user").markdown(user_query)
    # Add user query to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Get chatbot response
    response_text, thought_process = bot_response(user_query)

    # Combine response and thought process (if it exists) for display
    if thought_process:
        final_output = f"{response_text} \n\n {thought_process}"
    else:
        final_output = response_text

    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(final_output)
    
    # Add chatbot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_output})