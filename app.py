from flask import Flask, render_template, request,make_response,jsonify
from flask_bootstrap import Bootstrap
import spacy
from collections import Counter
import random
import PyPDF2
from PyPDF2 import PdfReader,PdfWriter  # Import PdfReader

import os 
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings,HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from transformers import pipeline


app = Flask(__name__, template_folder='templates1') 

Bootstrap(app)

# Global variable to store uploaded PDF path
uploaded_pdf_path = None

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")


def generate_mcqs(text, num_questions=5):
    # text = clean_text(text)
    if text is None:
        return []

    # Process the text with spaCy
    doc = nlp(text)

    # Extract sentences from the text
    sentences = [sent.text for sent in doc.sents]

    # Ensure that the number of questions does not exceed the number of sentences
    num_questions = min(num_questions, len(sentences))

    # Randomly select sentences to form questions
    selected_sentences = random.sample(sentences, num_questions)

    # Initialize list to store generated MCQs
    mcqs = []

    # Generate MCQs for each selected sentence
    for sentence in selected_sentences:
        # Process the sentence with spaCy
        sent_doc = nlp(sentence)

        # Extract entities (nouns) from the sentence
        nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]

        # Ensure there are enough nouns to generate MCQs
        if len(nouns) < 2:
            continue

        # Count the occurrence of each noun
        noun_counts = Counter(nouns)

        # Select the most common noun as the subject of the question
        if noun_counts:
            subject = noun_counts.most_common(1)[0][0]

            # Generate the question stem
            question_stem = sentence.replace(subject, "______")

            # Generate answer choices
            answer_choices = [subject]

            # Add some random words from the text as distractors
            distractors = list(set(nouns) - {subject})

            # Ensure there are at least three distractors
            while len(distractors) < 3:
                distractors.append("[Distractor]")  # Placeholder for missing distractors

            random.shuffle(distractors)
            for distractor in distractors[:3]:
                answer_choices.append(distractor)

            # Shuffle the answer choices
            random.shuffle(answer_choices)

            # Append the generated MCQ to the list
            correct_answer = chr(64 + answer_choices.index(subject) + 1)  # Convert index to letter
            mcqs.append((question_stem, answer_choices, correct_answer))

    return mcqs

def process_pdf(file):
    # Initialize an empty string to store the extracted text
    text = ""

    # Create a PyPDF2 PdfReader object
    pdf_reader = PdfReader(file)

    # Loop through each page of the PDF
    for page_num in range(len(pdf_reader.pages)):
        # Extract text from the current page
        page_text = pdf_reader.pages[page_num].extract_text()
        # Append the extracted text to the overall text
        text += page_text

    return text

"""RAG"""
def rag(text,question):

    load_dotenv()
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    retriver = vector_store.as_retriever(search_type ="similarity",search_kwargs={"k": 4})

    prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
    )

    retrived_docs = retriver.invoke(question)

    context_text = "\n\n".join(doc.page_content for doc in retrived_docs)

    final_prompt = prompt.invoke({"context": context_text,"question":question})

    llm1 = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100}
    )
    model=ChatHuggingFace(llm=llm1)
    ans = model.invoke(final_prompt)

    def final_context_text(retrived_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrived_docs)
        return context_text
    
    parallel_chain = RunnableParallel({
    'context': retriver|RunnableLambda(final_context_text),
    'question': RunnablePassthrough()
    })

    parser =StrOutputParser()
    final_chain= parallel_chain|prompt|model|parser
    ans = final_chain.invoke('question')

    #summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    #ans=summarizer(paragraph, max_length=1000, min_length=300, do_sample=False)
    
    return ans
    







@app.route('/')
def index():
    return render_template('index.html')


@app.route('/mcq', methods=['GET', 'POST'])
def mcq():
    if request.method == 'POST':
        text = ""

        # Check if files were uploaded
        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file.filename.endswith('.pdf'):
                    # Process PDF file
                    text += process_pdf(file)
                elif file.filename.endswith('.txt'):
                    # Process text file
                    text += file.read().decode('utf-8')
        else:
            # Process manual input
            text = request.form['text']

        # Get the selected number of questions from the dropdown menu
        num_questions = int(request.form['num_questions'])

        mcqs = generate_mcqs(text, num_questions=num_questions)  # Pass the selected number of questions
        print(mcqs)
        # Ensure each MCQ is formatted correctly as (question_stem, answer_choices, correct_answer)
        mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]
        return render_template('mcqs.html', mcqs=mcqs_with_index)
    
    return render_template('mcq.html')






@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global uploaded_pdf_text
    if request.method == 'POST':
        uploaded_pdf_text = ""
        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file.filename.endswith('.pdf'):
                    uploaded_pdf_text += process_pdf(file)
                elif file.filename.endswith('.txt'):
                    uploaded_pdf_text += file.read().decode('utf-8')
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    global uploaded_pdf_text
    question = request.form.get('user_query', '')
    print("User asked:", question)
    answer = rag(uploaded_pdf_text, question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)