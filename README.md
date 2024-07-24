# app.py

"""
YouTube Transcript Search

This application allows users to search for specific content within YouTube video transcripts
using natural language queries. It uses advanced NLP techniques to find the most relevant
segments of videos based on the user's search query.

# Features

- Search YouTube videos based on transcript content
- Natural language query processing
- Relevance scoring of search results
- Display of video thumbnails and relevant transcript segments

# Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/youtube-transcript-search.git
   cd youtube-transcript-search
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install flask google-api-python-client youtube_transcript_api nltk scikit-learn numpy
   ```

4. Set up your YouTube API key:
   - Go to the Google Developers Console (https://console.developers.google.com/)
   - Create a new project and enable the YouTube Data API v3
   - Create credentials (API Key)
   - Replace "YOUR_YOUTUBE_API_KEY" in the code below with your actual API key

# Usage

1. Save this entire file as `app.py`

2. Run the Flask application:
   ```
   python app.py
   ```

3. Open a web browser and go to `http://127.0.0.1:5000`

4. Enter a search query and submit to see the results

# Code
"""

import os
import logging
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from flask import Flask, request, jsonify, render_template_string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

# YouTube API setup
DEVELOPER_KEY = "YOUR_YOUTUBE_API_KEY"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

# NLP preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def get_noun_phrases(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    noun_phrases = []
    current_phrase = []
    for word, tag in pos_tags:
        if tag.startswith('N'):
            current_phrase.append(word)
        elif current_phrase:
            noun_phrases.append(' '.join(current_phrase))
            current_phrase = []
    if current_phrase:
        noun_phrases.append(' '.join(current_phrase))
    return noun_phrases

def search_transcripts(query, transcripts):
    logger.debug(f"Searching transcripts for query: {query}")
    logger.debug(f"Number of transcripts: {len(transcripts)}")
    
    preprocessed_query = preprocess_text(query)
    noun_phrases = get_noun_phrases(query)
    logger.debug(f"Preprocessed query: {preprocessed_query}")
    logger.debug(f"Noun phrases: {noun_phrases}")
    
    results = []
    for video_id, transcript in transcripts.items():
        logger.debug(f"Processing video: {video_id}")
        # Combine consecutive transcript segments
        combined_segments = []
        current_segment = {'text': '', 'start': 0}
        for segment in transcript:
            if len(current_segment['text'].split()) < 50:  # Combine up to ~50 words
                if not current_segment['text']:
                    current_segment['start'] = segment['start']
                current_segment['text'] += ' ' + segment['text']
            else:
                combined_segments.append(current_segment)
                current_segment = {'text': segment['text'], 'start': segment['start']}
        if current_segment['text']:
            combined_segments.append(current_segment)
        
        logger.debug(f"Number of combined segments: {len(combined_segments)}")
        
        # Preprocess transcript segments
        preprocessed_segments = [preprocess_text(segment['text']) for segment in combined_segments]
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_segments + [preprocessed_query])
        
        # Compute cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        
        logger.debug(f"Max cosine similarity: {np.max(cosine_similarities)}")
        
        # Boost scores for segments containing noun phrases
        for i, segment in enumerate(combined_segments):
            for phrase in noun_phrases:
                if phrase.lower() in segment['text'].lower():
                    cosine_similarities[i] += 0.1  # Boost score
        
        # Get top 3 matches for this video
        top_indices = np.argsort(cosine_similarities)[-3:][::-1]
        for index in top_indices:
            if cosine_similarities[index] > 0.1:  # Threshold to ensure relevance
                results.append({
                    'video_id': video_id,
                    'start_time': combined_segments[index]['start'],
                    'text': combined_segments[index]['text'],
                    'score': float(cosine_similarities[index])
                })
    
    logger.debug(f"Number of results before sorting: {len(results)}")
    
    # Sort results by score
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:5]  # Return top 5 overall results

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    logger.info(f"Received search query: {query}")
    
    try:
        # Search for videos
        search_response = youtube.search().list(
            q=query,
            type="video",
            part="id,snippet",
            maxResults=5
        ).execute()

        video_ids = [item['id']['videoId'] for item in search_response['items']]
        logger.debug(f"Found video IDs: {video_ids}")
        
        # Fetch transcripts
        transcripts = {}
        for video_id in video_ids:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                transcripts[video_id] = transcript
                logger.debug(f"Successfully fetched transcript for video {video_id}")
            except Exception as e:
                logger.error(f"Error fetching transcript for video {video_id}: {str(e)}")
        
        logger.debug(f"Number of transcripts fetched: {len(transcripts)}")
        
        # Search transcripts
        results = search_transcripts(query, transcripts)
        
        logger.debug(f"Number of results after search: {len(results)}")
        
        # Add video details to results
        for result in results:
            video_details = next((item for item in search_response['items'] if item['id']['videoId'] == result['video_id']), None)
            if video_details:
                result['title'] = video_details['snippet']['title']
                result['thumbnail'] = video_details['snippet']['thumbnails']['default']['url']
        
        logger.info(f"Returning {len(results)} results")
        return jsonify(results)
    
    except Exception as e:
        logger.exception("An error occurred during the search process")
        return jsonify({"error": str(e)}), 500

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Transcript Search</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        #results, #debug { margin-top: 20px; }
        .result { margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; display: flex; }
        .thumbnail { margin-right: 10px; }
        .content { flex-grow: 1; }
        .title { font-weight: bold; }
        .score { color: #666; font-size: 0.9em; }
        #debug { background-color: #f0f0f0; padding: 10px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>YouTube Transcript Search</h1>
    <form id="searchForm">
        <input type="text" id="query" name="query" required>
        <button type="submit">Search</button>
    </form>
    <div id="results"></div>
    <div id="debug"></div>

    <script>
        $(document).ready(function() {
            $('#searchForm').submit(function(e) {
                e.preventDefault();
                $('#debug').text('Sending request...');
                $.ajax({
                    url: '/search',
                    method: 'POST',
                    data: $(this).serialize(),
                    success: function(data) {
                        $('#debug').text('Response received: ' + JSON.stringify(data, null, 2));
                        $('#results').empty();
                        if (data.length === 0) {
                            $('#results').append('<p>No results found.</p>');
                        } else {
                            data.forEach(function(result) {
                                $('#results').append(
                                    '<div class="result">' +
                                    '<div class="thumbnail"><img src="' + result.thumbnail + '" alt="Video thumbnail"></div>' +
                                    '<div class="content">' +
                                    '<div class="title">' + result.title + '</div>' +
                                    '<p>Start time: ' + result.start_time.toFixed(2) + 's</p>' +
                                    '<p>' + result.text + '</p>' +
                                    '<p class="score">Relevance score: ' + result.score.toFixed(2) + '</p>' +
                                    '<a href="https://www.youtube.com/watch?v=' + result.video_id + '&t=' + Math.floor(result.start_time) + '" target="_blank">Watch Video</a>' +
                                    '</div></div>'
                                );
                            });
                        }
                    },
                    error: function(jqXHR, textStatus, errorThrown) {
                        $('#debug').text('Error: ' + textStatus + '\n' + errorThrown + '\n' + jqXHR.responseText);
                    }
                });
            });
        });
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
