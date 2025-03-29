from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
import openai

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

SIMILARITY_THRESHOLD = 0.4

# Initialize OpenAI and model
openai.api_key = settings.OPENAI_API_KEY
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(settings.PINECONE_INDEX_NAME)
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed(text):
    return model.encode(text).tolist()

@api_view(['POST'])
def generate_suggestion(request):
    query = request.data.get('query')
    if not query:
        return Response({'error': 'Query is required.'}, status=400)

    vector = embed(query)
    try:
        results = index.query(vector=vector, top_k=5, include_metadata=True)
    except Exception as e:
        return Response({'error': f'Pinecone query failed: {str(e)}'}, status=500)

    # Filter matches above threshold
    filtered_matches = [m for m in results.get('matches', []) if m.get('score', 0) >= SIMILARITY_THRESHOLD]

    if filtered_matches:
        context_snippets = ""
        for match in filtered_matches:
            context = match['metadata'].get('context', '')
            response = match['metadata'].get('response', '')
            context_snippets += f"Context: {context}\nResponse: {response}\n\n"

        prompt = f"""You are a helpful mental health assistant.

User's challenge:
{query}

Here are some relevant past context-response examples:
{context_snippets}

Based on the above, suggest the best way to help the patient.
"""
    else:
        prompt = f"""You are a helpful mental health assistant.

User's challenge:
{query}

There were no matching past counseling conversations. Please generate the best suggestion possible based solely on your training and best practices.
"""

    try:
        chat_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert mental health advisor."},
                {"role": "user", "content": prompt}
            ]
        )
        suggestion = chat_response['choices'][0]['message']['content']
    except Exception as e:
        return Response({'error': f'OpenAI error: {str(e)}'}, status=500)

    return Response({
        "query": query,
        "match_count": len(filtered_matches),
        "suggestion": suggestion,
        "note": "No relevant matches were found, so output generated from LLM." if not filtered_matches else "Generated using top matching examples in the database."
    })





# Below is the code without threshold implementation


# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from django.conf import settings
# from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer
# import openai

# # Initialize APIs
# openai.api_key = settings.OPENAI_API_KEY
# pc = Pinecone(api_key=settings.PINECONE_API_KEY)
# index = pc.Index(settings.PINECONE_INDEX_NAME)
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def embed(text):
#     return model.encode(text).tolist()

# @api_view(['POST'])
# def generate_suggestion(request):
#     query = request.data.get('query')
#     if not query:
#         return Response({'error': 'Query is required.'}, status=400)

#     vector = embed(query)
#     results = index.query(vector=vector, top_k=5, include_metadata=True)

#     context_snippets = ""
#     for match in results.get('matches', []):
#         context = match['metadata'].get('context', '')
#         response = match['metadata'].get('response', '')
#         context_snippets += f"Context: {context}\nResponse: {response}\n\n"

#     prompt = f"""You are a helpful mental health assistant.

# User's challenge:
# {query}

# Here are 5 similar context-response examples:
# {context_snippets}

# Based on the above, suggest the best possible way to help the patient.
# """

#     try:
#         chat_response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are an expert mental health advisor."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         suggestion = chat_response.choices[0].message['content']
#     except Exception as e:
#         return Response({"error": str(e)}, status=500)

#     return Response({"query": query, "suggestion": suggestion})

