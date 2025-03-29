from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

SIMILARITY_THRESHOLD = 0.4  # Set your cutoff score

# Setup
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(settings.PINECONE_INDEX_NAME)
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed(text):
    return model.encode(text).tolist()

@api_view(['POST'])
def get_top_matches(request):
    query = request.data.get('query')
    if not query:
        return Response({'error': 'Query is required.'}, status=400)

    vector = embed(query)
    results = index.query(vector=vector, top_k=5, include_metadata=True)

    # Filter based on cosine similarity threshold
    matches = []
    for match in results.get('matches', []):
        if match.get('score', 0) >= SIMILARITY_THRESHOLD:
            matches.append({
                "context": match['metadata'].get('context', ''),
                "response": match['metadata'].get('response', ''),
                "score": match.get('score')
            })


    # Return error if no match is found above threshold
    if not matches:
        return Response({"message": "No relevant matches found."}, status=200)

    return Response({
        "query": query,
        "matches": matches,
        "match_count": len(matches),
        "note": f"Filtered matches with score >= {SIMILARITY_THRESHOLD}"
    })








# Below is the implementation without cosine score threshold

# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from django.conf import settings
# from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer

# # Setup Pinecone and embedding model
# pc = Pinecone(api_key=settings.PINECONE_API_KEY)
# index = pc.Index(settings.PINECONE_INDEX_NAME)
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def embed(text):
#     return model.encode(text).tolist()

# @api_view(['POST'])
# def get_top_matches(request):
#     query = request.data.get('query')
#     if not query:
#         return Response({'error': 'Query is required.'}, status=400)

#     vector = embed(query)
#     results = index.query(vector=vector, top_k=5, include_metadata=True)

#     matches = [
#         {
#             "context": match['metadata'].get('context', ''),
#             "response": match['metadata'].get('response', ''),
#             "score": match.get('score')
#         }
#         for match in results.get('matches', [])
#     ]

#     return Response({"query": query, "matches": matches})
