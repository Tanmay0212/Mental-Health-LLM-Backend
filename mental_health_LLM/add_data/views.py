from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from django.conf import settings

import pandas as pd
import uuid
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone and model
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index_name = settings.PINECONE_INDEX_NAME

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )

index = pc.Index(index_name)
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed(text):
    return model.encode(text).tolist()

@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_csv_and_add_to_pinecone(request):
    csv_file = request.FILES.get('file')
    if not csv_file:
        return Response({'error': 'CSV file is required.'}, status=400)

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        return Response({'error': f'Failed to read CSV: {str(e)}'}, status=400)

    if 'context' not in df.columns or 'response' not in df.columns:
        return Response({'error': 'CSV must have \"context\" and \"response\" columns.'}, status=400)

    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    upserts = []
    for _, row in df.iterrows():
        context = str(row['context'])
        response = str(row['response'])
        vector = embed(context)
        uid = str(uuid.uuid4())
        upserts.append({
            "id": uid,
            "values": vector,
            "metadata": {
                "context": context,
                "response": response
            }
        })

    BATCH_SIZE = 100
    for chunk in chunk_list(upserts, BATCH_SIZE):
        index.upsert(vectors=chunk)

    return Response({"message": f"Uploaded and inserted {len(upserts)} entries into Pinecone."})




















## Previous logic 


# from rest_framework.decorators import api_view, parser_classes
# from rest_framework.parsers import MultiPartParser
# from rest_framework.response import Response
# from django.conf import settings
# import pandas as pd
# import uuid
# from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer

# # Initialize Pinecone and model
# pc = Pinecone(api_key=settings.PINECONE_API_KEY)
# index_name = settings.PINECONE_INDEX_NAME

# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=384,
#         metric="cosine",
#         spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
#     )

# index = pc.Index(index_name)
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def embed(text):
#     return model.encode(text).tolist()

# @api_view(['POST'])
# @parser_classes([MultiPartParser])
# def upload_csv_and_add_to_pinecone(request):
#     csv_file = request.FILES.get('file')
#     if not csv_file:
#         return Response({'error': 'CSV file is required.'}, status=400)

#     try:
#         df = pd.read_csv(csv_file)
#     except Exception as e:
#         return Response({'error': f'Failed to read CSV: {str(e)}'}, status=400)

#     # Check required columns
#     if 'context' not in df.columns or 'response' not in df.columns:
#         return Response({'error': 'CSV must have "context" and "response" columns.'}, status=400)

#     upserts = []
#     for _, row in df.iterrows():
#         context = str(row['context'])
#         response = str(row['response'])
#         vector = embed(context)
#         uid = str(uuid.uuid4())
#         upserts.append({
#             "id": uid,
#             "values": vector,
#             "metadata": {
#                 "context": context,
#                 "response": response
#             }
#         })

#     index.upsert(vectors=upserts)

#     return Response({"message": f"Uploaded and inserted {len(upserts)} entries into Pinecone."})
