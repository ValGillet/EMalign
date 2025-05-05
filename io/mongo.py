from pymongo import MongoClient

def check_progress(arguments, db_host, db_name, collection_name):
    
    client = MongoClient(db_host)
    db = client[db_name]
    progress_collection = db[collection_name]
    return progress_collection.count_documents(arguments) >= 1