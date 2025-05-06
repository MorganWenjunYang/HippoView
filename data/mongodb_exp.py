from utils import fetch_document_by_nct_id, fetch_documents_by_sponsor_name

if __name__ == "__main__":
    document=fetch_document_by_nct_id("NCT01724996")
    print(document)

    documents=fetch_documents_by_sponsor_name("AstraZeneca")
    print(documents[1])