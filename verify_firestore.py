import asyncio
import os
from google.cloud import firestore

# Initialize Firestore client (ensure your GOOGLE_APPLICATION_CREDENTIALS are set)
db = firestore.AsyncClient()


async def verify_collections():
    print("--- Verifying Firestore Collections ---")

    # 1. Check 'orders' collection
    print("\n[1] Fetching documents from 'orders' collection:")
    orders_ref = db.collection("orders")
    orders_docs = await orders_ref.get()

    if not orders_docs:
        print("  -> No documents found in 'orders'.")
    else:
        for doc in orders_docs:
            data = doc.to_dict()
            print(f"  - Order ID: {doc.id}")
            print(f"    Session ID: {data.get('session_id')}")
            print(f"    Total Value: Rs. {data.get('total_value')}")
            print(f"    Payment Gateway: {data.get('payment_gateway')}")
            print(f"    Shipping: {data.get('shipping_method')}")
            print("    ------------------------------------------")

    # 2. Check 'checkout_sessions' collection
    print("\n[2] Fetching COMPLETED documents from 'checkout_sessions':")
    sessions_ref = db.collection("checkout_sessions").where("status", "==", "COMPLETED")
    sessions_docs = await sessions_ref.get()

    if not sessions_docs:
        print("  -> No completed sessions found in 'checkout_sessions'.")
    else:
        for doc in sessions_docs:
            data = doc.to_dict()
            print(f"  - Session ID: {doc.id}")
            print(f"    Status: {data.get('status')}")
            print(f"    Cart Value: Rs. {data.get('cart_value')}")
            print(f"    Metadata: {data.get('metadata')}")
            print("    ------------------------------------------")


if __name__ == "__main__":
    asyncio.run(verify_collections())