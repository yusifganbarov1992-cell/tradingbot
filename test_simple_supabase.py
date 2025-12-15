"""Простой тест прямого подключения"""
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_KEY')

print(f"URL: {url}")
print(f"KEY length: {len(key)}")

client = create_client(url, key)
print("✅ Client created")

# Попробовать простой запрос
try:
    response = client.table('signals').select("*").limit(1).execute()
    print(f"✅ Query successful! Data: {response.data}")
except Exception as e:
    print(f"❌ Query failed: {e}")
