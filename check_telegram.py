import requests

TOKEN = "8193800020:AAEbM9jKBiKhCifVOGcvsavSqEDZ0K77tAs"

# Delete webhook and CLEAR pending updates
response = requests.post(f"https://api.telegram.org/bot{TOKEN}/deleteWebhook?drop_pending_updates=true")
print(f"Delete webhook + clear updates: {response.json()}")

# Get webhook info
response = requests.get(f"https://api.telegram.org/bot{TOKEN}/getWebhookInfo")
print(f"\nWebhook info: {response.json()}")

# Get pending updates count
response = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates?offset=-1")
updates = response.json()
print(f"\nPending updates: {updates}")

if updates.get('ok') and len(updates.get('result', [])) > 0:
    # Clear all pending updates by getting them all
    last_update_id = updates['result'][-1]['update_id']
    response = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates?offset={last_update_id + 1}")
    print(f"\nCleared updates: {response.json()}")

# Get bot info
response = requests.get(f"https://api.telegram.org/bot{TOKEN}/getMe")
print(f"\nBot info: {response.json()}")
