"""
Скрипт для обновления Supabase ключа в .env
"""
import os

print("🔑 Обновление Supabase ключа")
print("")
print("1. Открой: https://supabase.com/dashboard/project/ixovpisndoyhsaaqlypl/settings/api")
print("2. Найди 'service_role' → 'secret' ключ")
print("3. Нажми на иконку копирования 📋")
print("4. Вставь ключ сюда (Ctrl+V и Enter):")
print("")

new_key = input("Service Role Key: ").strip()

if len(new_key) < 100:
    print(f"❌ Ключ слишком короткий ({len(new_key)} символов)")
    print("   Должен быть ~200+ символов, начинаться с 'eyJ'")
    exit(1)

# Читаем .env
with open('.env', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Обновляем ключ
updated = False
for i, line in enumerate(lines):
    if line.startswith('SUPABASE_SERVICE_KEY='):
        lines[i] = f'SUPABASE_SERVICE_KEY={new_key}\n'
        updated = True
        break

if not updated:
    print("❌ SUPABASE_SERVICE_KEY не найден в .env")
    exit(1)

# Сохраняем .env
with open('.env', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"✅ Ключ обновлён! Длина: {len(new_key)} символов")
print("")
print("Тестирую подключение...")

# Тестируем
from dotenv import load_dotenv
load_dotenv()
from database_supabase import SupabaseDatabase

try:
    db = SupabaseDatabase()
    print("✅ Supabase подключена!")
    
    # Простой тест
    open_trades = db.get_open_trades()
    print(f"✅ Запрос успешен! Открытых сделок: {len(open_trades)}")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
