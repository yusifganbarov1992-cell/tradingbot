# КАК ПОЛУЧИТЬ ПРАВИЛЬНЫЕ КЛЮЧИ SUPABASE

## 📍 Твой проект Supabase:
https://supabase.com/dashboard/project/ixovpisndoyhsaaqlypl

## 🔑 Где взять ключи API:

1. Открой: **https://supabase.com/dashboard/project/ixovpisndoyhsaaqlypl/settings/api**

2. На странице будут **API Settings**:

   **Project URL:**
   ```
   https://ixovpisndoyhsaaqlypl.supabase.co
   ```

   **anon / public key** (длинный JWT ~200+ символов):
   ```
   eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3M...ПОЛНЫЙ_ТОКЕН...
   ```

   **service_role / secret key** (длинный JWT ~200+ символов):
   ```
   eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3M...ПОЛНЫЙ_ТОКЕН...
   ```

## ⚠️ ВАЖНО:

Ключи которые ты дал:
- `sb_secret_cahDh2U6xR9BkXbrFfPfgA_tlZk649T` - это НЕ API ключ
- `sb_publishable_Q-t_oQ2Cjok2OY6hNZTnLA_mmdElBbA` - это НЕ API ключ

Это какие-то другие идентификаторы.

## 📋 Что нужно:

1. Открой ссылку выше
2. Найди раздел **"Project API keys"**
3. Скопируй **ПОЛНОСТЬЮ** ключ `anon public` (начинается с `eyJ...`)
4. Скопируй **ПОЛНОСТЬЮ** ключ `service_role` (начинается с `eyJ...`)

Пришли мне ЭТИ ключи (каждый должен быть 200+ символов длиной, в формате JWT).

## 🔍 Пример правильного ключа:

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml4b3ZwaXNuZG95aHNhYXFseXBsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQ0MjIxMTcsImV4cCI6MjA0OTk5ODExN30.Q-t_oQ2Cjok2OY6hNZTnLA_mmdElBbA
```

Это JWT токен (JSON Web Token) - три части разделённые точками.
