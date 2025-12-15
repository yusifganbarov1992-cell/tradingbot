import re

with open('trading_bot.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove backslash before triple quotes
content = re.sub(r'\\"\\"\\"', '"""', content)

# Remove double backslash-n
content = re.sub(r'\\\\n', r'\\n', content)

with open('trading_bot.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed!")
