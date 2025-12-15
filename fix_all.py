with open('trading_bot.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_lines = []
for line in lines:
    # Remove backslash before quotes in docstrings
    if '\\"' in line:
        line = line.replace('\\"', '"')
    # Fix escaped newlines in f-strings
    if '\\\\n' in line:
        line = line.replace('\\\\n', '\\n')
    fixed_lines.append(line)

with open('trading_bot.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print(f"Fixed {len(fixed_lines)} lines!")
