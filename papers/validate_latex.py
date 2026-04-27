import re
from collections import Counter

with open('f:/unified-sel/papers/boundary_local_amplification.tex', 'r', encoding='utf-8') as f:
    content = f.read()

issues = []
warnings = []

# 1. Check all citep have bibitem
citep_keys = re.findall(r'\\citep\{([^}]+)\}', content)
all_cite_keys = set()
for c in citep_keys:
    for key in c.split(','):
        all_cite_keys.add(key.strip())

bibitem_keys = re.findall(r'\\bibitem\[[^\]]*\]\{([^}]+)\}', content)
bibitem_keys_set = set(bibitem_keys)

missing = all_cite_keys - bibitem_keys_set
if missing:
    issues.append(f'Missing bibitem: {missing}')
else:
    print(f'Citations: {len(all_cite_keys)} citep keys, {len(bibitem_keys)} bibitem entries')

# 2. Check environments
envs = re.findall(r'\\begin\{([^}]+)\}', content)
end_envs = re.findall(r'\\end\{([^}]+)\}', content)
env_counts = Counter(envs)
end_counts = Counter(end_envs)
for env in set(envs + end_envs):
    if env_counts[env] != end_counts[env]:
        issues.append(f'Unbalanced environment: {env} ({env_counts[env]} begin, {end_counts[env]} end)')
print(f'Environments: {len(envs)} begin, {len(end_envs)} end')

# 3. Check for common LaTeX issues
if '$$' in content:
    issues.append('Found $$ (use \\[ \\] instead)')

# 4. Check math mode $ pairs
single_dollars = re.findall(r'(?<!\\)\$(?!\$)', content)
if len(single_dollars) % 2 != 0:
    issues.append(f'Unmatched single $ (count: {len(single_dollars)})')
else:
    print(f'Math mode: {len(single_dollars)//2} inline math pairs')

# 5. Check table column counts
tables = re.findall(r'\\begin\{tabular\}\{([^}]+)\}', content)
for i, align in enumerate(tables):
    cols = len(align.replace('@{}', '').replace('|', ''))
    print(f'Table {i+1}: {cols} columns (alignment: {align})')

# 6. Check for required packages
required = ['amsmath', 'amssymb', 'graphicx', 'booktabs', 'hyperref', 'natbib', 'geometry', 'algorithm', 'algpseudocode', 'xcolor', 'caption', 'subcaption']
for pkg in required:
    # Handle comma-separated packages like \usepackage{amsmath,amssymb}
    if re.search(rf'\\usepackage\{{[^}}]*{pkg}[^}}]*\}}', content):
        print(f'Package {pkg}: OK')
    else:
        warnings.append(f'Missing package: {pkg}')

# 7. Check sections
sections = re.findall(r'\\section\{([^}]+)\}', content)
print(f'Sections: {len(sections)}')
for s in sections:
    print(f'  - {s}')

# 8. Check abstract length
abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
if abstract_match:
    abstract = abstract_match.group(1).strip()
    words = len(abstract.split())
    print(f'Abstract: {words} words')
    if words > 250:
        warnings.append(f'Abstract long: {words} words')

# 9. Check figure/table references
fig_refs = len(re.findall(r'\\ref\{fig:', content))
tab_refs = len(re.findall(r'\\ref\{tab:', content))
print(f'Figure references: {fig_refs}')
print(f'Table references: {tab_refs}')

# 10. Check for empty author
if '\\author{Anonymous}' in content:
    print('Author: Anonymous (OK for double-blind)')
elif '\\author{}' in content:
    issues.append('Author field is empty')

print()
if issues:
    print('ERRORS:')
    for i in issues:
        print(f'  [x] {i}')
else:
    print('No errors found.')

if warnings:
    print('WARNINGS:')
    for w in warnings:
        print(f'  [!] {w}')
else:
    print('No warnings.')
