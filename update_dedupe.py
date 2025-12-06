#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Update app.py to make deduplication conditional"""

# Read the file
with open(r'd:\Python codes\Face recognition\app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the section to replace (around line 444)
output_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Check if we're at the start of the deduplication section
    if i == 443 and '# Search using FAISS' in line:
        # Add the modified section
        output_lines.append(line)  # Keep "# Search using FAISS"
        output_lines.append(lines[i+1])  # Keep "results = index_manager.search(...)"
        output_lines.append('\n')
        output_lines.append('            # Apply deduplication if enabled\n')
        output_lines.append('            if remove_duplicates:\n')
        output_lines.append('                print(f"\\n{\'=\'*60}")\n')
        output_lines.append('                print(f"🔍 DEDUPLICATION ENABLED")\n')
        output_lines.append('                print(f"   Total results from FAISS: {len(results)}")\n')
        output_lines.append('                print(f"{\'=\'*60}\\n")\n')
        output_lines.append('\n')
        output_lines.append('                deduplicated_results, duplicates_removed, duplicates_log = deduplicate_faiss_results(\n')
        output_lines.append('                    results,\n')
        output_lines.append('                    file_paths,\n')
        output_lines.append('                    duplicate_threshold=0.97\n')
        output_lines.append('                )\n')
        output_lines.append('\n')
        output_lines.append('                print(f"📊 Deduplication: {len(results)} results → {len(deduplicated_results)} unique ({duplicates_removed} duplicates removed)")\n')
        output_lines.append('                for duplicate in duplicates_log[:5]:\n')
        output_lines.append('                    print(f"   • {duplicate[\'path\']} → {duplicate[\'reason\']}")\n')
        output_lines.append('\n')
        output_lines.append('                yield f"data: {json.dumps({\'type\': \'info\', \'message\': f\'📊 Deduplication removed {duplicates_removed} duplicates\'})}\\n\\n"\n')
        output_lines.append('                \n')
        output_lines.append('                final_results = deduplicated_results\n')
        output_lines.append('            else:\n')
        output_lines.append('                print(f"\\n🔍 DEDUPLICATION DISABLED - Showing all {len(results)} results\\n")\n')
        output_lines.append('                final_results = results\n')
        output_lines.append('            \n')
        output_lines.append('            # Stream results\n')
        output_lines.append('            for idx, similarity in final_results:\n')
        output_lines.append('                match = {\n')
        output_lines.append('                    \'path\': file_paths[int(idx)],\n')
        output_lines.append('                    \'similarity\': float(similarity * 100)  # Convert back to 0-100\n')
        output_lines.append('                }\n')
        output_lines.append('                yield f"data: {json.dumps({\'type\': \'match\', \'match\': match})}\\n\\n"\n')
        output_lines.append('            \n')
        output_lines.append('            yield f"data: {json.dumps({\'type\': \'complete\', \'total_processed\': len(file_paths), \'matches\': len(final_results)})}\\n\\n"\n')
        
        # Skip the old lines (from i+2 to around i+30)
        i += 31  # Skip to after the old code
        continue
    
    output_lines.append(line)
    i += 1

# Write the updated file
with open(r'd:\Python codes\Face recognition\app.py', 'w', encoding='utf-8') as f:
    f.writelines(output_lines)

print('✅ Successfully updated app.py with conditional deduplication!')
